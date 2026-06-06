import torch
import torch.nn.functional as F

from .registry import METHODS, register


def cfm_batch(model, x0, x1, t):
    xt = (1 - t) * x0 + t * x1
    target = x1 - x0
    vt = model(xt, t)
    return xt, target, vt


def material_derivative_jvp(model, x, t, velocity=None):
    """Compute d/ds v(x+s*velocity, t+s) at s=0 with a JVP."""
    if velocity is None:
        velocity = model(x, t)
    _, material = torch.autograd.functional.jvp(
        lambda x_in, t_in: model(x_in, t_in),
        (x, t),
        (velocity.detach(), torch.ones_like(t)),
        create_graph=True,
    )
    return material



class Method:
    min_t = 0.0

    def __init__(self, config):
        self.weight = config.get("weight", 0.0)
        self.dt = config.get("dt", 0.05)

    def sample_t(self, batch_size, device):
        return torch.rand(batch_size, 1, device=device) * (1.0 - self.min_t) + self.min_t


@register(METHODS, "standard_cfm")
class StandardCFM(Method):
    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        _, target, vt = cfm_batch(model, x0, x1, t)
        return {"cfm": F.mse_loss(vt, target)}


@register(METHODS, "lc_finite_difference")
class LagrangianConsistencyFiniteDifference(Method):
    def __init__(self, config):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)
        self.min_t = self.dt

    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        xt, target, vt = cfm_batch(model, x0, x1, t)
        x_prev = xt - vt.detach() * self.dt
        t_prev = t - self.dt
        vt_prev = model(x_prev, t_prev)
        lc = F.mse_loss(vt, vt_prev)
        return {"cfm": F.mse_loss(vt, target), "lc_fd": self.weight * lc}


@register(METHODS, "iso_fm_finite_difference")
class IsoFMFiniteDifference(Method):
    def __init__(self, config):
        super().__init__(config)
        self.weight = config.get("weight", 4.0)
        self.epsilon = config.get("epsilon", self.dt)
        self.alpha = config.get("alpha", 2.0)
        self.zeta = config.get("zeta", 1e-3)
        self.max_t = 1.0 - self.epsilon

    def sample_t(self, batch_size, device):
        return torch.rand(batch_size, 1, device=device) * self.max_t

    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        xt, target, vt = cfm_batch(model, x0, x1, t)
        x_next = xt + vt.detach() * self.epsilon
        t_next = t + self.epsilon
        vt_next = model(x_next, t_next).detach()
        speed = torch.linalg.vector_norm(vt.detach(), dim=1, keepdim=True) + self.zeta
        temporal_weight = (1.0 - t).pow(self.alpha) / self.epsilon
        iso_per_sample = temporal_weight * torch.sum(torch.abs((vt - vt_next) / speed), dim=1, keepdim=True)
        iso = torch.mean(iso_per_sample)
        return {"cfm": F.mse_loss(vt, target), "iso_fd": self.weight * iso}


@register(METHODS, "lc_jvp_material_derivative")
class LagrangianConsistencyJVP(Method):
    def __init__(self, config):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)

    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        xt = ((1 - t) * x0 + t * x1).detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        target = x1 - x0
        vt = model(xt, t)
        cfm = F.mse_loss(vt, target)

        material = material_derivative_jvp(model, xt, t, vt)
        return {"cfm": cfm, "lc_jvp": self.weight * torch.mean(material.pow(2))}


@register(METHODS, "weighted_material_residual")
class WeightedMaterialResidual(Method):
    """Material-derivative residual penalty with time-only gate.

    weighting options
    -----------------
    "uniform"    λ ‖R‖²          (ablation: no gating)
    "time_gate"  λ t^β ‖R‖²     (suppress penalty at early t where ambiguity is high)
    """

    def __init__(self, config):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)
        self.weighting = config.get("weighting", "time_gate")
        self.beta = config.get("beta", 1.0)
        self.k_neighbours = config.get("k_neighbours", 8)
        self.kappa = config.get("kappa", 1.0)

    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        xt = ((1 - t) * x0 + t * x1).detach().requires_grad_(True)
        t_g = t.detach().requires_grad_(True)
        target = x1 - x0
        vt = model(xt, t_g)
        cfm = F.mse_loss(vt, target)

        material = material_derivative_jvp(model, xt, t_g, vt)
        residual_sq = material.pow(2).sum(dim=1, keepdim=True)

        if self.weighting == "uniform":
            w = torch.ones(t.shape[0], 1, device=t.device)
        elif self.weighting == "time_gate":
            w = t.pow(self.beta)
        elif self.weighting == "variance_gate":
            w = self._variance_gate(xt.detach(), x1 - x0)
        else:
            raise ValueError(f"Unknown weighting: {self.weighting!r}")

        reg = torch.mean(w * residual_sq)
        return {"cfm": cfm, "wmr": self.weight * reg}

    def _variance_gate(self, xt, u):
        """g(x_t) = 1 / (1 + kappa * Var[u | x_t]) estimated via k-NN in minibatch."""
        # pairwise squared distances in x_t space
        diff = xt.unsqueeze(1) - xt.unsqueeze(0)          # (B, B, D)
        dist2 = diff.pow(2).sum(dim=2)                     # (B, B)
        # exclude self by setting diagonal to large value
        dist2 = dist2 + torch.eye(xt.shape[0], device=xt.device) * 1e9
        # k nearest neighbours
        _, idx = dist2.topk(self.k_neighbours, dim=1, largest=False)  # (B, k)
        u_nn = u[idx]                                      # (B, k, D)
        var = u_nn.var(dim=1).mean(dim=1, keepdim=True)   # (B, 1)
        return (1.0 / (1.0 + self.kappa * var.detach()))


def build_method(name, config):
    return METHODS[name](config)
