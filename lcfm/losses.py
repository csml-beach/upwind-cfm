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


def spatial_directional_jvp(model, x, t, velocity):
    """Compute (velocity dot grad_x) v(x,t) with a JVP in x only."""
    _, convective = torch.autograd.functional.jvp(
        lambda x_in: model(x_in, t),
        x,
        velocity.detach(),
        create_graph=False,
    )
    return convective


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


@register(METHODS, "directional_regularization_cfm")
class DirectionalRegularizationCFM(Method):
    """CFM with optional directional regularization.

    residual_loss selects how the material residual is approximated:
    "l2"      squared JVP material derivative
    "iso_l1"  Iso-style normalized L1 JVP material derivative
    "fd_iso"  Iso-style forward finite-difference residual

    directional_approx selects how the directional change weight is estimated:
    "jvp"  directional JVP of the learned velocity
    "fd"   forward finite difference of the learned velocity
    """

    def __init__(self, config):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)
        self.weighting = config.get("weighting", "uniform")
        self.directional_dt = config.get("directional_dt", 0.2)
        self.epsilon = config.get("epsilon", self.dt)
        self.normalize_residual = config.get("normalize_residual", False)
        self.residual_loss = config.get("residual_loss", "l2")
        self.directional_approx = config.get("directional_approx", "jvp")
        self.alpha = config.get("alpha", 2.0)
        self.zeta = config.get("zeta", 1e-4)

    def sample_t(self, batch_size, device):
        if self.residual_loss == "fd_iso":
            max_t = 1.0 - self.epsilon
            return torch.rand(batch_size, 1, device=device) * max_t
        return super().sample_t(batch_size, device)

    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        xt = ((1 - t) * x0 + t * x1).detach().requires_grad_(True)
        t_g = t.detach().requires_grad_(True)
        target = x1 - x0
        vt = model(xt, t_g)
        cfm = F.mse_loss(vt, target)

        if self.residual_loss == "fd_iso":
            residual = self._fd_iso_penalty(model, xt, t_g, vt)
        else:
            material = material_derivative_jvp(model, xt, t_g, vt)
            residual = self._residual_penalty(material, vt.detach(), t)

        if self.weighting == "uniform":
            w = torch.ones(t.shape[0], 1, device=t.device)
        elif self.weighting == "directional":
            w = self._directional_weight(model, xt.detach(), t.detach(), vt.detach())
        else:
            raise ValueError(f"Unknown weighting: {self.weighting!r}")

        reg = torch.mean(w * residual)
        return {"cfm": cfm, "wmr": self.weight * reg}

    def _fd_iso_penalty(self, model, xt, t, vt):
        x_next = xt + vt.detach() * self.epsilon
        t_next = t + self.epsilon
        vt_next = model(x_next, t_next).detach()
        speed = torch.linalg.vector_norm(vt.detach(), dim=1, keepdim=True) + self.zeta
        temporal_weight = (1.0 - t).pow(self.alpha) / self.epsilon
        return temporal_weight * torch.sum(torch.abs((vt - vt_next) / speed), dim=1, keepdim=True)

    def _residual_penalty(self, material, velocity, t):
        if self.residual_loss == "l2":
            residual = material.pow(2).sum(dim=1, keepdim=True)
            if self.normalize_residual:
                speed_sq = velocity.pow(2).sum(dim=1, keepdim=True)
                residual = residual / (speed_sq + self.zeta)
            return residual
        if self.residual_loss == "iso_l1":
            speed = torch.linalg.vector_norm(velocity, dim=1, keepdim=True) + self.zeta
            residual = torch.sum(torch.abs(material / speed), dim=1, keepdim=True)
            return (1.0 - t).pow(self.alpha) * residual
        raise ValueError(f"Unknown residual_loss: {self.residual_loss!r}")

    def _directional_weight(self, model, xt, t, vt):
        """Bounded weight based on velocity change along the learned direction."""
        if self.directional_approx == "jvp":
            directional_change = spatial_directional_jvp(model, xt, t, vt).detach()
        elif self.directional_approx == "fd":
            x_next = xt + vt * self.epsilon
            vt_next = model(x_next, t).detach()
            directional_change = (vt_next - vt) / self.epsilon
        else:
            raise ValueError(f"Unknown directional_approx: {self.directional_approx!r}")

        speed = torch.linalg.vector_norm(vt, dim=1, keepdim=True).clamp_min(self.zeta)
        local_scale = torch.linalg.vector_norm(directional_change, dim=1, keepdim=True) / speed
        directional_number = self.directional_dt * local_scale
        weight = directional_number.pow(2) / (1.0 + directional_number.pow(2))
        return (self.directional_dt**2) * weight


def build_method(name, config):
    return METHODS[name](config)
