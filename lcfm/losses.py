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


@register(METHODS, "directional_regularization_cfm")
class DirectionalRegularizationCFM(Method):
    def __init__(self, config):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)
        self.epsilon = config.get("epsilon", self.dt)
        self.alpha = config.get("alpha", 2.0)
        self.zeta = config.get("zeta", 1e-3)
        self.directional_dt = config.get("directional_dt", 0.2)
        self.directional_epsilon = config.get("directional_epsilon", self.epsilon)
        self.residual_loss = config.get("residual_loss", "fd_iso")
        self.directional_approx = config.get("directional_approx", "fd")
        self.weighting = config.get("weighting", "directional")
        self.max_t = 1.0 - self.epsilon
        if self.epsilon <= 0.0 or self.epsilon >= 1.0:
            raise ValueError("epsilon must be in (0, 1).")
        if self.directional_epsilon <= 0.0:
            raise ValueError("directional_epsilon must be positive.")
        if self.directional_dt <= 0.0:
            raise ValueError("directional_dt must be positive.")
        if self.residual_loss != "fd_iso":
            raise ValueError("directional_regularization_cfm currently supports residual_loss='fd_iso' only.")
        if self.directional_approx != "fd":
            raise ValueError("directional_regularization_cfm currently supports directional_approx='fd' only.")
        if self.weighting != "directional":
            raise ValueError("directional_regularization_cfm currently supports weighting='directional' only.")

    def sample_t(self, batch_size, device):
        return torch.rand(batch_size, 1, device=device) * self.max_t

    def directional_weight_fd(self, model, x, t, velocity):
        velocity_detached = velocity.detach()
        speed = torch.linalg.vector_norm(velocity_detached, dim=1, keepdim=True) + self.zeta
        x_probe = x + self.directional_epsilon * velocity_detached
        v_probe = model(x_probe, t).detach()
        directional_derivative = (v_probe - velocity_detached) / self.directional_epsilon
        l_dir = torch.linalg.vector_norm(directional_derivative, dim=1, keepdim=True) / speed
        c_dir = self.directional_dt * l_dir
        return (self.directional_dt**2) * c_dir.pow(2) / (1.0 + c_dir.pow(2))

    def loss(self, model, x0, x1):
        t = self.sample_t(x0.shape[0], x0.device)
        xt, target, vt = cfm_batch(model, x0, x1, t)
        cfm = F.mse_loss(vt, target)

        x_next = xt + self.epsilon * vt.detach()
        t_next = t + self.epsilon
        vt_next = model(x_next, t_next).detach()

        speed = torch.linalg.vector_norm(vt.detach(), dim=1, keepdim=True) + self.zeta
        temporal_weight = (1.0 - t).pow(self.alpha) / self.epsilon
        residual = torch.sum(torch.abs((vt - vt_next) / speed), dim=1, keepdim=True)
        solver_weight = self.directional_weight_fd(model, xt, t, vt)
        directional_reg = torch.mean(temporal_weight * solver_weight * residual)
        return {"cfm": cfm, "directional_reg": self.weight * directional_reg}


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


def build_method(name, config):
    return METHODS[name](config)
