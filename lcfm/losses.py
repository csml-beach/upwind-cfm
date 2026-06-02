import torch
import torch.nn.functional as F

from .registry import METHODS, register


def cfm_batch(model, x0, x1, t):
    xt = (1 - t) * x0 + t * x1
    target = x1 - x0
    vt = model(xt, t)
    return xt, target, vt


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

        _, material = torch.autograd.functional.jvp(
            lambda x_in, t_in: model(x_in, t_in),
            (xt, t),
            (vt.detach(), torch.ones_like(t)),
            create_graph=True,
        )
        return {"cfm": cfm, "lc_jvp": self.weight * torch.mean(material.pow(2))}


def build_method(name, config):
    return METHODS[name](config)
