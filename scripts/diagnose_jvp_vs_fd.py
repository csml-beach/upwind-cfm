#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm.losses import material_derivative_jvp, spatial_directional_jvp
from lcfm.models import build_model
from lcfm.registry import DATASETS, get
from lcfm.utils import set_seed


def summarize_pair(name, reference, approximation):
    reference_flat = reference.reshape(reference.shape[0], -1)
    approximation_flat = approximation.reshape(approximation.shape[0], -1)
    diff = approximation_flat - reference_flat
    ref_norm = torch.linalg.vector_norm(reference_flat, dim=1).clamp_min(1e-12)
    app_norm = torch.linalg.vector_norm(approximation_flat, dim=1).clamp_min(1e-12)
    rel = torch.linalg.vector_norm(diff, dim=1) / ref_norm
    cosine = torch.sum(reference_flat * approximation_flat, dim=1) / (ref_norm * app_norm)
    print(
        f"{name:>18s} | "
        f"rel_mean={rel.mean().item():.4f} rel_p90={rel.quantile(0.9).item():.4f} "
        f"cos_mean={cosine.mean().item():.4f} "
        f"ref_norm={ref_norm.mean().item():.4f} app_norm={app_norm.mean().item():.4f}"
    )


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    dataset_cls = get(DATASETS, config["dataset"])
    problem = dataset_cls(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    return config, problem, model


def flat_grad(model):
    chunks = []
    for parameter in model.parameters():
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(parameter.grad.detach().reshape(-1))
    return torch.cat(chunks)


def regularizer_gradient(model, x0, x1, t, kind, epsilon, alpha, zeta):
    model.zero_grad(set_to_none=True)
    xt = ((1.0 - t) * x0 + t * x1).detach().requires_grad_(True)
    t_grad = t.detach().requires_grad_(True)
    vt = model(xt, t_grad)
    speed = torch.linalg.vector_norm(vt.detach(), dim=1, keepdim=True) + zeta

    if kind == "fd_stopgrad":
        vt_next = model(xt + vt.detach() * epsilon, t_grad + epsilon).detach()
        residual = (vt - vt_next) / speed
        loss = torch.mean((1.0 - t_grad).pow(alpha) / epsilon * torch.sum(torch.abs(residual), dim=1, keepdim=True))
    elif kind == "jvp_detached_tangent":
        material = material_derivative_jvp(model, xt, t_grad, vt)
        residual = material / speed
        loss = torch.mean((1.0 - t_grad).pow(alpha) * torch.sum(torch.abs(residual), dim=1, keepdim=True))
    elif kind == "jvp_full_tangent":
        _, material = torch.autograd.functional.jvp(
            lambda x_in, t_in: model(x_in, t_in),
            (xt, t_grad),
            (vt, torch.ones_like(t_grad)),
            create_graph=True,
        )
        residual = material / speed
        loss = torch.mean((1.0 - t_grad).pow(alpha) * torch.sum(torch.abs(residual), dim=1, keepdim=True))
    else:
        raise ValueError(f"Unknown gradient kind: {kind!r}")

    loss.backward()
    return float(loss.detach()), flat_grad(model)


def summarize_gradients(model, x0, x1, t, epsilon, alpha, zeta):
    fd_loss, fd_grad = regularizer_gradient(model, x0, x1, t, "fd_stopgrad", epsilon, alpha, zeta)
    jvp_loss, jvp_grad = regularizer_gradient(model, x0, x1, t, "jvp_detached_tangent", epsilon, alpha, zeta)
    full_loss, full_grad = regularizer_gradient(model, x0, x1, t, "jvp_full_tangent", epsilon, alpha, zeta)
    fd_norm = fd_grad.norm().clamp_min(1e-12)
    jvp_norm = jvp_grad.norm().clamp_min(1e-12)
    full_norm = full_grad.norm().clamp_min(1e-12)
    fd_jvp_cos = torch.dot(fd_grad, jvp_grad) / (fd_norm * jvp_norm)
    fd_full_cos = torch.dot(fd_grad, full_grad) / (fd_norm * full_norm)
    jvp_full_cos = torch.dot(jvp_grad, full_grad) / (jvp_norm * full_norm)
    print(
        "gradient alignment | "
        f"fd_loss={fd_loss:.4f} jvp_det_loss={jvp_loss:.4f} jvp_full_loss={full_loss:.4f} "
        f"cos(fd,jvp_det)={fd_jvp_cos.item():.4f} "
        f"cos(fd,jvp_full)={fd_full_cos.item():.4f} "
        f"cos(jvp_det,jvp_full)={jvp_full_cos.item():.4f} "
        f"norm_ratio(jvp_det/fd)={(jvp_norm / fd_norm).item():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-3, 1e-2, 5e-2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gradient-check", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    for run_dir in args.run_dirs:
        config, problem, model = load_run(run_dir, device)
        set_seed(args.seed)
        x0, x1 = problem.sample_train_batch(args.batch_size, device)
        t = torch.rand(args.batch_size, 1, device=device)
        xt = ((1.0 - t) * x0 + t * x1).detach().requires_grad_(True)
        t_grad = t.detach().requires_grad_(True)
        vt = model(xt, t_grad)

        full_jvp = material_derivative_jvp(model, xt, t_grad, vt).detach()
        spatial_jvp = spatial_directional_jvp(model, xt, t_grad, vt).detach()
        temporal_jvp = full_jvp - spatial_jvp

        print(f"\n== {Path(run_dir).name} ==")
        print(f"method={config['method']} kwargs={config.get('method_kwargs', {})}")
        print(
            "norms | "
            f"full={torch.linalg.vector_norm(full_jvp, dim=1).mean().item():.4f} "
            f"spatial={torch.linalg.vector_norm(spatial_jvp, dim=1).mean().item():.4f} "
            f"temporal={torch.linalg.vector_norm(temporal_jvp, dim=1).mean().item():.4f}"
        )

        with torch.no_grad():
            vt_base = model(xt.detach(), t.detach())
            for eps in args.eps:
                eps_t = torch.as_tensor(eps, device=device, dtype=xt.dtype)
                full_fd = (model(xt.detach() + vt_base * eps_t, t.detach() + eps_t) - vt_base) / eps_t
                spatial_fd = (model(xt.detach() + vt_base * eps_t, t.detach()) - vt_base) / eps_t
                temporal_fd = (model(xt.detach(), t.detach() + eps_t) - vt_base) / eps_t
                summarize_pair(f"full fd eps={eps:g}", full_jvp, full_fd)
                summarize_pair(f"spatial fd eps={eps:g}", spatial_jvp, spatial_fd)
                summarize_pair(f"temporal fd eps={eps:g}", temporal_jvp, temporal_fd)

        if args.gradient_check:
            epsilon = float(config.get("method_kwargs", {}).get("epsilon", 0.05))
            alpha = float(config.get("method_kwargs", {}).get("alpha", 2.0))
            zeta = float(config.get("method_kwargs", {}).get("zeta", 1e-3))
            summarize_gradients(model, x0, x1, t, epsilon, alpha, zeta)


if __name__ == "__main__":
    main()
