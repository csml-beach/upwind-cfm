import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experiment import _condition_model
from .models import build_model
from .solvers import solve
from .utils import set_seed, write_json


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def flat_to_uint8_images(samples, image_shape):
    images = samples.reshape(samples.shape[0], *image_shape).clamp(-1.0, 1.0)
    return ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def uint8_to_classifier_float(images):
    images = images.float() / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device)[None, :, None, None]
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=images.device)[None, :, None, None]
    return (images - mean) / std


def make_eval_labels(n_samples, device):
    return torch.arange(n_samples, dtype=torch.long, device=device) % len(CIFAR10_CLASSES)


def build_cifar_resnet18(num_classes=10):
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_cifar_classifier(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_cifar_resnet18(num_classes=int(checkpoint.get("num_classes", 10))).to(device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.eval()
    info = {
        "checkpoint": str(path),
        "architecture": checkpoint.get("architecture", "cifar_resnet18") if isinstance(checkpoint, dict) else "cifar_resnet18",
        "best": checkpoint.get("best") if isinstance(checkpoint, dict) else None,
        "normalization": checkpoint.get("normalization") if isinstance(checkpoint, dict) else None,
    }
    return model, info


@torch.no_grad()
def classifier_metrics(classifier, images_uint8, labels, batch_size, device):
    confusion = torch.zeros(len(CIFAR10_CLASSES), len(CIFAR10_CLASSES), dtype=torch.long)
    total = 0
    correct = 0
    confidence_sum = 0.0

    for start in range(0, images_uint8.shape[0], batch_size):
        end = min(start + batch_size, images_uint8.shape[0])
        images = uint8_to_classifier_float(images_uint8[start:end].to(device))
        batch_labels = labels[start:end].to(device)
        logits = classifier(images)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        correct += int((pred == batch_labels).sum().item())
        total += int(batch_labels.numel())
        confidence_sum += float(probs.gather(1, batch_labels[:, None]).sum().item())
        for true_label, pred_label in zip(batch_labels.cpu(), pred.cpu()):
            confusion[int(true_label), int(pred_label)] += 1

    per_class = {}
    for idx, name in enumerate(CIFAR10_CLASSES):
        denom = int(confusion[idx].sum().item())
        per_class[name] = float(confusion[idx, idx].item() / denom) if denom else None

    return {
        "classifier_accuracy": float(correct / total) if total else 0.0,
        "classifier_condition_confidence": float(confidence_sum / total) if total else 0.0,
        "classifier_per_class_accuracy": per_class,
        "classifier_confusion_matrix": confusion.tolist(),
    }


def _reference_indices(labels, data_labels):
    labels = labels.detach().cpu().long()
    indices = []
    counters = {}
    for label in labels.tolist():
        matches = torch.nonzero(data_labels == label, as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(f"CIFAR-10 reference split has no examples for label {label}.")
        position = counters.get(label, 0) % matches.numel()
        counters[label] = position + 1
        indices.append(matches[position])
    return torch.stack(indices)


def cifar_reference_uint8(problem, n_samples, split, labels=None):
    if split == "train":
        data, data_labels = problem.train, problem.train_labels
    elif split == "test":
        data, data_labels = problem.test, problem.test_labels
    else:
        raise ValueError("reference_split must be 'train' or 'test'.")

    if labels is not None:
        index = _reference_indices(labels, data_labels)
        data = data[index]
    else:
        data = data[:n_samples] if n_samples <= data.shape[0] else data[torch.randint(data.shape[0], (n_samples,))]
    return flat_to_uint8_images(data, problem.image_shape)


def load_or_make_reference_cache(problem, n_samples, split, labels, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_tag = "balanced" if labels is not None else "plain"
    path = cache_dir / f"cifar10_{split}_{label_tag}_{n_samples}_uint8.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    payload = {
        "images": cifar_reference_uint8(problem, n_samples, split, labels=labels).cpu(),
        "labels": labels.detach().cpu() if labels is not None else None,
        "split": split,
        "n_samples": n_samples,
    }
    torch.save(payload, path)
    return payload


@torch.no_grad()
def fid_kid_metrics(fake_uint8, real_uint8, batch_size, device, kid_subset_size=100):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    metric_device = torch.device(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
    kid = KernelInceptionDistance(
        subset_size=min(int(kid_subset_size), int(fake_uint8.shape[0]), int(real_uint8.shape[0])),
        normalize=False,
    ).to(metric_device)

    for is_real, images in [(True, real_uint8), (False, fake_uint8)]:
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(metric_device)
            fid.update(batch, real=is_real)
            kid.update(batch, real=is_real)

    kid_mean, kid_std = kid.compute()
    return {
        "fid": float(fid.compute().detach().cpu()),
        "kid_mean": float(kid_mean.detach().cpu()),
        "kid_std": float(kid_std.detach().cpu()),
    }


@torch.no_grad()
def generate_cifar_samples(model, problem, config, n_samples, nfe, batch_size, seed, device):
    set_seed(seed)
    chunks = []
    label_chunks = []
    solver_name = config.get("solver", "euler")

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        size = end - start
        labels = (
            (torch.arange(start, end, dtype=torch.long, device=device) % len(CIFAR10_CLASSES))
            if getattr(problem, "class_conditional", False)
            else None
        )
        x0 = problem.eval_initial(size, device)
        eval_model = _condition_model(model, labels)
        traj = solve(solver_name, eval_model, x0, {"steps": int(nfe)})
        chunks.append(traj[-1].detach().cpu())
        if labels is not None:
            label_chunks.append(labels.detach().cpu())

    samples = torch.cat(chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0) if label_chunks else None
    return samples, labels


def load_cifar_generator(run_dir, problem, config, checkpoint_name, device, use_ema=False):
    model = build_model(config.get("model", "unet2d"), problem.dim, config).to(device)
    checkpoint_path = Path(run_dir) / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema:
        if not isinstance(checkpoint, dict) or "ema_model" not in checkpoint:
            raise ValueError(f"{checkpoint_path} does not contain EMA weights.")
        state = checkpoint["ema_model"]
    else:
        state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def write_metric_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (dict, list))})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_eval_outputs(out_dir, metrics, rows):
    out_dir = Path(out_dir)
    write_json(out_dir / "cifar10_eval_metrics.json", metrics)
    write_metric_csv(out_dir / "cifar10_eval_metrics.csv", rows)
