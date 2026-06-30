from pathlib import Path

import numpy as np
import torch
from scipy.integrate import odeint

from .registry import DATASETS, register


class SpiralProblem:
    name = "spiral"
    dim = 2

    def __init__(self, config):
        self.n_train = config.get("n_train", 2000)
        self.n_test = config.get("n_test", 1000)
        self.noise = config.get("noise", 0.15)
        self.train = self._make_spiral(self.n_train, self.noise)
        self.test = self._make_spiral(self.n_test, self.noise)

    @staticmethod
    def _make_spiral(n_samples, noise):
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        data = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        data = (data + noise * np.random.randn(n_samples, 2)) / 5.0
        return torch.tensor(data, dtype=torch.float32)

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = torch.randn_like(x1)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return torch.randn(n_eval, self.dim, device=device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)


class FiveModesProblem:
    name = "five_modes"
    dim = 2

    def __init__(self, config):
        self.n_train = config.get("n_train", 5000)
        self.n_test = config.get("n_test", 2000)
        self.radius = config.get("radius", 4.0)
        self.sigma_mode = config.get("sigma_mode", 0.20)
        self.source_mean = torch.tensor(config.get("source_mean", [0.0, 0.0]), dtype=torch.float32)
        self.source_std = config.get("source_std", 1.0)
        self.n_modes = 5
        self.mode_centers = self._make_centers()
        self.train = self._sample_modes(self.n_train)
        self.test = self._sample_modes(self.n_test)

    def _make_centers(self):
        angles = torch.arange(self.n_modes, dtype=torch.float32) * (2.0 * torch.pi / self.n_modes)
        return self.radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    def _sample_modes(self, n_samples):
        mode_idx = torch.randint(self.n_modes, (n_samples,))
        centers = self.mode_centers[mode_idx]
        return centers + self.sigma_mode * torch.randn(n_samples, self.dim)

    def _source_std_tensor(self, device):
        return torch.as_tensor(self.source_std, dtype=torch.float32, device=device).reshape(-1)

    def _sample_source(self, n_samples, device):
        mean = self.source_mean.to(device)
        std = self._source_std_tensor(device)
        if std.numel() == 1:
            return mean + std.item() * torch.randn(n_samples, self.dim, device=device)
        if std.numel() != self.dim:
            raise ValueError("source_std must be a scalar or length-2 sequence.")
        return mean + torch.randn(n_samples, self.dim, device=device) * std

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = self._sample_source(batch_size, device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return self._sample_source(n_eval, device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def centers(self, device):
        return self.mode_centers.to(device)


class FanModesProblem:
    name = "fan_modes"
    dim = 2

    def __init__(self, config):
        self.n_train = config.get("n_train", 5000)
        self.n_test = config.get("n_test", 2000)
        self.sigma_mode = config.get("sigma_mode", 0.20)
        self.source_mean = torch.tensor(config.get("source_mean", [-8.0, 0.0]), dtype=torch.float32)
        self.source_std = config.get("source_std", 1.0)
        self.mode_centers = self._make_centers(config)
        self.n_modes = self.mode_centers.shape[0]
        self.train = self._sample_modes(self.n_train)
        self.test = self._sample_modes(self.n_test)

    def _make_centers(self, config):
        if "target_centers" in config:
            centers = torch.tensor(config["target_centers"], dtype=torch.float32)
            if centers.ndim != 2 or centers.shape[1] != self.dim:
                raise ValueError("target_centers must have shape [n_modes, 2].")
            return centers

        target_x = config.get("target_x", 4.5)
        target_ys = torch.tensor(config.get("target_ys", [-4.0, -2.0, 0.0, 2.0, 4.0]), dtype=torch.float32)
        fan_curve = config.get("fan_curve", 1.0)
        max_abs_y = torch.max(torch.abs(target_ys)).clamp_min(1.0)
        x = target_x + fan_curve * (1.0 - torch.abs(target_ys) / max_abs_y)
        return torch.stack([x, target_ys], dim=1)

    def _source_std_tensor(self, device):
        return torch.as_tensor(self.source_std, dtype=torch.float32, device=device).reshape(-1)

    def _sample_source(self, n_samples, device):
        mean = self.source_mean.to(device)
        std = self._source_std_tensor(device)
        if std.numel() == 1:
            return mean + std.item() * torch.randn(n_samples, self.dim, device=device)
        if std.numel() != self.dim:
            raise ValueError("source_std must be a scalar or length-2 sequence.")
        return mean + torch.randn(n_samples, self.dim, device=device) * std

    def _sample_modes(self, n_samples):
        mode_idx = torch.randint(self.n_modes, (n_samples,))
        centers = self.mode_centers[mode_idx]
        return centers + self.sigma_mode * torch.randn(n_samples, self.dim)

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = self._sample_source(batch_size, device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return self._sample_source(n_eval, device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def centers(self, device):
        return self.mode_centers.to(device)


class StagedModesProblem:
    name = "staged_modes"
    dim = 2

    def __init__(self, config):
        self.n_train = config.get("n_train", 5000)
        self.n_test = config.get("n_test", 2000)
        self.sigma_mode = config.get("sigma_mode", 0.20)
        self.source_mean = torch.tensor(config.get("source_mean", [0.0, 0.0]), dtype=torch.float32)
        self.source_std = config.get("source_std", 0.15)
        self.mode_centers = self._make_centers(config)
        self.n_modes = self.mode_centers.shape[0]
        self.train = self._sample_modes(self.n_train)
        self.test = self._sample_modes(self.n_test)

    def _make_centers(self, config):
        default_centers = [
            [2.4, -0.4],
            [3.4, 2.1],
            [4.8, -2.8],
            [6.2, 0.8],
            [7.4, 3.6],
        ]
        centers = torch.tensor(config.get("target_centers", default_centers), dtype=torch.float32)
        if centers.ndim != 2 or centers.shape[1] != self.dim:
            raise ValueError("target_centers must have shape [n_modes, 2].")
        return centers

    def _source_std_tensor(self, device):
        return torch.as_tensor(self.source_std, dtype=torch.float32, device=device).reshape(-1)

    def _sample_source(self, n_samples, device):
        mean = self.source_mean.to(device)
        std = self._source_std_tensor(device)
        if std.numel() == 1:
            return mean + std.item() * torch.randn(n_samples, self.dim, device=device)
        if std.numel() != self.dim:
            raise ValueError("source_std must be a scalar or length-2 sequence.")
        return mean + torch.randn(n_samples, self.dim, device=device) * std

    def _sample_modes(self, n_samples):
        mode_idx = torch.randint(self.n_modes, (n_samples,))
        centers = self.mode_centers[mode_idx]
        return centers + self.sigma_mode * torch.randn(n_samples, self.dim)

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = self._sample_source(batch_size, device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return self._sample_source(n_eval, device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def centers(self, device):
        return self.mode_centers.to(device)


class GaussianMixtureNDProblem:
    name = "gaussian_mixture_nd"

    def __init__(self, config):
        self.dim = config.get("dim", 16)
        self.n_modes = config.get("n_modes", 8)
        self.n_train = config.get("n_train", 5000)
        self.n_test = config.get("n_test", 2000)
        self.radius = config.get("radius", 4.0)
        self.sigma_mode = config.get("sigma_mode", 0.20)
        self.source_mean = torch.tensor(config.get("source_mean", [0.0] * self.dim), dtype=torch.float32)
        self.source_std = config.get("source_std", 0.15)
        if self.source_mean.numel() != self.dim:
            raise ValueError("source_mean must have length dim.")
        self.mode_centers = self._make_centers(config)
        self.train = self._sample_modes(self.n_train)
        self.test = self._sample_modes(self.n_test)

    def _make_centers(self, config):
        if "target_centers" in config:
            centers = torch.tensor(config["target_centers"], dtype=torch.float32)
            if centers.ndim != 2 or centers.shape[1] != self.dim:
                raise ValueError("target_centers must have shape [n_modes, dim].")
            self.n_modes = centers.shape[0]
            return centers

        if config.get("center_type", "simplex") == "simplex" and self.dim >= self.n_modes:
            centers = torch.eye(self.n_modes, dtype=torch.float32)
            centers = centers - centers.mean(dim=0, keepdim=True)
            if self.dim > self.n_modes:
                padding = torch.zeros(self.n_modes, self.dim - self.n_modes, dtype=torch.float32)
                centers = torch.cat([centers, padding], dim=1)
        else:
            generator = torch.Generator().manual_seed(config.get("center_seed", 0))
            centers = torch.randn(self.n_modes, self.dim, generator=generator)
            centers = centers - centers.mean(dim=0, keepdim=True)
        return self.radius * centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def _source_std_tensor(self, device):
        return torch.as_tensor(self.source_std, dtype=torch.float32, device=device).reshape(-1)

    def _sample_source(self, n_samples, device):
        mean = self.source_mean.to(device)
        std = self._source_std_tensor(device)
        if std.numel() == 1:
            return mean + std.item() * torch.randn(n_samples, self.dim, device=device)
        if std.numel() != self.dim:
            raise ValueError("source_std must be a scalar or length dim sequence.")
        return mean + torch.randn(n_samples, self.dim, device=device) * std

    def _sample_modes(self, n_samples):
        mode_idx = torch.randint(self.n_modes, (n_samples,))
        centers = self.mode_centers[mode_idx]
        return centers + self.sigma_mode * torch.randn(n_samples, self.dim)

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = self._sample_source(batch_size, device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return self._sample_source(n_eval, device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def centers(self, device):
        return self.mode_centers.to(device)


class CIFAR10Problem:
    name = "cifar10"
    image_shape = (3, 32, 32)
    dim = 3 * 32 * 32
    num_classes = 10

    def __init__(self, config):
        self.data_root = Path(config.get("data_root", "data"))
        self.download = config.get("download", True)
        self.fake_data = config.get("fake_data", False)
        self.class_conditional = bool(config.get("class_conditional", False))
        self.n_train = config.get("n_train")
        self.n_test = config.get("n_test")
        self.data_seed = config.get("data_seed", 0)
        self.train, self.train_labels = self._load_split(train=True)
        self.test, self.test_labels = self._load_split(train=False)
        self.n_train = self.train.shape[0]
        self.n_test = self.test.shape[0]

    def _fake_split(self, n_samples, seed_offset):
        generator = torch.Generator().manual_seed(self.data_seed + seed_offset)
        images = 2.0 * torch.rand(n_samples, *self.image_shape, generator=generator) - 1.0
        labels = torch.arange(n_samples, dtype=torch.long) % self.num_classes
        return images.reshape(n_samples, self.dim).contiguous(), labels

    def _load_split(self, train):
        max_samples = self.n_train if train else self.n_test
        if self.fake_data:
            default_n = 512 if train else 128
            return self._fake_split(max_samples or default_n, 0 if train else 10_000)

        try:
            from torchvision.datasets import CIFAR10
        except ImportError as exc:
            raise ImportError("CIFAR-10 runs require torchvision. Install requirements.txt first.") from exc

        dataset = CIFAR10(root=str(self.data_root), train=train, download=self.download)
        data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float()
        labels = torch.tensor(dataset.targets, dtype=torch.long)
        data = data / 127.5 - 1.0
        if max_samples is not None:
            data = data[:max_samples]
            labels = labels[:max_samples]
        return data.reshape(data.shape[0], self.dim).contiguous(), labels

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = torch.randn(batch_size, self.dim, device=device)
        if self.class_conditional:
            return x0, x1, self.train_labels[idx].to(device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return torch.randn(n_eval, self.dim, device=device)

    def eval_labels(self, n_eval, device):
        labels = torch.arange(n_eval, dtype=torch.long, device=device)
        return labels % self.num_classes

    def _select_by_labels(self, data, data_labels, labels, device, split_name):
        labels = labels.detach().cpu().long()
        indices = []
        counters = {}
        for label in labels.tolist():
            matches = torch.nonzero(data_labels == label, as_tuple=False).flatten()
            if matches.numel() == 0:
                raise ValueError(f"CIFAR-10 {split_name} split has no examples for label {label}.")
            position = counters.get(label, 0) % matches.numel()
            counters[label] = position + 1
            indices.append(matches[position])
        index = torch.stack(indices)
        return data[index].to(device)

    def target_eval(self, n_eval, device, labels=None):
        if labels is not None:
            return self._select_by_labels(self.test, self.test_labels, labels, device, "eval")
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def metric_reference(self, n_eval, device, labels=None):
        if labels is not None:
            return self._select_by_labels(self.train, self.train_labels, labels, device, "train")
        if n_eval <= self.train.shape[0]:
            return self.train[:n_eval].to(device)
        idx = torch.randint(self.train.shape[0], (n_eval,))
        return self.train[idx].to(device)


class StagedShapesEasyProblem:
    name = "staged_shapes_easy"
    image_shape = (3, 32, 32)
    dim = 3 * 32 * 32

    DEFAULT_MODES = [
        {"shape": "circle", "color": [0.95, -0.55, -0.55], "center": [14.0, 16.0], "scale": 4.5},
        {"shape": "square", "color": [-0.55, -0.20, 0.95], "center": [22.0, 12.0], "scale": 5.5},
        {"shape": "triangle", "color": [-0.40, 0.90, -0.45], "center": [25.0, 25.0], "scale": 6.5},
        {"shape": "ring", "color": [0.95, 0.80, -0.45], "center": [7.0, 25.0], "scale": 5.8},
        {"shape": "cross", "color": [0.55, -0.45, 0.95], "center": [5.0, 7.0], "scale": 7.2},
    ]

    def __init__(self, config):
        self.n_train = int(config.get("n_train", 5000))
        self.n_test = int(config.get("n_test", 2000))
        self.data_seed = int(config.get("data_seed", 0))
        self.class_conditional = bool(config.get("class_conditional", False))
        self.background = float(config.get("background", -0.90))
        self.background_noise = float(config.get("background_noise", 0.025))
        self.center_jitter = float(config.get("center_jitter", 1.25))
        self.scale_jitter = float(config.get("scale_jitter", 0.35))
        self.color_jitter = float(config.get("color_jitter", 0.035))
        self.source_noise = float(config.get("source_noise", 0.045))
        self.source_blob_amplitude = float(config.get("source_blob_amplitude", 0.35))
        self.source_blob_sigma = float(config.get("source_blob_sigma", 6.0))
        self.mode_specs = config.get("modes", self.DEFAULT_MODES)
        self.n_modes = len(self.mode_specs)
        self.train, self.train_labels = self._make_split(self.n_train, self.data_seed)
        self.test, self.test_labels = self._make_split(self.n_test, self.data_seed + 10_000)

    def _grid(self):
        _, height, width = self.image_shape
        ys = torch.arange(height, dtype=torch.float32) + 0.5
        xs = torch.arange(width, dtype=torch.float32) + 0.5
        return torch.meshgrid(ys, xs, indexing="ij")

    def _source_template(self):
        yy, xx = self._grid()
        cy = self.image_shape[1] / 2.0
        cx = self.image_shape[2] / 2.0
        dist_sq = (yy - cy).pow(2) + (xx - cx).pow(2)
        blob = torch.exp(-0.5 * dist_sq / (self.source_blob_sigma**2))
        image = self.background + self.source_blob_amplitude * blob
        return image.expand(self.image_shape[0], *image.shape).clone()

    def _sample_source(self, n_samples, device):
        template = self._source_template().to(device)
        noise = self.source_noise * torch.randn(n_samples, *self.image_shape, device=device)
        return (template.unsqueeze(0) + noise).clamp(-1.0, 1.0).reshape(n_samples, self.dim)

    @staticmethod
    def _triangle_mask(xx, yy, cx, cy, scale):
        x1, y1 = cx, cy - scale
        x2, y2 = cx - 0.95 * scale, cy + 0.85 * scale
        x3, y3 = cx + 0.95 * scale, cy + 0.85 * scale
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        a = ((y2 - y3) * (xx - x3) + (x3 - x2) * (yy - y3)) / denom
        b = ((y3 - y1) * (xx - x3) + (x1 - x3) * (yy - y3)) / denom
        c = 1.0 - a - b
        return (a >= 0.0) & (b >= 0.0) & (c >= 0.0)

    def _shape_mask(self, shape, center, scale):
        yy, xx = self._grid()
        cx, cy = float(center[0]), float(center[1])
        dx = xx - cx
        dy = yy - cy
        radius = torch.sqrt(dx.pow(2) + dy.pow(2))
        if shape == "circle":
            return radius <= scale
        if shape == "square":
            return (dx.abs() <= scale) & (dy.abs() <= scale)
        if shape == "triangle":
            return self._triangle_mask(xx, yy, cx, cy, scale)
        if shape == "ring":
            thickness = max(1.5, 0.28 * scale)
            return (radius >= scale - thickness) & (radius <= scale + thickness)
        if shape == "cross":
            half_width = max(1.5, 0.25 * scale)
            return ((dx.abs() <= half_width) & (dy.abs() <= scale)) | ((dy.abs() <= half_width) & (dx.abs() <= scale))
        raise ValueError(f"Unknown staged shape: {shape}")

    def _render_one(self, mode_idx, generator):
        spec = self.mode_specs[int(mode_idx)]
        center = torch.tensor(spec["center"], dtype=torch.float32)
        center = center + (2.0 * torch.rand(2, generator=generator) - 1.0) * self.center_jitter
        scale = float(spec["scale"]) + float((2.0 * torch.rand((), generator=generator) - 1.0) * self.scale_jitter)
        color = torch.tensor(spec["color"], dtype=torch.float32)
        color = (color + self.color_jitter * torch.randn(3, generator=generator)).clamp(-1.0, 1.0)
        image = torch.full(self.image_shape, self.background, dtype=torch.float32)
        if self.background_noise > 0.0:
            image = image + self.background_noise * torch.randn(*self.image_shape, generator=generator)
        mask = self._shape_mask(spec["shape"], center, scale)
        image[:, mask] = color[:, None]
        return image.clamp(-1.0, 1.0)

    def _make_split(self, n_samples, seed):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(n_samples, dtype=torch.long) % self.n_modes
        perm = torch.randperm(n_samples, generator=generator)
        labels = labels[perm]
        images = torch.stack([self._render_one(label, generator) for label in labels.tolist()])
        return images.reshape(n_samples, self.dim).contiguous(), labels

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = self._sample_source(batch_size, device)
        if self.class_conditional:
            return x0, x1, self.train_labels[idx].to(device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return self._sample_source(n_eval, device)

    def eval_labels(self, n_eval, device):
        labels = torch.arange(n_eval, dtype=torch.long, device=device)
        return labels % self.n_modes

    def _select_by_labels(self, data, data_labels, labels, device):
        labels = labels.detach().cpu().long()
        indices = []
        counters = {}
        for label in labels.tolist():
            matches = torch.nonzero(data_labels == label, as_tuple=False).flatten()
            position = counters.get(label, 0) % matches.numel()
            counters[label] = position + 1
            indices.append(matches[position])
        return data[torch.stack(indices)].to(device)

    def target_eval(self, n_eval, device, labels=None):
        if labels is not None:
            return self._select_by_labels(self.test, self.test_labels, labels, device)
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def metric_reference(self, n_eval, device, labels=None):
        if labels is not None:
            return self._select_by_labels(self.train, self.train_labels, labels, device)
        if n_eval <= self.train.shape[0]:
            return self.train[:n_eval].to(device)
        idx = torch.randint(self.train.shape[0], (n_eval,))
        return self.train[idx].to(device)


class CheckerboardRefinementProblem:
    name = "checkerboard_refinement"
    dim = 2

    def __init__(self, config):
        self.n_train = int(config.get("n_train", 5000))
        self.n_test = int(config.get("n_test", 2000))
        self.n_coarse_per_axis = int(config.get("n_coarse_per_axis", 2))
        self.n_fine_per_coarse = int(config.get("n_fine_per_coarse", 4))
        self.extent = float(config.get("extent", 4.0))
        self.source_std = float(config.get("source_std", 0.18))
        self.target_margin = float(config.get("target_margin", 0.10))
        self.paired_modes = bool(config.get("paired_modes", True))
        self.coarse_centers = self._make_coarse_centers()
        self.n_modes = self.coarse_centers.shape[0]
        self.train, self.train_labels = self._sample_target(self.n_train)
        self.test, self.test_labels = self._sample_target(self.n_test)

    def _make_coarse_centers(self):
        coords = torch.linspace(
            -self.extent + self.extent / self.n_coarse_per_axis,
            self.extent - self.extent / self.n_coarse_per_axis,
            self.n_coarse_per_axis,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=1)

    def _sample_labels(self, n_samples):
        return torch.randint(self.n_modes, (n_samples,), dtype=torch.long)

    def _coarse_cell_size(self):
        return 2.0 * self.extent / self.n_coarse_per_axis

    def _sample_source_for_labels(self, labels, device):
        centers = self.coarse_centers[labels.cpu()].to(device)
        return centers + self.source_std * torch.randn(labels.numel(), self.dim, device=device)

    def _sample_target_for_labels(self, labels, device):
        labels_cpu = labels.cpu()
        centers = self.coarse_centers[labels_cpu].to(device)
        coarse_size = self._coarse_cell_size()
        fine_size = coarse_size / self.n_fine_per_coarse

        candidates = []
        for iy in range(self.n_fine_per_coarse):
            for ix in range(self.n_fine_per_coarse):
                if (ix + iy) % 2 == 0:
                    candidates.append((ix, iy))
        choice = torch.randint(len(candidates), (labels.numel(),), device=device)
        offsets = torch.tensor(candidates, dtype=torch.float32, device=device)[choice]
        lower = -0.5 * coarse_size + offsets * fine_size + self.target_margin * fine_size
        upper = -0.5 * coarse_size + (offsets + 1.0) * fine_size - self.target_margin * fine_size
        local = lower + torch.rand(labels.numel(), self.dim, device=device) * (upper - lower)
        return centers + local

    def _sample_target(self, n_samples):
        labels = self._sample_labels(n_samples)
        return self._sample_target_for_labels(labels, torch.device("cpu")), labels

    def sample_train_batch(self, batch_size, device):
        labels = self._sample_labels(batch_size)
        if self.paired_modes:
            return self._sample_source_for_labels(labels, device), self._sample_target_for_labels(labels, device)
        target_labels = self._sample_labels(batch_size)
        return self._sample_source_for_labels(labels, device), self._sample_target_for_labels(target_labels, device)

    def eval_initial(self, n_eval, device):
        labels = torch.arange(n_eval, dtype=torch.long) % self.n_modes
        return self._sample_source_for_labels(labels, device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        labels = self._sample_labels(n_eval)
        return self._sample_target_for_labels(labels, device)


class CheckerboardRefinementImageProblem:
    name = "checkerboard_refinement_image"
    image_shape = (3, 32, 32)
    dim = 3 * 32 * 32

    DEFAULT_PHASES = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    def __init__(self, config):
        self.n_train = int(config.get("n_train", 5000))
        self.n_test = int(config.get("n_test", 2000))
        self.data_seed = int(config.get("data_seed", 0))
        self.class_conditional = bool(config.get("class_conditional", False))
        self.coarse_cells = int(config.get("coarse_cells", 4))
        self.fine_cells = int(config.get("fine_cells", 8))
        self.coarse_contrast = float(config.get("coarse_contrast", 0.48))
        self.fine_contrast = float(config.get("fine_contrast", 0.42))
        self.source_noise = float(config.get("source_noise", 0.025))
        self.target_noise = float(config.get("target_noise", 0.015))
        self.paired_modes = bool(config.get("paired_modes", True))
        self.phase_specs = config.get("phases", self.DEFAULT_PHASES)
        self.n_modes = len(self.phase_specs)
        self.train, self.train_labels = self._make_split(self.n_train, self.data_seed, target=True)
        self.test, self.test_labels = self._make_split(self.n_test, self.data_seed + 10_000, target=True)

    def _grid(self):
        _, height, width = self.image_shape
        ys = torch.arange(height, dtype=torch.float32)
        xs = torch.arange(width, dtype=torch.float32)
        return torch.meshgrid(ys, xs, indexing="ij")

    def _checker_sign(self, cells, phase):
        _, height, width = self.image_shape
        yy, xx = self._grid()
        cell_h = height / float(cells)
        cell_w = width / float(cells)
        px, py = int(phase[0]), int(phase[1])
        ix = torch.floor(xx / cell_w).long() + px
        iy = torch.floor(yy / cell_h).long() + py
        return torch.where((ix + iy) % 2 == 0, 1.0, -1.0)

    def _render_source_one(self, mode_idx, generator):
        phase = self.phase_specs[int(mode_idx)]
        coarse = self._checker_sign(self.coarse_cells, phase)
        scalar = self.coarse_contrast * coarse
        image = scalar.expand(self.image_shape[0], *scalar.shape).clone()
        if self.source_noise > 0.0:
            image = image + self.source_noise * torch.randn(*self.image_shape, generator=generator)
        return image.clamp(-1.0, 1.0)

    def _render_target_one(self, mode_idx, generator):
        phase = self.phase_specs[int(mode_idx)]
        coarse = self._checker_sign(self.coarse_cells, phase)
        fine = self._checker_sign(self.fine_cells, phase)
        scalar = self.coarse_contrast * coarse + self.fine_contrast * fine
        image = scalar.expand(self.image_shape[0], *scalar.shape).clone()
        if self.target_noise > 0.0:
            image = image + self.target_noise * torch.randn(*self.image_shape, generator=generator)
        return image.clamp(-1.0, 1.0)

    def _make_split(self, n_samples, seed, target):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(n_samples, dtype=torch.long) % self.n_modes
        perm = torch.randperm(n_samples, generator=generator)
        labels = labels[perm]
        render = self._render_target_one if target else self._render_source_one
        images = torch.stack([render(label, generator) for label in labels.tolist()])
        return images.reshape(n_samples, self.dim).contiguous(), labels

    def sample_train_batch(self, batch_size, device):
        generator = torch.Generator().manual_seed(torch.randint(2**31 - 1, (1,)).item())
        if self.paired_modes:
            labels = torch.randint(self.n_modes, (batch_size,), device="cpu")
            x0 = torch.stack([self._render_source_one(label, generator) for label in labels.tolist()])
            x1 = torch.stack([self._render_target_one(label, generator) for label in labels.tolist()])
            x0 = x0.reshape(batch_size, self.dim).to(device)
            x1 = x1.reshape(batch_size, self.dim).to(device)
            if self.class_conditional:
                return x0, x1, labels.to(device)
            return x0, x1
        x0_labels = torch.randint(self.n_modes, (batch_size,), device="cpu")
        x1_labels = torch.randint(self.n_modes, (batch_size,), device="cpu")
        x0 = torch.stack([self._render_source_one(label, generator) for label in x0_labels.tolist()])
        x1 = torch.stack([self._render_target_one(label, generator) for label in x1_labels.tolist()])
        x0 = x0.reshape(batch_size, self.dim).to(device)
        x1 = x1.reshape(batch_size, self.dim).to(device)
        if self.class_conditional:
            return x0, x1, x1_labels.to(device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        labels = torch.arange(n_eval, dtype=torch.long) % self.n_modes
        generator = torch.Generator().manual_seed(self.data_seed + 20_000)
        images = torch.stack([self._render_source_one(label, generator) for label in labels.tolist()])
        return images.reshape(n_eval, self.dim).to(device)

    def eval_labels(self, n_eval, device):
        labels = torch.arange(n_eval, dtype=torch.long, device=device)
        return labels % self.n_modes

    def _select_by_labels(self, data, data_labels, labels, device):
        labels = labels.detach().cpu().long()
        indices = []
        counters = {}
        for label in labels.tolist():
            matches = torch.nonzero(data_labels == label, as_tuple=False).flatten()
            position = counters.get(label, 0) % matches.numel()
            counters[label] = position + 1
            indices.append(matches[position])
        return data[torch.stack(indices)].to(device)

    def target_eval(self, n_eval, device, labels=None):
        if labels is not None:
            return self._select_by_labels(self.test, self.test_labels, labels, device)
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def metric_reference(self, n_eval, device, labels=None):
        if labels is not None:
            return self._select_by_labels(self.train, self.train_labels, labels, device)
        if n_eval <= self.train.shape[0]:
            return self.train[:n_eval].to(device)
        idx = torch.randint(self.train.shape[0], (n_eval,))
        return self.train[idx].to(device)


def burgers_rhs(u, t, dx, nu):
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
    return -u * u_x + nu * u_xx


class BurgersAutoregressiveProblem:
    name = "burgers_autoregressive"

    def __init__(self, config):
        self.n_train = config.get("n_train", 1000)
        self.n_test = config.get("n_test", 100)
        self.nx = config.get("nx", 64)
        self.nt = config.get("nt", 32)
        self.nu = config.get("nu", 0.02)
        self.dim = self.nx
        self.train_videos = self._make_data(self.n_train)
        self.test_videos = self._make_data(self.n_test)
        self.train_x0 = self.train_videos[:, :-1, :].reshape(-1, self.nx)
        self.train_x1 = self.train_videos[:, 1:, :].reshape(-1, self.nx)

    def _make_data(self, n_samples):
        length = 2.0 * np.pi
        dx = length / self.nx
        x = np.linspace(0, length, self.nx, endpoint=False)
        t = np.linspace(0, 1.0, self.nt)
        data = []
        for _ in range(n_samples):
            phi = np.random.uniform(0, 2 * np.pi)
            u0 = np.sin(x - phi)
            sol = odeint(burgers_rhs, u0, t, args=(dx, self.nu))
            data.append(sol)
        tensor = torch.tensor(np.array(data), dtype=torch.float32)
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        return (tensor - mean) / (std + 1e-5)

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train_x0.shape[0], (batch_size,))
        return self.train_x0[idx].to(device), self.train_x1[idx].to(device)

    def eval_initial(self, n_eval, device):
        return self.test_videos[:n_eval, 0, :].to(device)

    def target_eval(self, n_eval, device):
        return self.test_videos[:n_eval].to(device)


class BurgersSolutionMapProblem:
    name = "burgers_solution_map"

    def __init__(self, config):
        self.cache_path = config.get("cache_path")
        self.n_train = config.get("n_train", 1000)
        self.n_test = config.get("n_test", 100)
        self.nx = config.get("nx", 64)
        self.nt = config.get("nt", 64)
        self.nu = config.get("nu", 0.02)
        self.final_time = config.get("final_time", 1.5)
        self.ic_type = config.get("ic_type", "fourier")
        self.n_fourier_modes = config.get("n_fourier_modes", 3)
        self.spectral_decay = config.get("spectral_decay", 1.5)
        self.ic_scale = config.get("ic_scale", 0.3)
        self.ode_mxstep = config.get("ode_mxstep", 5000)
        self.dim = self.nx
        if self.cache_path:
            self.train_videos, self.test_videos = self._load_cache(self.cache_path)
            self.n_train = self.train_videos.shape[0]
            self.n_test = self.test_videos.shape[0]
            self.nt = self.train_videos.shape[1]
            self.nx = self.train_videos.shape[2]
            self.dim = self.nx
        else:
            self.train_videos = self._make_data(self.n_train)
            self.test_videos = self._make_data(self.n_test)
        self.train_x0 = self.train_videos[:, 0, :]
        self.train_x1 = self.train_videos[:, -1, :]
        self.test_x0 = self.test_videos[:, 0, :]
        self.test_x1 = self.test_videos[:, -1, :]

    def _load_cache(self, path):
        payload = np.load(Path(path).expanduser(), allow_pickle=False)
        train = torch.tensor(payload["train_videos"], dtype=torch.float32)
        test = torch.tensor(payload["test_videos"], dtype=torch.float32)
        if train.ndim != 3 or test.ndim != 3:
            raise ValueError("Burgers solution-map cache must contain 3D train_videos/test_videos arrays.")
        if train.shape[1:] != test.shape[1:]:
            raise ValueError("Cached train/test Burgers videos must have matching nt and nx.")
        return train, test

    def _sample_initial_condition(self, x):
        if self.ic_type == "phase_sine":
            phi = np.random.uniform(0, 2 * np.pi)
            return np.sin(x - phi)
        if self.ic_type != "fourier":
            raise ValueError("BurgersSolutionMapProblem ic_type must be 'fourier' or 'phase_sine'.")

        u0 = np.zeros_like(x)
        for k in range(1, self.n_fourier_modes + 1):
            scale = 1.0 / (k**self.spectral_decay)
            amp_sin = np.random.normal(0.0, scale)
            amp_cos = np.random.normal(0.0, scale)
            u0 += amp_sin * np.sin(k * x) + amp_cos * np.cos(k * x)
        u0 = u0 - u0.mean()
        u0 = u0 / (u0.std() + 1e-8)
        return self.ic_scale * u0

    def _make_data(self, n_samples):
        length = 2.0 * np.pi
        dx = length / self.nx
        x = np.linspace(0, length, self.nx, endpoint=False)
        t = np.linspace(0, self.final_time, self.nt)
        data = []
        for _ in range(n_samples):
            u0 = self._sample_initial_condition(x)
            sol = odeint(burgers_rhs, u0, t, args=(dx, self.nu), mxstep=self.ode_mxstep)
            data.append(sol)
        tensor = torch.tensor(np.array(data), dtype=torch.float32)
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        return (tensor - mean) / (std + 1e-5)

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train_x0.shape[0], (batch_size,))
        return self.train_x0[idx].to(device), self.train_x1[idx].to(device)

    def eval_initial(self, n_eval, device):
        if n_eval <= self.test_x0.shape[0]:
            return self.test_x0[:n_eval].to(device)
        idx = torch.randint(self.test_x0.shape[0], (n_eval,))
        return self.test_x0[idx].to(device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test_x1.shape[0]:
            return self.test_x1[:n_eval].to(device)
        idx = torch.randint(self.test_x1.shape[0], (n_eval,))
        return self.test_x1[idx].to(device)


register(DATASETS, "spiral")(SpiralProblem)
register(DATASETS, "five_modes")(FiveModesProblem)
register(DATASETS, "fan_modes")(FanModesProblem)
register(DATASETS, "staged_modes")(StagedModesProblem)
register(DATASETS, "gaussian_mixture_nd")(GaussianMixtureNDProblem)
register(DATASETS, "cifar10")(CIFAR10Problem)
register(DATASETS, "staged_shapes_easy")(StagedShapesEasyProblem)
register(DATASETS, "checkerboard_refinement")(CheckerboardRefinementProblem)
register(DATASETS, "checkerboard_refinement_image")(CheckerboardRefinementImageProblem)
register(DATASETS, "burgers_autoregressive")(BurgersAutoregressiveProblem)
register(DATASETS, "burgers_solution_map")(BurgersSolutionMapProblem)
