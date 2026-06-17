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

    def __init__(self, config):
        self.data_root = Path(config.get("data_root", "data"))
        self.download = config.get("download", True)
        self.fake_data = config.get("fake_data", False)
        self.n_train = config.get("n_train")
        self.n_test = config.get("n_test")
        self.data_seed = config.get("data_seed", 0)
        self.train = self._load_split(train=True)
        self.test = self._load_split(train=False)
        self.n_train = self.train.shape[0]
        self.n_test = self.test.shape[0]

    def _fake_split(self, n_samples, seed_offset):
        generator = torch.Generator().manual_seed(self.data_seed + seed_offset)
        images = 2.0 * torch.rand(n_samples, *self.image_shape, generator=generator) - 1.0
        return images.reshape(n_samples, self.dim).contiguous()

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
        data = data / 127.5 - 1.0
        if max_samples is not None:
            data = data[:max_samples]
        return data.reshape(data.shape[0], self.dim).contiguous()

    def sample_train_batch(self, batch_size, device):
        idx = torch.randint(self.train.shape[0], (batch_size,))
        x1 = self.train[idx].to(device)
        x0 = torch.randn(batch_size, self.dim, device=device)
        return x0, x1

    def eval_initial(self, n_eval, device):
        return torch.randn(n_eval, self.dim, device=device)

    def target_eval(self, n_eval, device):
        if n_eval <= self.test.shape[0]:
            return self.test[:n_eval].to(device)
        idx = torch.randint(self.test.shape[0], (n_eval,))
        return self.test[idx].to(device)

    def metric_reference(self, n_eval, device):
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


register(DATASETS, "spiral")(SpiralProblem)
register(DATASETS, "five_modes")(FiveModesProblem)
register(DATASETS, "fan_modes")(FanModesProblem)
register(DATASETS, "staged_modes")(StagedModesProblem)
register(DATASETS, "gaussian_mixture_nd")(GaussianMixtureNDProblem)
register(DATASETS, "cifar10")(CIFAR10Problem)
register(DATASETS, "burgers_autoregressive")(BurgersAutoregressiveProblem)
