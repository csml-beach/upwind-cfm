import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math
from scipy.integrate import odeint

# --- 0. Set Seed Function ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. Dataset: 1D Burgers' Equation ---
def burgers_rhs(u, t, dx, nu):
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
    return -u * u_x + nu * u_xx

def get_burgers_data(n_samples=5, nx=64, nt=32, nu=0.02):
    L = 2.0 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, 1.0, nt)
    data = []
    for _ in range(n_samples):
        phi1 = np.random.uniform(0, 2 * np.pi)
        u0 = np.sin(x - phi1)
        sol = odeint(burgers_rhs, u0, t, args=(dx, nu))
        data.append(sol)
    
    data = np.array(data)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    
    # Normalize per-sample
    mean = tensor_data.mean(dim=(1, 2), keepdim=True)
    std = tensor_data.std(dim=(1, 2), keepdim=True)
    tensor_data = (tensor_data - mean) / (std + 1e-5)
    return tensor_data

# --- 2. 1D U-Net Architecture ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
    def forward(self, x, t):
        h = self.conv(x)
        time_emb = self.time_mlp(t)[:, :, None]
        return h + time_emb

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.down1 = Block1D(in_channels, hidden_channels, time_dim)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = Block1D(hidden_channels, hidden_channels * 2, time_dim)
        self.pool2 = nn.MaxPool1d(2)
        self.mid = Block1D(hidden_channels * 2, hidden_channels * 2, time_dim)
        self.up1 = nn.ConvTranspose1d(hidden_channels * 2, hidden_channels * 2, 2, stride=2)
        self.block_up1 = Block1D(hidden_channels * 4, hidden_channels, time_dim)
        self.up2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, 2, stride=2)
        self.block_up2 = Block1D(hidden_channels * 2, hidden_channels, time_dim)
        self.out = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x, t):
        x = x.unsqueeze(1)
        t_expand = t.expand(x.shape[0], 1)
        t_emb = self.time_mlp(t_expand)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        xm = self.mid(self.pool2(x2), t_emb)
        u1 = self.up1(xm)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.block_up1(u1, t_emb)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.block_up2(u2, t_emb)
        out = self.out(u2)
        return out.squeeze(1)

# --- 3. Solver ---
@torch.no_grad()
def generate_step(model, x0, device, steps=5, inference_noise=0.0, upwind_alpha=0.0):
    model.eval()
    dt = 1.0 / steps
    x = x0.clone()
    v_prev = None
    epsilon = 1e-6
    for i in range(steps):
        tau = torch.tensor([[i * dt]], dtype=torch.float32, device=device)
        v_raw = model(x, tau)
        if inference_noise > 0:
            v_raw += torch.randn_like(v_raw) * inference_noise
        if upwind_alpha > 0 and v_prev is not None:
            dot_product = torch.sum(v_raw * v_prev, dim=-1, keepdim=True)
            norm_sq = torch.sum(v_prev * v_prev, dim=-1, keepdim=True) + epsilon
            v_proj = (dot_product / norm_sq) * v_prev
            v_tilde = (1 - upwind_alpha) * v_raw + upwind_alpha * v_proj
        else:
            v_tilde = v_raw
        x = x + v_tilde * dt
        v_prev = v_tilde.clone()
    return x

@torch.no_grad()
def generate_video(model, u0_test, nt, device, steps_per_frame=30, inference_noise=0.0, upwind_alpha=0.0):
    video = [u0_test.cpu()]
    curr_frame = u0_test.to(device)
    for _ in range(nt - 1):
        next_frame = generate_step(model, curr_frame, device, steps=steps_per_frame, 
                                   inference_noise=inference_noise, upwind_alpha=upwind_alpha)
        video.append(next_frame.cpu())
        curr_frame = next_frame
    return torch.stack(video, dim=1)

# --- 4. Main ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    NX, NT = 64, 32
    
    # Load Models
    model_std = UNet1D().to(device)
    model_upw = UNet1D().to(device)
    
    model_std.load_state_dict(torch.load("upwind-cfm/models/model_std_extended.pt", map_location=device))
    model_upw.load_state_dict(torch.load("upwind-cfm/models/model_upwind_extended.pt", map_location=device))
    
    # Generate Test Data
    set_seed(123) # Match previous test set
    n_eval = 4
    test_data = get_burgers_data(n_samples=n_eval, nx=NX, nt=NT, nu=0.02)
    u0_test = test_data[:, 0, :]
    true_traj = test_data
    
    # High-Res Generation
    print("Generating High-Resolution Trajectories (30 steps per frame)...")
    inf_noise = 0.3 # Slightly lower noise for high-res clarity
    set_seed(42)
    traj_std = generate_video(model_std, u0_test, nt=NT, device=device, steps_per_frame=30, inference_noise=inf_noise, upwind_alpha=0.0)
    
    set_seed(42)
    traj_upw = generate_video(model_upw, u0_test, nt=NT, device=device, steps_per_frame=30, inference_noise=inf_noise, upwind_alpha=0.8)
    
    # Plotting
    fig, axes = plt.subplots(3, n_eval, figsize=(16, 9))
    for i in range(n_eval):
        axes[0, i].imshow(true_traj[i].numpy(), aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, 2*np.pi, 0, 1])
        axes[0, i].set_title(f"True Physics\nTest {i+1}")
        
        axes[1, i].imshow(traj_std[i].numpy(), aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, 2*np.pi, 0, 1])
        axes[1, i].set_title("Standard CFM (High-Res)")
        
        axes[2, i].imshow(traj_upw[i].numpy(), aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, 2*np.pi, 0, 1])
        axes[2, i].set_title("Upwind-CFM (High-Res)")
        
        for j in range(3):
            axes[j, i].axis('off')
            if i == 0:
                axes[j, i].axis('on')
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([0, 1])

    plt.tight_layout()
    out_file = "upwind-cfm/images/burgers_high_res_comparison.png"
    plt.savefig(out_file, dpi=200)
    print(f"Saved high-resolution plot to '{out_file}'")
