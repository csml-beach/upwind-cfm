import torch
import torch.nn as nn
import torch.optim as optim
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
    # Periodic central differences
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
    return -u * u_x + nu * u_xx

def get_burgers_data(n_samples=500, nx=64, nt=32, nu=0.02):
    L = 2.0 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, 1.0, nt)
    data = []
    for _ in range(n_samples):
        # Mix of sines to create complex, varying shocks
        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = np.random.uniform(0, 2 * np.pi)
        u0 = np.sin(x - phi1) + 0.5 * np.sin(2 * x - phi2)
        sol = odeint(burgers_rhs, u0, t, args=(dx, nu))
        data.append(sol)
    
    data = np.array(data)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    
    # Normalize globally
    mean = tensor_data.mean()
    std = tensor_data.std()
    tensor_data = (tensor_data - mean) / (std + 1e-5)
    return tensor_data

# --- 2. Sinusoidal Time Embedding & Network ---
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

class PhysicsNet(nn.Module):
    def __init__(self, dim, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, dim)
        )
        
    def forward(self, x, t):
        t_expand = t.expand(x.shape[0], 1)
        t_emb = self.time_mlp(t_expand)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)

# --- 3. Training Loop (Physics-as-a-Flow) ---
def train_physics_cfm(u0, uT, dim, n_epochs=2000, batch_size=128, lambda_upwind=0.0):
    model = PhysicsNet(dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    dt_upwind = 0.05 
    model.train()
    
    for epoch in range(n_epochs):
        idx = torch.randperm(u0.shape[0])[:batch_size]
        x0_batch = u0[idx]
        x1_batch = uT[idx]
        
        t = torch.rand(batch_size, 1) * (1.0 - dt_upwind) + dt_upwind
        xt = (1 - t) * x0_batch + t * x1_batch
        ut = x1_batch - x0_batch
        
        vt = model(xt, t)
        loss_std = torch.mean((vt - ut) ** 2)
        
        loss_upwind = torch.tensor(0.0)
        if lambda_upwind > 0:
            x_upstream = xt - vt.detach() * dt_upwind
            t_upstream = t - dt_upwind
            vt_upstream = model(x_upstream, t_upstream)
            upwind_diff = vt - vt_upstream
            loss_upwind = lambda_upwind * torch.mean(upwind_diff ** 2)
            
        loss = loss_std + loss_upwind
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:04d} | Std Loss: {loss_std.item():.4f} | Upwind Loss: {loss_upwind.item():.4f}")
            
    return model

# --- 4. Solvers ---
@torch.no_grad()
def generate_trajectory(model, x0, steps=15, inference_noise=0.0, upwind_alpha=0.0):
    model.eval()
    dt = 1.0 / steps
    x = x0.clone()
    trajectories = [x.clone()]
    v_prev = None
    epsilon = 1e-6
    
    for i in range(steps):
        t = torch.tensor([[i * dt]], dtype=torch.float32)
        v_raw = model(x, t)
        
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
        trajectories.append(x.clone())
        
    return torch.stack(trajectories)

# --- 5. Execution ---
if __name__ == "__main__":
    NX = 64
    NT = 32
    
    print("Generating Burgers' Equation Data...")
    set_seed(42)
    full_data = get_burgers_data(n_samples=1500, nx=NX, nt=NT, nu=0.02)
    
    # Physics-as-a-Flow: Map t=0 to t=T
    u0_train = full_data[:, 0, :]
    uT_train = full_data[:, -1, :]
    
    print("\nTraining STANDARD Model (lambda=0)...")
    set_seed(42)
    model_std = train_physics_cfm(u0_train, uT_train, dim=NX, n_epochs=2000, lambda_upwind=0.0)
    
    print("\nTraining UPWIND-REGULARIZED Model (lambda=2.0)...")
    set_seed(42)
    model_upwind = train_physics_cfm(u0_train, uT_train, dim=NX, n_epochs=2000, lambda_upwind=2.0)
    
    print("\nGenerating Physical Trajectories from Unseen Test Initial Conditions...")
    n_eval = 4
    inf_noise = 0.5
    steps = 15
    
    # Use different seeds to grab test data
    set_seed(123) 
    test_data = get_burgers_data(n_samples=n_eval, nx=NX, nt=NT, nu=0.02)
    u0_test = test_data[:, 0, :]
    true_traj = test_data # Shape: (n_eval, NT, NX)
    
    # Generate 
    set_seed(42)
    traj_std = generate_trajectory(model_std, u0_test, steps=steps, inference_noise=inf_noise, upwind_alpha=0.0)
    
    set_seed(42)
    traj_upw = generate_trajectory(model_upwind, u0_test, steps=steps, inference_noise=inf_noise, upwind_alpha=0.8)
    
    # Plotting
    fig, axes = plt.subplots(3, n_eval, figsize=(16, 9))
    
    for i in range(n_eval):
        # 1. True Physical Evolution
        im_true = true_traj[i].numpy()
        axes[0, i].imshow(im_true, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, 2*np.pi, 0, 1])
        axes[0, i].set_title(f"Test Condition {i+1}\nTrue Physics (PDE)")
        if i == 0: axes[0, i].set_ylabel("Time t")
        
        # 2. Standard CFM
        im_std = traj_std[:, i, :].numpy()
        axes[1, i].imshow(im_std, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, 2*np.pi, 0, 1])
        axes[1, i].set_title(f"Standard CFM")
        if i == 0: axes[1, i].set_ylabel("Generative \tau")
        
        # 3. Upwind CFM
        im_upw = traj_upw[:, i, :].numpy()
        axes[2, i].imshow(im_upw, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, 2*np.pi, 0, 1])
        axes[2, i].set_title(f"Upwind-CFM")
        if i == 0: axes[2, i].set_ylabel(r"Generative $\tau$")

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "physics_flow_comparison.png")
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved physics flow comparison plot to '{out_file}'")