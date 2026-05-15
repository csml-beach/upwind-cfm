import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
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

def get_burgers_data(n_samples=500, nx=32, nt=32, nu=0.05):
    L = 2.0 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, 2.0, nt)
    data = []
    for _ in range(n_samples):
        phi = np.random.uniform(0, 2 * np.pi)
        u0 = np.sin(x - phi)
        sol = odeint(burgers_rhs, u0, t, args=(dx, nu))
        data.append(sol) # sol is (nt, nx)
    data = np.array(data)
    
    # Flatten to 1D vector per sample: (n_samples, nt * nx)
    tensor_data = torch.tensor(data.reshape(n_samples, -1), dtype=torch.float32)
    
    # Normalize data to prevent loss explosion
    mean = tensor_data.mean()
    std = tensor_data.std()
    return (tensor_data - mean) / (std + 1e-5)

# --- 2. Surface Network ---
class SurfaceNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, dim)
        )
        
    def forward(self, x, t):
        t_expand = t.expand(x.shape[0], 1)
        xt = torch.cat([x, t_expand], dim=-1)
        return self.net(xt)

# --- 3. Training Loop ---
def train_cfm_surface(x1, dim, n_epochs=1000, batch_size=64, lambda_upwind=0.0):
    model = SurfaceNet(dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    dt_upwind = 0.05 
    model.train()
    
    for epoch in range(n_epochs):
        idx = torch.randperm(x1.shape[0])[:batch_size]
        x1_batch = x1[idx]
        x0_batch = torch.randn_like(x1_batch)
        
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
def standard_euler_surface(model, x0, steps=15, inference_noise=0.0):
    model.eval()
    dt = 1.0 / steps
    x = x0.clone()
    for i in range(steps):
        t = torch.tensor([[i * dt]], dtype=torch.float32)
        v = model(x, t)
        if inference_noise > 0:
            v += torch.randn_like(v) * inference_noise
        x = x + v * dt
    return x

# --- 5. Execution ---
if __name__ == "__main__":
    NX = 32
    NT = 32
    DIM = NX * NT
    
    print("Generating Burgers' Equation Data...")
    set_seed(42)
    x1_target = get_burgers_data(n_samples=1000, nx=NX, nt=NT, nu=0.05)
    
    print("\nTraining STANDARD Model (lambda=0)...")
    set_seed(42)
    model_std = train_cfm_surface(x1_target, dim=DIM, n_epochs=1500, batch_size=128, lambda_upwind=0.0)
    
    print("\nTraining UPWIND-REGULARIZED Model (lambda=5.0)...")
    set_seed(42) # Stronger lambda since dimensionality is high
    model_upwind = train_cfm_surface(x1_target, dim=DIM, n_epochs=1500, batch_size=128, lambda_upwind=5.0)
    
    print("\nGenerating Spatiotemporal Surfaces...")
    n_eval = 5
    inf_noise = 0.5 # High noise to simulate difficult generation
    steps = 15
    
    set_seed(42)
    x0_eval = torch.randn(n_eval, DIM)
    
    set_seed(42)
    gen_std = standard_euler_surface(model_std, x0_eval, steps=steps, inference_noise=inf_noise)
    
    set_seed(42)
    gen_upw = standard_euler_surface(model_upwind, x0_eval, steps=steps, inference_noise=inf_noise)
    
    # --- Visualization ---
    fig, axes = plt.subplots(3, n_eval, figsize=(15, 9))
    
    for i in range(n_eval):
        # Target
        im_target = x1_target[i].numpy().reshape(NT, NX)
        axes[0, i].imshow(im_target, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[0, i].set_title(f"Target Surface {i+1}" if i==0 else "")
        axes[0, i].axis('off')
        
        # Standard Model
        im_std = gen_std[i].numpy().reshape(NT, NX)
        axes[1, i].imshow(im_std, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1, i].set_title(f"Standard Model" if i==0 else "")
        axes[1, i].axis('off')
        
        # Upwind Model
        im_upw = gen_upw[i].numpy().reshape(NT, NX)
        axes[2, i].imshow(im_upw, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[2, i].set_title(f"Upwind Model" if i==0 else "")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "burgers_video_comparison.png")
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved spatiotemporal comparison plot to '{out_file}'")
