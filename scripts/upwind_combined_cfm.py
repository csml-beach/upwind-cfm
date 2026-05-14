import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# --- 0. Set Seed Function ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. Dataset: Noisy Spiral ---
def get_noisy_spiral(n_samples=2000, noise=0.15):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + noise * np.random.randn(n_samples, 2)
    x_a = x_a / 5.0 
    return torch.tensor(x_a, dtype=torch.float32)

# --- 2. Simple Velocity Network ---
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x, t):
        t_expand = t.expand(x.shape[0], 1)
        xt = torch.cat([x, t_expand], dim=-1)
        return self.net(xt)

# --- 3. Training Loop with Upwind Advection Loss ---
def train_cfm_upwind_loss(x1, n_epochs=1000, batch_size=256, lambda_upwind=0.0):
    model = VelocityNet()
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
def standard_euler(model, x0, steps=15, inference_noise=1.0):
    model.eval()
    dt = 1.0 / steps
    x = x0.clone()
    trajectories = [x.clone()]
    
    for i in range(steps):
        t = torch.tensor([[i * dt]], dtype=torch.float32)
        v = model(x, t)
        if inference_noise > 0:
            v += torch.randn_like(v) * inference_noise
        x = x + v * dt
        trajectories.append(x.clone())
        
    return torch.stack(trajectories)

@torch.no_grad()
def upwind_euler(model, x0, steps=15, alpha=0.8, inference_noise=1.0):
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
            
        if v_prev is None:
            v_tilde = v_raw
        else:
            dot_product = torch.sum(v_raw * v_prev, dim=-1, keepdim=True)
            norm_sq = torch.sum(v_prev * v_prev, dim=-1, keepdim=True) + epsilon
            v_proj = (dot_product / norm_sq) * v_prev
            v_tilde = (1 - alpha) * v_raw + alpha * v_proj
            
        x = x + v_tilde * dt
        v_prev = v_tilde.clone()
        trajectories.append(x.clone())
        
    return torch.stack(trajectories)

# --- 5. Execution ---
if __name__ == "__main__":
    print("Generating data...")
    set_seed(42)
    x1_target = get_noisy_spiral(2000, noise=0.15)
    
    print("\nTraining STANDARD Model (lambda=0)...")
    set_seed(42)
    model_std = train_cfm_upwind_loss(x1_target, n_epochs=2000, lambda_upwind=0.0)
    
    print("\nTraining UPWIND-REGULARIZED Model (lambda=2.0)...")
    set_seed(42)
    model_upwind = train_cfm_upwind_loss(x1_target, n_epochs=2000, lambda_upwind=2.0)
    
    print("\nGenerating samples...")
    n_eval = 500
    inf_noise = 1.0
    steps = 15
    
    x0_eval = torch.randn(n_eval, 2)
    
    # Combination 1: Std Model + Std Solver
    set_seed(42)
    traj_std_std = standard_euler(model_std, x0_eval, steps=steps, inference_noise=inf_noise)
    
    # Combination 2: Std Model + Upwind Solver
    set_seed(42)
    traj_std_upw = upwind_euler(model_std, x0_eval, steps=steps, alpha=0.8, inference_noise=inf_noise)
    
    # Combination 3: Upwind Model + Std Solver
    set_seed(42)
    traj_upw_std = standard_euler(model_upwind, x0_eval, steps=steps, inference_noise=inf_noise)
    
    # Combination 4: Upwind Model + Upwind Solver
    set_seed(42)
    traj_upw_upw = upwind_euler(model_upwind, x0_eval, steps=steps, alpha=0.8, inference_noise=inf_noise)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    combinations = [
        (traj_std_std, "Standard Model + Standard Solver", 'blue'),
        (traj_std_upw, "Standard Model + Upwind Solver ($\\alpha=0.8$)", 'orange'),
        (traj_upw_std, "Upwind Model ($\\lambda=2.0$) + Standard Solver", 'green'),
        (traj_upw_upw, "Upwind Model ($\\lambda=2.0$) + Upwind Solver ($\\alpha=0.8$)", 'purple')
    ]
    
    for i, (traj, title, color) in enumerate(combinations):
        ax = axes[i]
        ax.scatter(x1_target[:, 0], x1_target[:, 1], s=5, alpha=0.2, c='gray') # slightly darker background target
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=5, c=color, alpha=0.5)
        for j in range(20):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color=color, alpha=0.2, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "combined_comparison.png")
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved plot to '{out_file}'")
