import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Dataset: Noisy Spiral ---
def get_noisy_spiral(n_samples=2000, noise=0.1):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + noise * np.random.randn(n_samples, 2)
    
    # Scale to roughly [-3, 3]
    x_a = x_a / 5.0 
    return torch.tensor(x_a, dtype=torch.float32)

# --- 2. Simple Velocity Network ---
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), # x, y, t
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x, t):
        # t needs to be same shape as x batch
        t_expand = t.expand(x.shape[0], 1)
        xt = torch.cat([x, t_expand], dim=-1)
        return self.net(xt)

# --- 3. Training OT-CFM ---
def train_cfm(x1, n_epochs=1500, batch_size=256):
    model = VelocityNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(x1.shape[0])[:batch_size]
        x1_batch = x1[idx]
        
        # Prior p0 is standard Gaussian
        x0_batch = torch.randn_like(x1_batch)
        
        # Sample time t
        t = torch.rand(batch_size, 1)
        
        # Optimal Transport conditional path (straight lines)
        xt = (1 - t) * x0_batch + t * x1_batch
        
        # Target velocity for OT-CFM is x1 - x0
        ut = x1_batch - x0_batch
        
        # Predict velocity
        vt = model(xt, t)
        
        loss = torch.mean((vt - ut) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f}")
            
    return model

# --- 4. Solvers ---
@torch.no_grad()
def standard_euler(model, x0, steps=100, inference_noise=0.0):
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
def upwind_euler(model, x0, steps=100, alpha=0.5, inference_noise=0.0):
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
            # Vector projection of v_raw onto v_prev
            # Proj_a(b) = (a . b) / ||a||^2 * a
            dot_product = torch.sum(v_raw * v_prev, dim=-1, keepdim=True)
            norm_sq = torch.sum(v_prev * v_prev, dim=-1, keepdim=True) + epsilon
            v_proj = (dot_product / norm_sq) * v_prev
            
            v_tilde = (1 - alpha) * v_raw + alpha * v_proj
            
        x = x + v_tilde * dt
        v_prev = v_tilde.clone()
        
        trajectories.append(x.clone())
        
    return torch.stack(trajectories)

# --- 5. Main Execution & Plotting ---
if __name__ == "__main__":
    print("Generating data...")
    x1_target = get_noisy_spiral(2000, noise=0.15)
    
    print("Training CFM Model...")
    # Train for fewer epochs to leave the vector field a bit "raw/noisy"
    model = train_cfm(x1_target, n_epochs=1000)
    
    print("Generating samples...")
    n_eval = 500
    torch.manual_seed(42) # Set seed for fair comparison
    x0_eval = torch.randn(n_eval, 2)
    
    # Use fewer steps to exacerbate Euler integration errors on curves
    # Add inference noise to simulate an imperfectly learned high-D vector field
    eval_steps = 15
    inf_noise = 0.5
    
    traj_std = standard_euler(model, x0_eval, steps=eval_steps, inference_noise=inf_noise)
    traj_upwind = upwind_euler(model, x0_eval, steps=eval_steps, alpha=0.5, inference_noise=inf_noise)
    traj_upwind_high = upwind_euler(model, x0_eval, steps=eval_steps, alpha=0.8, inference_noise=inf_noise)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Target Data
    axes[0].scatter(x1_target[:, 0], x1_target[:, 1], s=5, alpha=0.5, c='gray')
    axes[0].set_title("Target Data (Noisy Spiral)")
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    
    # 2. Standard Euler
    axes[1].scatter(traj_std[-1, :, 0], traj_std[-1, :, 1], s=5, c='blue', alpha=0.5)
    for i in range(20): # plot 20 paths
        axes[1].plot(traj_std[:, i, 0], traj_std[:, i, 1], 'b-', alpha=0.2, linewidth=0.5)
    axes[1].set_title("Standard Euler")
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3)
    
    # 3. Upwind Euler (alpha=0.5)
    axes[2].scatter(traj_upwind[-1, :, 0], traj_upwind[-1, :, 1], s=5, c='green', alpha=0.5)
    for i in range(20):
        axes[2].plot(traj_upwind[:, i, 0], traj_upwind[:, i, 1], 'g-', alpha=0.2, linewidth=0.5)
    axes[2].set_title(r"Upwind Euler ($\alpha=0.5$)")
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3)
    
    # 4. Upwind Euler (alpha=0.8)
    axes[3].scatter(traj_upwind_high[-1, :, 0], traj_upwind_high[-1, :, 1], s=5, c='orange', alpha=0.5)
    for i in range(20):
        axes[3].plot(traj_upwind_high[:, i, 0], traj_upwind_high[:, i, 1], 'orange', alpha=0.2, linewidth=0.5)
    axes[3].set_title(r"Upwind Euler ($\alpha=0.8$)")
    axes[3].set_xlim(-3, 3); axes[3].set_ylim(-3, 3)
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "cfm_upwind_comparison.png")
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to '{out_file}'")
    
