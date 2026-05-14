import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os

# --- 0. Set Seed Function ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. Dataset & Model ---
def get_noisy_spiral(n_samples=2000, noise=0.15):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + noise * np.random.randn(n_samples, 2)
    x_a = x_a / 5.0 
    return torch.tensor(x_a, dtype=torch.float32)

class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x, t):
        t_expand = t.expand(x.shape[0], 1)
        xt = torch.cat([x, t_expand], dim=-1)
        return self.net(xt)

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

# --- 2. Solvers ---
@torch.no_grad()
def standard_euler(model, x0, steps=15, inference_noise=0.0):
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
def upwind_euler(model, x0, steps=15, alpha=0.8, inference_noise=0.0):
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

# --- 3. Metrics Definitions ---
def compute_wasserstein(pts1, pts2):
    # Exact 2D Earth Mover's Distance using Scipy's linear sum assignment
    d_matrix = cdist(pts1.numpy(), pts2.numpy(), metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(d_matrix)
    return d_matrix[row_ind, col_ind].mean()

def compute_path_length_ratio(traj):
    # traj shape: (steps+1, n_samples, 2)
    # Sum of segment lengths
    diffs = traj[1:] - traj[:-1]
    path_lengths = torch.norm(diffs, dim=-1).sum(dim=0)
    # Straight line distance from start to end
    straight_lengths = torch.norm(traj[-1] - traj[0], dim=-1)
    
    ratios = path_lengths / (straight_lengths + 1e-6)
    return ratios.mean().item()

def compute_local_lipschitz(model, traj, eps=1e-4):
    # Estimate max ||v(x+eps) - v(x)|| / ||eps|| along trajectory
    model.eval()
    steps = traj.shape[0] - 1
    dt = 1.0 / steps
    max_lip = 0.0
    
    for i in range(steps):
        t = torch.tensor([[i * dt]], dtype=torch.float32)
        x = traj[i] # (n_samples, 2)
        
        # Perturb randomly
        perturbation = torch.randn_like(x)
        perturbation = (perturbation / torch.norm(perturbation, dim=-1, keepdim=True)) * eps
        x_perturbed = x + perturbation
        
        with torch.no_grad():
            v_orig = model(x, t)
            v_pert = model(x_perturbed, t)
            
        diff_v = torch.norm(v_pert - v_orig, dim=-1)
        lip = diff_v / eps
        max_lip = max(max_lip, lip.max().item())
        
    return max_lip

# --- 4. Main Execution ---
if __name__ == "__main__":
    print("Generating data...")
    set_seed(42)
    x1_target = get_noisy_spiral(1000, noise=0.15) # 1000 for faster W-distance computation
    
    print("\nTraining STANDARD Model (lambda=0)...")
    set_seed(42)
    model_std = train_cfm_upwind_loss(x1_target, n_epochs=2000, lambda_upwind=0.0)
    
    print("\nTraining UPWIND-REGULARIZED Model (lambda=2.0)...")
    set_seed(42)
    model_upwind = train_cfm_upwind_loss(x1_target, n_epochs=2000, lambda_upwind=2.0)
    
    print("\n==============================================")
    print(" METRICS REPORT: 15 STEPS, HIGH NOISE (sigma=1.0)")
    print("==============================================")
    
    n_eval = 1000
    inf_noise = 1.0
    steps = 15
    
    set_seed(42)
    x0_eval = torch.randn(n_eval, 2)
    
    # Generate Trajectories
    set_seed(42); traj_std_std = standard_euler(model_std, x0_eval, steps=steps, inference_noise=inf_noise)
    set_seed(42); traj_std_upw = upwind_euler(model_std, x0_eval, steps=steps, alpha=0.8, inference_noise=inf_noise)
    set_seed(42); traj_upw_std = standard_euler(model_upwind, x0_eval, steps=steps, inference_noise=inf_noise)
    set_seed(42); traj_upw_upw = upwind_euler(model_upwind, x0_eval, steps=steps, alpha=0.8, inference_noise=inf_noise)
    
    combinations = [
        ("Std Model + Std Solver", model_std, traj_std_std),
        ("Std Model + Upw Solver", model_std, traj_std_upw),
        ("Upw Model + Std Solver", model_upwind, traj_upw_std),
        ("Upw Model + Upw Solver", model_upwind, traj_upw_upw)
    ]
    
    print(f"{'Configuration':<25} | {'Path Length Ratio':<20} | {'Wasserstein Dist':<20} | {'Max Local Lipschitz':<20}")
    print("-" * 95)
    for name, model, traj in combinations:
        plr = compute_path_length_ratio(traj)
        w_dist = compute_wasserstein(traj[-1], x1_target)
        lip = compute_local_lipschitz(model, traj)
        print(f"{name:<25} | {plr:<20.4f} | {w_dist:<20.4f} | {lip:<20.4f}")
        
    print("\n==============================================")
    print(" METRIC: DIVERGENCE UNDER PERTURBATION")
    print("==============================================")
    # Compare clean run vs noisy run (Standard Euler)
    set_seed(42); clean_std = standard_euler(model_std, x0_eval, steps=15, inference_noise=0.0)[-1]
    set_seed(42); clean_upw = standard_euler(model_upwind, x0_eval, steps=15, inference_noise=0.0)[-1]
    
    div_std = compute_wasserstein(clean_std, traj_std_std[-1])
    div_upw = compute_wasserstein(clean_upw, traj_upw_std[-1])
    
    print(f"Standard Model Noise Divergence (W-Dist): {div_std:.4f}")
    print(f"Upwind Model Noise Divergence (W-Dist):   {div_upw:.4f}")
    
    print("\n==============================================")
    print(" METRIC: NFE TO THRESHOLD (WASSERSTEIN DISTANCE)")
    print("==============================================")
    # Test across step sizes with no inference noise to purely evaluate ODE integration error
    step_sizes = [5, 10, 15, 30]
    print(f"{'Step Size':<10} | {'Standard Model W-Dist':<25} | {'Upwind Model W-Dist':<25}")
    print("-" * 65)
    
    for s in step_sizes:
        t_std = standard_euler(model_std, x0_eval, steps=s, inference_noise=0.0)[-1]
        t_upw = standard_euler(model_upwind, x0_eval, steps=s, inference_noise=0.0)[-1]
        
        w_std = compute_wasserstein(t_std, x1_target)
        w_upw = compute_wasserstein(t_upw, x1_target)
        print(f"{s:<10} | {w_std:<25.4f} | {w_upw:<25.4f}")
