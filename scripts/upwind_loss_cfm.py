import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    dt_upwind = 0.05 # How far upstream to look
    
    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(x1.shape[0])[:batch_size]
        x1_batch = x1[idx]
        
        x0_batch = torch.randn_like(x1_batch)
        
        # Ensure t is large enough to allow looking back dt_upwind safely
        # We sample t in [dt_upwind, 1.0] to avoid t < 0
        t = torch.rand(batch_size, 1) * (1.0 - dt_upwind) + dt_upwind
        
        xt = (1 - t) * x0_batch + t * x1_batch
        ut = x1_batch - x0_batch
        
        # 1. Forward Pass (Current Step)
        vt = model(xt, t)
        
        # Standard CFM Loss
        loss_std = torch.mean((vt - ut) ** 2)
        
        # 2. Upwind Regularization
        loss_upwind = torch.tensor(0.0)
        if lambda_upwind > 0:
            # Look upstream: x_{upwind} = x_t - v(x_t) * dt
            # We detach vt here because we just want the geometric location, 
            # we don't necessarily want to differentiate through the path finding step itself
            # though doing so is a valid choice (similar to REINFORCE). We'll detach for stability.
            x_upstream = xt - vt.detach() * dt_upwind
            t_upstream = t - dt_upwind
            
            # Predict velocity at upstream location
            vt_upstream = model(x_upstream, t_upstream)
            
            # Upwind Difference (Approximation of Material Derivative D_v / D_t)
            # Penalize the change in velocity along the advection path
            upwind_diff = vt - vt_upstream
            loss_upwind = lambda_upwind * torch.mean(upwind_diff ** 2)
            
        loss = loss_std + loss_upwind
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Std Loss: {loss_std.item():.4f} | Upwind Loss: {loss_upwind.item():.4f}")
            
    return model

# --- 4. Standard Solver for Evaluation ---
@torch.no_grad()
def standard_euler(model, x0, steps=15, inference_noise=0.5):
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

# --- 5. Execution ---
if __name__ == "__main__":
    print("Generating data...")
    torch.manual_seed(42)
    np.random.seed(42)
    x1_target = get_noisy_spiral(2000, noise=0.15)
    
    print("Training STANDARD Model (lambda=0)...")
    model_std = train_cfm_upwind_loss(x1_target, n_epochs=1000, lambda_upwind=0.0)
    
    print("\nTraining UPWIND-REGULARIZED Model (lambda=2.0)...")
    model_upwind = train_cfm_upwind_loss(x1_target, n_epochs=1000, lambda_upwind=2.0)
    
    print("\nGenerating samples...")
    n_eval = 500
    
    # We use the STANDARD Euler solver for both, with inference noise.
    # We want to see if the Upwind Loss trained a vector field that is naturally more stable 
    # and resistant to inference noise compared to the standard model.
    
    torch.manual_seed(42)
    x0_eval = torch.randn(n_eval, 2)
    traj_std_model = standard_euler(model_std, x0_eval, steps=15, inference_noise=1.0)
    
    torch.manual_seed(42)
    x0_eval = torch.randn(n_eval, 2) # Re-generate identical x0_eval just to be safe
    traj_upwind_model = standard_euler(model_upwind, x0_eval, steps=15, inference_noise=1.0)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Target Data panel
    axes[0].scatter(x1_target[:, 0], x1_target[:, 1], s=5, alpha=0.5, c='gray')
    axes[0].set_title("Target Data (Noisy Spiral)")
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    
    # Standard Model panel
    axes[1].scatter(x1_target[:, 0], x1_target[:, 1], s=5, alpha=0.1, c='gray') # Plot target background
    axes[1].scatter(traj_std_model[-1, :, 0], traj_std_model[-1, :, 1], s=5, c='blue', alpha=0.5)
    for i in range(20):
        axes[1].plot(traj_std_model[:, i, 0], traj_std_model[:, i, 1], 'b-', alpha=0.2, linewidth=0.5)
    axes[1].set_title("Standard Model (No Reg)\nStandard Euler")
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3)
    
    # Upwind Model panel
    axes[2].scatter(x1_target[:, 0], x1_target[:, 1], s=5, alpha=0.1, c='gray') # Plot target background
    axes[2].scatter(traj_upwind_model[-1, :, 0], traj_upwind_model[-1, :, 1], s=5, c='green', alpha=0.5)
    for i in range(20):
        axes[2].plot(traj_upwind_model[:, i, 0], traj_upwind_model[:, i, 1], 'g-', alpha=0.2, linewidth=0.5)
    axes[2].set_title(r"Upwind-Loss Model ($\lambda=2.0$)" + "\nStandard Euler")
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3)
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "loss_comparison.png")
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to '{out_file}'")
