# Upwind-CFM Metrics and Future Plans

## Current 2D Metrics Strategy
To quantitatively validate the advantages of the Upwind Advection Loss and Upwind ODE Solver, we will implement the following metrics on our 2D spiral experiments:

### Geometry & Accuracy
1. **Path Length Ratio (Straightness / Smoothness):**
   - **Concept:** Measures the ratio of the actual integrated trajectory length to the straight-line distance between the initial noise sample $x_0$ and the final generated point $x_1$. 
   - **Why it matters:** Standard CFM trajectories trained with point-wise MSE often zig-zag. Upwinding penalizes orthogonal shifts, resulting in smoother, straighter trajectories. An ideal path length ratio is close to 1.0.

2. **Wasserstein Distance (Distributional Accuracy):**
   - **Concept:** Calculates the Earth Mover's Distance between the final generated point cloud and the true target data distribution.
   - **Why it matters:** Ensures that the enforced smoothness does not degrade the model's ability to accurately capture the complex geometry of the target distribution.

### Robustness & Efficiency
3. **NFE to Threshold (Number of Function Evaluations):**
   - **Concept:** Evaluate the Wasserstein distance across a range of step sizes (e.g., 5, 10, 15, 20 steps).
   - **Why it matters:** Proves **Fast Inference**. If Upwind-CFM achieves a low Wasserstein distance in just 10 steps, but Standard CFM requires 30 steps to achieve that same quality, it quantifies the computational savings.

4. **Divergence under Perturbation (Noise Sensitivity):**
   - **Concept:** Run generation twice: once with zero inference noise, and once with high inference noise ($\sigma = 1.0$). Measure the Wasserstein distance between the two final distributions.
   - **Why it matters:** Quantifies **Robustness**. The Upwind model's inherent momentum acts as a shock absorber, keeping noisy outputs close to clean outputs (low divergence), whereas the Standard model scatters (high divergence).

5. **Local Lipschitz Constant (Field Stiffness):**
   - **Concept:** Estimate the gradient of the vector field $\nabla v_\theta(x)$ along the trajectories using finite differences or automatic differentiation.
   - **Why it matters:** A vector field with high gradients (high Lipschitz constant) is mathematically "stiff." Stiff ODEs require tiny step sizes to avoid numerical explosion. By proving the Upwind Loss lowers the Lipschitz constant of the learned field, we mathematically prove *why* it allows for larger step sizes and faster inference.

## Future Metrics (For Sequential Data & Video Generation)
When scaling this research to temporal data (e.g., 1D moving waves, fluid simulations, or video frames), we will evaluate the temporal regularizing properties of Upwind-CFM using:

1. **Temporal Total Variation:**
   - **Concept:** Measures the variation between consecutive generated states: $||x_t - x_{t-1}||$.
   - **Why it matters:** In dynamic flow matching, intermediate integration steps represent physical time. High variation indicates unnatural "flickering" or structural tearing. Upwinding should inherently minimize this.

2. **Fréchet Video Distance (FVD):**
   - **Concept:** The industry standard for evaluating video generation quality.
   - **Why it matters:** Evaluates both the visual fidelity of individual frames and the temporal coherence across the entire sequence.
