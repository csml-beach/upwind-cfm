# Lagrangian Consistency Flow Matching Metrics and Future Plans

## Naming and Paper Narrative

We will rename the current method from **Upwind-CFM** to **Lagrangian Consistency Flow Matching**.

The current method should be described as a finite-difference regularizer that encourages velocity consistency along the learned transport path. We should avoid claiming that it is a full numerical upwind scheme.

When we overhaul the paper writeup, the narrative should proceed in a structured derivation:

1. Start from the Lagrangian view of a moving sample under a learned flow-matching velocity field.
2. Introduce the material derivative:
   $$
   \frac{Dv}{Dt} = \partial_t v + (v \cdot \nabla_x)v
   $$
3. Explain that this quantity measures how the learned velocity changes for a sample moving along its trajectory.
4. Argue that large material derivative corresponds to high acceleration, trajectory curvature, solver sensitivity, and poor low-NFE behavior.
5. Derive a backward characteristic / semi-Lagrangian finite-difference approximation:
   $$
   \frac{Dv_\theta}{Dt}(x,t)
   \approx
   \frac{
   v_\theta(x,t) - v_\theta(x - \Delta t\, v_\theta(x,t), t-\Delta t)
   }{\Delta t}
   $$
6. Present the current loss as penalizing the numerator of this finite-difference material derivative:
   $$
   \mathcal{L}_{LC}
   =
   \left\|
   v_\theta(x,t) -
   v_\theta(x - \Delta t\, v_\theta(x,t), t-\Delta t)
   \right\|^2
   $$
7. Position the method as a pathwise velocity-consistency regularizer for flow matching, then compare it directly against curvature, acceleration, rectification, consistency, and smoothing baselines.

## Current 2D Metrics Strategy
To quantitatively validate the advantages of the Lagrangian consistency regularizer and the velocity-smoothing solver, we will implement the following metrics on our 2D spiral experiments:

### Geometry & Accuracy
1. **Path Length Ratio (Straightness / Smoothness):**
   - **Concept:** Measures the ratio of the actual integrated trajectory length to the straight-line distance between the initial noise sample $x_0$ and the final generated point $x_1$. 
   - **Why it matters:** Standard CFM trajectories trained with point-wise MSE can zig-zag. Lagrangian consistency penalizes abrupt velocity changes along the trajectory, resulting in smoother, straighter trajectories. An ideal path length ratio is close to 1.0.

2. **Wasserstein Distance (Distributional Accuracy):**
   - **Concept:** Calculates the Earth Mover's Distance between the final generated point cloud and the true target data distribution.
   - **Why it matters:** Ensures that the enforced smoothness does not degrade the model's ability to accurately capture the complex geometry of the target distribution.

### Robustness & Efficiency
3. **NFE to Threshold (Number of Function Evaluations):**
   - **Concept:** Evaluate the Wasserstein distance across a range of step sizes (e.g., 5, 10, 15, 20 steps).
   - **Why it matters:** Proves **Fast Inference**. If Upwind-CFM achieves a low Wasserstein distance in just 10 steps, but Standard CFM requires 30 steps to achieve that same quality, it quantifies the computational savings.

4. **Divergence under Perturbation (Noise Sensitivity):**
   - **Concept:** Run generation twice: once with zero inference noise, and once with high inference noise ($\sigma = 1.0$). Measure the Wasserstein distance between the two final distributions.
   - **Why it matters:** Quantifies **Robustness**. The Lagrangian consistency regularizer should reduce sensitivity to perturbations by discouraging abrupt pathwise velocity changes.

5. **Local Lipschitz Constant (Field Stiffness):**
   - **Concept:** Estimate the gradient of the vector field $\nabla v_\theta(x)$ along the trajectories using finite differences or automatic differentiation.
   - **Why it matters:** A vector field with high gradients (high Lipschitz constant) is mathematically "stiff." Stiff ODEs require tiny step sizes to avoid numerical explosion. If the Lagrangian consistency loss lowers effective field stiffness or trajectory acceleration, this helps explain why it can support larger step sizes and faster inference.

## Future Metrics (For Sequential Data & Video Generation)
When scaling this research to temporal data (e.g., 1D moving waves, fluid simulations, or video frames), we will evaluate the temporal regularizing properties of Lagrangian Consistency Flow Matching using:

1. **Temporal Total Variation:**
   - **Concept:** Measures the variation between consecutive generated states: $||x_t - x_{t-1}||$.
   - **Why it matters:** In dynamic flow matching, intermediate integration steps may represent physical or sequence time. High variation indicates unnatural "flickering" or structural tearing. Pathwise velocity consistency should reduce this when it does not oversmooth the dynamics.

2. **Fréchet Video Distance (FVD):**
   - **Concept:** The industry standard for evaluating video generation quality.
   - **Why it matters:** Evaluates both the visual fidelity of individual frames and the temporal coherence across the entire sequence.
