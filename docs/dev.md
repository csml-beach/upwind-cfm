# Upwind Conditional Flow Matching (Upwind-CFM)

## Overview
Conditional Flow Matching (CFM) trains a neural network to approximate a time-dependent vector field that transports samples from a simple prior distribution to a complex data distribution. While Optimal Transport (OT) CFM provides straighter paths, the standard point-wise Mean Squared Error (MSE) loss is agnostic to the temporal continuity of the learned vector field. This often results in vector fields that exhibit sharp, non-physical shifts along advection paths, making them highly susceptible to integration errors during inference, leading to chaotic trajectories.

This project explores the application of **upwinding schemes**—a classic concept from numerical PDEs used to stabilize convective acceleration—to Conditional Flow Matching. By forcing the model to respect the "upstream" flow direction, we introduce mathematical "momentum" or "inertia" into the generation process.

We investigate two synergistic approaches:
1. **The Upwind Advection Loss (Training-Time)**
2. **The Upwind Euler Solver (Inference-Time)**

## 1. The Upwind Advection Loss (Training-Time)
Drawing from numerical PDE theory, we introduce a regularization term that penalizes changes in velocity along the upstream flow path, mathematically approximating the material derivative $\frac{Dv}{Dt}$.

### Formulation
During training, for a given state $x_t$, the network calculates an upstream coordinate:
$$ x_{upwind} = x_t - v_\theta(x_t, t) \cdot \Delta t $$

We then penalize the finite-difference approximation of the material derivative along this path:
$$ D_{upwind} = v_\theta(x_t, t) - v_\theta(x_{upwind}, t - \Delta t) $$

The total loss becomes:
$$ \mathcal{L}_{total} = \underbrace{|| v_\theta(x_t, t) - u_t ||^2}_{\text{Standard CFM Target}} + \lambda \underbrace{|| D_{upwind} ||^2}_{\text{Upwind Advection Penalty}} $$

This explicitly forces the learned vector field to possess "momentum," penalizing internal shocks and shears, and smoothing the flow mathematically during training.

## 2. The Upwind Euler Solver (Inference-Time)
Instead of a standard Explicit Euler solver, we designed a custom multi-step ODE solver that keeps track of the previous step's velocity (the "upwind" direction) to construct a stabilized velocity $\tilde{v}_t$.

### Formulation
We construct a stabilized velocity $\tilde{v}_t$ that penalizes sudden orthogonal shifts, projecting a portion of the current velocity onto the previous velocity vector:

$\tilde{v}_t = (1 - \alpha) \cdot v_\theta(x_t, t) + \alpha \cdot \text{Proj}_{v_{t-\Delta t}}(v_\theta(x_t, t))$

Where:
* $\text{Proj}_{a}(b) = \frac{a \cdot b}{||a||^2} a$
* $\alpha \in [0, 1]$ is the upwind coefficient (momentum).

## Experiments & Results
We evaluated both approaches using a 2D noisy spiral target distribution. To simulate a highly complex, imperfectly learned high-dimensional space, we injected high-variance **inference noise** at every solver step and used a small number of integration steps (15 steps).

### 4-Way Comparison
We tested the combinations of Standard vs. Upwind Model, and Standard vs. Upwind Solver:

1. **Standard Model + Standard Solver:** The injected noise overwhelmed the standard learned vector field. The trajectories became highly chaotic and tangled, missing the target data.
2. **Standard Model + Upwind Solver:** The Upwind solver damped the jitter significantly, but because the underlying vector field lacked physical continuity, the paths still slightly missed the true shape of the distribution.
3. **Upwind Model + Standard Solver:** The Upwind Advection Loss gave the vector field inherent momentum. Even with a naive solver, it powerfully resisted the noise and successfully targeted the spiral.
4. **Upwind Model + Upwind Solver (The Ultimate Combination):** The network is inherently regularized *and* the solver enforces physical continuity. The result is the smoothest, tightest trajectories, sweeping perfectly into the spiral and completely shrugging off massive noise injections.

*(See `images/combined_comparison.png` for visual results)*

### 1D Spatiotemporal Dynamics (Burgers' Equation)
To validate Upwind-CFM on high-dimensional sequential data, we scaled the architecture to a **1D U-Net** and simulated the **viscous Burgers' equation**. We formulated this as an **autoregressive (next-step) generation task**, where the model learns the physical transport operator mapping the wave from state $t_k$ to $t_{k+1}$.

Autoregressive video/sequence generation is notoriously difficult because standard models accumulate errors that quickly compound into structural failure (static or tearing). We rolled out our model autoregressively over 32 frames while injecting inference noise:
1. **Standard Autoregressive CFM:** Fails to maintain the physical shockwave. Compounding errors cause the spatiotemporal surface to disintegrate.
2. **Upwind-CFM (Autoregressive):** The Upwind Advection Loss embeds powerful momentum into the vector field. It effectively absorbs inference noise and actively suppresses compounding errors, maintaining the sharp structural integrity of the wave across all 32 frames.

*(See `images/burgers_autoregressive_comparison.png` for visual results)*

## Strategic Implications & Future Work
The success of upwind-stabilization has profound implications for generative modeling:

1. **Sequential Data and Video Generation:** The integration time $t$ represents actual physical time. A jagged trajectory means pixels jumping unnaturally between frames. Upwinding acts as a **temporal regularizer**, ensuring visual consistency across generated frames and preventing video flickering.
2. **Fast Inference:** By dampening jitter and stabilizing the trajectory, we can take much larger integration steps ($\Delta t$), achieving high-quality generation in significantly fewer steps (e.g., 10-15 steps instead of 100+).
3. **Stable Likelihood Estimation:** Chaotic trajectories cause compounding numerical errors that destroy exact probability (likelihood) calculations. Smooth trajectories are a strict mathematical prerequisite for stable likelihood estimation.
