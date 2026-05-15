# Experiment 2: 1D Burgers' Equation (Spatiotemporal Video Generation)

## Objective
To demonstrate that the Upwind Advection Loss and Upwind ODE Solver scale beyond 2D point clouds and actively solve the "temporal consistency" problem inherent in video and sequential data generation.

## The Physical Model
We use the 1D viscous Burgers' equation, a fundamental PDE in fluid mechanics:
$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$
Where:
* $u(x, t)$ is the fluid velocity (the amplitude of our wave).
* $u \frac{\partial u}{\partial x}$ is the nonlinear advection term (which causes waves to steepen and form shocks).
* $\nu \frac{\partial^2 u}{\partial x^2}$ is the diffusion term.

## The Generative Task
A single "video" of this 1D wave evolving over time can be represented as a 2D surface matrix $U \in \mathbb{R}^{T \times S}$, where $T$ is the number of time frames and $S$ is the spatial resolution. 

We frame this as a continuous generation task:
*   **Prior ($p_0$):** A 2D matrix of pure Gaussian noise.
*   **Target ($p_1$):** A 2D matrix representing a valid Burgers' spatiotemporal surface.
*   **The Model:** An MLP (or 1D ConvNet) that learns the vector field $v(x_t, \tau)$ to transport the noise into the physical surface over generative time $\tau \in [0, 1]$.

## Why Upwinding Matters Here
In standard CFM, if the learned vector field is imperfect or if we use a low number of function evaluations (NFE) during inference, the generated 2D surface will contain high-frequency noise. If we extract the rows of this surface to play as a video, the wave will "flicker" and tear apart unnaturally from frame to frame. 

By applying our **Upwind Advection Loss** during training, we force the generative vector field to have momentum. This acts as a powerful structural regularizer. When subjected to inference noise or low step counts, the Upwind-CFM model should "ignore" the noise and generate a smooth, temporally consistent physical surface.