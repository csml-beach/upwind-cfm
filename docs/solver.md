# Upwind ODE Solver for Conditional Flow Matching

## The Intuition
In standard Flow Matching generation, we integrate forward using a solver like Explicit Euler:
$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$

The problem is that $v_\theta(x_t, t)$ is evaluated purely locally. If the neural network has learned a slightly noisy vector field, the solver will follow that noise, leading to jittery trajectories. 

In numerical PDEs, an **upwind scheme** stabilizes this by looking at where the flow *came from* to calculate gradients, dampening unnatural oscillations. 

## Formulation: UpwindEuler

We translate upwinding into a multi-step ODE solver. We keep track of the previous step's velocity (the "upwind" direction) to inform the current step.

Let:
* $v_\theta(x_t, t)$ be the raw network prediction at the current step.
* $v_{t-\Delta t}$ be the actual velocity taken in the previous step.

We construct a stabilized velocity $\tilde{v}_t$ that penalizes sudden orthogonal shifts, forcing the current step to respect the upstream direction. 

A geometric interpretation of this upwind penalty is projecting a portion of the current velocity onto the previous velocity vector:

$\tilde{v}_t = (1 - \alpha) \cdot v_\theta(x_t, t) + \alpha \cdot \text{Proj}_{v_{t-\Delta t}}(v_\theta(x_t, t))$

Where:
* $\text{Proj}_{a}(b) = \frac{a \cdot b}{||a||^2} a$ is the vector projection of $b$ onto $a$.
* $\alpha \in [0, 1]$ is the "upwind coefficient". 
  * $\alpha = 0$ recovers standard Explicit Euler.
  * $\alpha > 0$ adds "inertia" or "momentum" derived from the upstream trajectory.

### The Algorithm step:
1. At step $t=0$, we have no upwind history, so: $\tilde{v}_0 = v_\theta(x_0, 0)$.
2. For $t > 0$:
   $v_{raw} = v_\theta(x_t, t)$
   $v_{proj} = \left( \frac{v_{raw} \cdot \tilde{v}_{t-\Delta t}}{||\tilde{v}_{t-\Delta t}||^2 + \epsilon} \right) \tilde{v}_{t-\Delta t}$
   $\tilde{v}_t = (1 - \alpha) v_{raw} + \alpha v_{proj}$
3. Update state: $x_{t+\Delta t} = x_t + \Delta t \cdot \tilde{v}_t$

This enforces a temporal stiffness, acting as a low-pass filter on the directional changes of the velocity field, guided by the causal direction of the flow.
