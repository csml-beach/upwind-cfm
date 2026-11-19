# Nearby Literature And Current Differentiation

## Narrow Positioning

Adaptive timesteps, state adjoints, tangent-linear models, randomized projections, and
goal-oriented residual weighting are established. The paper's defensible claim is narrower:

> SCTW and Adjoint-SCTW are offline, population-calibrated, fixed-NFE meshes for frozen native
> flow-matching velocity fields. Adjoint-SCTW uses terminal state sensitivity to weight the same
> local Euler defects used by SCTW.

The profile is calibrated once and reused for rectangular GPU batches. The method does not retrain
the checkpoint, optimize a validation objective, learn a solver, form a Jacobian, or run a
per-sample error controller during generation.

## Closest Method Families

### Tangent-Linear And Adjoint Models

Errico's review presents adjoint models as tools for estimating the sensitivity of a model output
to its input. That is exactly the role played here by $M(1,t)^Tz$: it asks which local state-space
defects affect the terminal generated sample. Our adjoint is not a parameter-gradient training
method.

- Ronald M. Errico, *What Is an Adjoint Model?*, BAMS 1997.
- https://doi.org/10.1175/1520-0477(1997)078%3C2577:WIAAM%3E2.0.CO;2

### Dual-Weighted Residual Error Estimation

Becker and Rannacher weight local residuals by a dual solution associated with a quantity of
interest. Adjoint-SCTW has the same structural idea: terminal sensitivity weights a local Euler
defect. Our setting is a discrete neural ODE, the quantity of interest is the terminal state norm,
and randomized probes avoid one adjoint per output dimension.

- Roland Becker and Rolf Rannacher, *An Optimal Control Approach to A Posteriori Error Estimation
  in Finite Element Methods*, Acta Numerica 2001.
- https://doi.org/10.1017/S0962492901000010

### Neural-ODE Adjoints

Neural ODE work popularized adjoint sensitivity in differentiable learned dynamics. The important
distinction is purpose: neural-ODE adjoints are commonly used for parameter gradients during
training, whereas our checkpoint is frozen and the reverse pass estimates state sensitivity for
an offline sampler mesh.

- Ricky T. Q. Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018.
- https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html

### Randomized Matrix Estimation

The identity $\mathbb{E}(a^TM^Tz)^2=\|Ma\|^2$ for isotropic probes is a randomized norm estimate.
It is closely related to Hutchinson-style estimators. We use Gaussian probes and estimate a
transported vector norm, not a trace; the citation establishes the randomized projection lineage,
not an identical estimator.

- Michael F. Hutchinson, *A Stochastic Estimator of the Trace of the Influence Matrix for
  Laplacian Smoothing Splines*, 1989.
- https://doi.org/10.1080/03610918908812806

### Adaptive And Learned Generative-ODE Schedules

Trajectory-regularity and adaptive diffusion schedules are the main novelty risk. They already
show that trajectory geometry can guide nonuniform integration. Bespoke solvers can optimize
coefficients and schedules and are more expressive than our fixed Euler update.

- GITS / trajectory regularity: https://proceedings.mlr.press/v235/chen24bm.html
- Adaptive Time-Stepping Schedules for Diffusion Models:
  https://proceedings.mlr.press/v244/chen24c.html
- Bespoke Solvers for Generative Flow Models: https://openreview.net/forum?id=1PXEY7ofFX
- Bespoke Non-Stationary Solvers: https://arxiv.org/abs/2403.01329
- A-FloPS adaptive flow-path sampling: https://arxiv.org/abs/2509.00036

Differences to state carefully:

- native FM velocity rather than a required diffusion/noise parameterization;
- population profile rather than per-sample online accept/reject stepping;
- fixed Euler update rather than learned solver coefficients;
- no validation target or reference trajectory when constructing the deployed mesh;
- calibration cost amortized over all subsequent generated samples.

These are meaningful distinctions, not proof of superiority. A BOSS-style dynamic program is
retained on small problems as an offline optimizer of its additive defect surrogate. Because that
surrogate is not the final endpoint metric, it is a strong comparator rather than a true oracle
ceiling.

### Iso-FM And Acceleration-Regularized Training

Iso-FM-style methods alter the learned field by penalizing pathwise material acceleration during
training. SCTW and Adjoint-SCTW leave the checkpoint unchanged and allocate inference effort to the
field it actually learned. The shared material derivative makes Iso-FM scientifically close, but
the intervention and cost model differ.

### Hand Meshes And VSFM

Early-, late-, and symmetric-power grids are simple strong baselines. VSFM coasting and
uniform-deceleration grids add fixed velocity-schedule baselines. A hand grid can win when its
shape matches a benchmark; the paper's test is whether model-calibrated meshes remain useful
without knowing that shape beforehand.

## Explicit Non-Claims

- Nonuniform generative-ODE sampling is not new.
- Adjoint or tangent-linear sensitivity is not new.
- Hutchinson-style randomization is not new.
- The deployed methods are not pressure-oracle samplers.
- Better learned-ODE integration does not automatically imply better FID or Wasserstein distance.
- Pressure-aware training and coupling were explored but were not robust enough to support the
  current paper thesis.

## Novelty Bar

Adjoint-SCTW adds credible method depth only if terminal sensitivity improves or matches SCTW on
most benchmark families and closes a hand-grid gap on at least one non-Burgers problem. A
Burgers-only gain remains scientifically useful, but it should be framed as a sensitivity-analysis
extension rather than a general sampler improvement.
