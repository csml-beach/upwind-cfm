# Label Bookkeeping

## Equations

- eq:fm-ode — Learned flow-matching ODE used for sampling.
- eq:material-derivative — Material derivative of the learned velocity field.
- eq:cfm-objective — Standard paired linear-interpolant CFM regression objective.
- eq:euler-grid — Euler update on a nonuniform time grid.
- eq:euler-expansion — Taylor expansion showing Euler local error and material derivative.
- eq:euler-defect — Leading material-acceleration term in the discrete Euler defect.
- eq:flow-tangent — State tangent of the learned flow map from an intermediate time to the endpoint.
- eq:terminal-defect — First-order propagation of a local Euler defect to terminal time.
- eq:transported-profile — Local SCTW and terminally transported Adjoint-SCTW profiles.
- eq:self-curvature-estimator — Finite-difference self-curvature estimator used by SCTW.
- eq:tangent-linear-model — Continuous tangent-linear sensitivity equation initialized by a local acceleration.
- eq:discrete-heun-step — Fine-pilot discrete Heun state update.
- eq:discrete-heun-tangent — Exact JVP propagation through a discrete Heun step.
- eq:transport-finite-difference — Centered finite-difference estimate of transported acceleration.
- eq:discrete-adjoint — State adjoint induced by a Gaussian terminal probe.
- eq:randomized-norm-identity — Randomized identity for the squared transported-defect norm.
- eq:randomized-adjoint-estimator — Practical R-probe Adjoint-SCTW estimator.
- eq:mesh-density — Tempered density used to construct SCTW knots.
- eq:equidistribution — Equal cumulative mass rule for SCTW time knots.
- eq:conditional-covariance — Conditional covariance/pressure tensor for linear-interpolant FM.
- eq:pressure-identity — Exact pressure/momentum identity for the marginal FM velocity.
- eq:oracle-material — Oracle material-derivative side of the appendix verification.
- eq:oracle-pressure — Oracle pressure-source side of the appendix verification.

## Figures

- fig:profile-comparison — Normalized local and terminal-sensitivity profiles with NFE-20 knot rugs.
- fig:staged-samples — Uniform, best-fixed, and Adjoint-SCTW staged-shapes samples at NFE 5.
- fig:lowdim-panels — Low-dimensional SCTW diagnostic panels for five modes, fan modes, and spiral.
- fig:burgers-data — Rich Burgers initial and final PDE reference examples.
- fig:burgers-diagnostic — Burgers self-curvature profile, NFE-20 meshes, and metrics.

## Tables

- tab:benchmarks — Benchmark purposes and primary metrics.
- tab:cifar-conditional — Conditional CIFAR-10 FID comparison across time meshes.
- tab:cifar-unconditional — Unconditional CIFAR-10 FID comparison across time meshes.
- tab:staged — Staged-shapes feature W1 comparison across time meshes.
- tab:gm16 — Three-checkpoint GM16 endpoint W1 comparison.
- tab:burgers-integration — Burgers learned-ODE integration error comparison.
- tab:oracle-verification — Oracle pressure-identity verification checks.

## Algorithms

No algorithm labels yet.

## Sections

- sec:introduction — Introduction and literature narrative.
- sec:method — Theory and method.
- sec:flow-matching-background — Flow matching setup.
- sec:material-error — Material derivative and Euler local error.
- sec:transported-defect — Terminal transport of local Euler defects.
- sec:self-curvature-probe — Self-curvature probe estimator.
- sec:transport-estimators — Exact JVP and centered finite-difference verification estimators.
- sec:randomized-adjoint — Scalable randomized discrete-adjoint estimator.
- sec:mesh-rule — SCTW mesh construction.
- sec:offline-calibration — Frozen-checkpoint population calibration and cache protocol.
- sec:pressure-identity — Pressure identity motivation.
- sec:process — Development process and lessons learned.
- sec:experiments — Experimental design.
- sec:samplers — Compared sampler time meshes.
- sec:calibration-protocol — Calibration defaults and FD/JVP/adjoint equivalence tier.
- sec:benchmarks — Benchmark suite and metrics.
- sec:figures — Figure categories.
- sec:results — Results.
- sec:estimator-results — FD/JVP/adjoint estimator verification.
- sec:alignment-results — Relationship between integration error and endpoint quality.
- sec:staged-results — Staged-shapes endpoint and integration results.
- sec:lowdim-results — Visual low-dimensional and GM16 results.
- sec:cifar-results — CIFAR-10 results.
- sec:burgers-results — Hard Burgers solution-map results.
- sec:summary-evidence — Evidence summary.
- sec:conclusion — Conclusion.
- sec:data-availability — Data and code availability.
- sec:appendix-oracle — Oracle verification appendix.
