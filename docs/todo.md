# Active Todo: SCTW and Adjoint-SCTW

## Engineering

- [x] Implement constant-memory stepwise discrete adjoint.
- [x] Add the shared profile bundle/cache API.
- [x] Use one sampler vocabulary: `uniform`, `sctw`, `adjoint_sctw`, `power`, `vsfm`, `heun`.
- [x] Bind conditional CIFAR labels exactly during calibration.
- [x] Add analytic tangent, FD/JVP, cache replay, and NFE-accounting checks.
- [x] Add suite-level endpoint and profile-equivalence aggregation.
- [x] Cache versioned real Inception features and pair KID subset randomness across methods.
- [x] Complete staged-shapes and CIFAR GPU endpoint integration checks.
- [x] Add amortized calibration cost columns to the aggregate report.

## Local Experiments

- [x] Five modes full endpoint evaluation.
- [x] Fan modes full endpoint evaluation.
- [x] Spiral full endpoint evaluation.
- [x] GM16 seeds 0, 1, 2 full endpoint evaluation.
- [x] Hard Burgers full endpoint and equivalence evaluation.
- [x] Run the local equivalence tier with all declared probe seeds.
- [x] Check local Heun-1000 versus Heun-2000 acceptance ratios.

## Remote Experiments

- [x] Staged-shapes endpoint suite.
- [x] Conditional CIFAR endpoint suite.
- [x] Unconditional CIFAR endpoint suite.
- [x] Staged-shapes small equivalence pilot, including all three declared adjoint probe seeds.
- [x] Conditional CIFAR small equivalence pilot with all three declared probe seeds.
- [x] Unconditional CIFAR small equivalence pilot with all three declared probe seeds.
- [x] Check image Heun-1000 versus Heun-2000 reference convergence.
- [x] Sync profile bundles, grids, metrics, selected sample grids, and runtime summaries.

## Analysis And Paper

- [x] Add TLM, FD, discrete-adjoint, and randomized-estimator derivations.
- [x] Add adjoint-model, DWR, neural-ODE, and Hutchinson references.
- [x] Generate local aggregate endpoint and equivalence tables from artifacts.
- [x] Revise abstract, introduction, results, and conclusion after all endpoint runs finish.
- [x] Retain the five-modes FD failure and distinguish integration error from endpoint quality.
- [x] Compile the current LaTeX draft and verify every label and citation.
- [x] DVC-track the selected aggregate result group and paper-facing figures.

## Acceptance Gates

- FD/JVP normalized profile L2 below `2%` and maximum knot displacement below `0.02`.
- Adjoint `R=4` profile L2 below `10%` and integration error within `5%` of JVP on equivalence cases.
- Probe-seed endpoint integration variation below `5%`.
- Reference self-error below `10%` of the smallest reported method gap; increase resolution if not.
- Cached profiles reproduce identical grids and samples.

These gates diagnose the method; they are not filters for suppressing negative outcomes.
