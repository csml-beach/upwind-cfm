# Experiment Notes

Brief reference so we do not repeat discarded branches.

- The five-mode benchmark became useful only after making it harder: radius 8, 3 Euler inference steps, and a 0.4 hit radius. The earlier radius 6 / 5-step setup was too forgiving.
- Uniform FD residual reproduces Iso-FD closely. This is a useful implementation check: the Directional-Regularization CFM path can express the Iso-FD-style penalty.
- JVP material residuals improved smoothness but did not close the gap to Iso-FD on endpoint quality. The finite-difference, stop-gradient lookahead appears to matter.
- The JVP material derivative implementation is value-correct: it matches small-epsilon finite differences on trained models. The problem is optimization alignment. The Iso-FD stop-gradient loss and the JVP derivative loss have similar scalar values but nearly opposite parameter-gradient directions on the same batch, so they should not be treated as interchangeable training objectives.
- Variance-gate residual weighting is shelved. Kernel/kNN uncertainty gates reduced regularization in ambiguous regions, but on this benchmark they mostly weakened mode capture instead of improving solver behavior.
- Directional-Regularization CFM is the current live branch. The first promising result was FD residual plus a JVP directional weight at weight 10 on the hard five-mode benchmark: slightly better Wasserstein than Iso-FD in one seed, with similar hit rate. This needs a multi-seed check before it becomes a claim.
- The 10-seed hard five-mode sweep did not support a robust improvement for Directional-Regularization CFM at weight 10. Directional FD+JVP had worse mean Wasserstein than Iso-FD, higher acceleration, and lower mean coverage, although it achieved the best hit rate in 6/10 seeds. Treat it as a tuning candidate, not as a demonstrated win.
- A 6x4 FD+FD tuning grid over weight and directional_dt found useful tradeoff regions but no mean Wasserstein win over Iso-FD. Best W in the grid was weight 40, directional_dt 0.2 (W 0.817), still worse than Iso-FD (W 0.782). The most interesting hit/acceleration tradeoffs were weight 20, directional_dt 1/3 (hit 60.4%, acc 1.06) and weight 10, directional_dt 0.5 (hit 57.5%, acc 0.85), but both have worse W than Iso-FD.
