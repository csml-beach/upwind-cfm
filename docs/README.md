# Documentation Map

This folder is organized around the current research direction, not the chronological history of
every idea we tried.

## Active Paper Direction

- `plan.md`: current thesis, method sketch, evidence, non-claims, and immediate next step.
- `literature.md`: closest related work and how SCTW differs.
- `todo.md`: active implementation and experiment checklist.
- `phase1_sampler.md`: detailed experimental record for self-curvature/time-warp samplers.
- `cifar10_benchmark.md`: CIFAR-10 image benchmark setup and results.
- `experiment_spine.md`: code organization, registered datasets, methods, pairings, and solvers.
- `remote_gpu.md`: non-secret VPS/GPU operational notes.

## Archived History

- `archive/`: old proposals, negative results, and earlier benchmark notes. These files are kept
  for memory, but they should not be read as the current paper plan.

The active candidates are **Self-Curvature Time Warping (SCTW)** and **Adjoint-SCTW**:
training-free, oracle-free population timestep meshes for fixed-NFE flow-matching sampling.
Pressure theory remains the motivation and diagnostic lens, but direct pressure
training/coupling is not the main claim at this stage.
