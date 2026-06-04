# Experiment Log: spiral_multiseed_10seed_selected

The `spiral_multiseed_10seed_selected` experiment extends the 5-seed
`spiral_multiseed_selected` pilot to 10 seeds. The goal is to test whether the
observed low-NFE spiral improvements persist strongly enough for paired
seed-level analysis.

This run keeps the focused comparison small:

- `standard_cfm`
- `lc_finite_difference` with weight `4.0`
- `lc_jvp_material_derivative` with weight `0.005`
- `lc_jvp_material_derivative` with weight `0.01`

The finite-difference weight is the strongest setting from the selected pilot.
The two JVP weights test whether stronger material-derivative regularization
can produce a useful smoothness/transport tradeoff, even if it does not beat the
finite-difference formulation.
