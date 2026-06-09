# Ring-Mode Transport Benchmarks

## Purpose

The ring-mode benchmarks test whether directional regularization helps when a compact source distribution must split into several target modes. This is complementary to the fan benchmark:

- `fan_modes` has a coherent left-to-right advective direction before branching.
- `five_modes` places five modes on a ring around the source, so there is no single preferred outgoing direction.
- `five_modes` with `source_std = 0.15` makes the source tightly clumped near the origin and stresses mode allocation from an ambiguous source.

These benchmarks are deliberately small. Their job is to expose mechanism and failure modes before we scale to images or higher-dimensional latent problems.

## Implemented Variants

The `five_modes` dataset now supports optional source parameters:

```json
"dataset_kwargs": {
  "source_mean": [0.0, 0.0],
  "source_std": 0.15
}
```

Training also supports optional minibatch OT pairing:

```json
"pairing": "minibatch_ot"
```

OT pairing reorders each minibatch target set by Hungarian assignment on `||x0_i - x1_j||^2` before the method loss is evaluated. Methods are unchanged.

## Ring, Standard Prior

Setup:

- Dataset: `five_modes`
- Source: default `N(0, I)`
- Pairing: independent
- Solver: Euler, 5 steps
- Seeds: 0-9

| Method | Wasserstein | Hit rate | Coverage | Acceleration |
| --- | ---: | ---: | ---: | ---: |
| Standard CFM | 0.4965 +/- 0.0963 | 0.7679 +/- 0.0264 | 5.0/5 | 0.3550 +/- 0.0080 |
| Iso-FD w0.5 | 0.7272 +/- 0.1707 | 0.5669 +/- 0.0985 | 5.0/5 | 0.0692 +/- 0.0025 |
| Directional FD+FD w2 | 0.3718 +/- 0.0958 | 0.8791 +/- 0.0148 | 5.0/5 | 0.3009 +/- 0.0084 |
| Directional LVV w2 | 0.3796 +/- 0.0984 | 0.8704 +/- 0.0132 | 5.0/5 | 0.3091 +/- 0.0096 |

Takeaway: directional regularization improves endpoint quality and hit rate over standard CFM while reducing acceleration somewhat. Iso-FD is much smoother but loses endpoint quality. Local velocity variance does not improve over the plain directional FD+FD variant here.

Artifacts:

- `results/five_modes_ring_w2_multiseed/summary.csv`
- `results/five_modes_ring_w2_multiseed/ring_seed8_comparison.png`

## Clumped Source, Independent Pairing

Setup:

- Dataset: `five_modes`
- Source: `0.15 * N(0, I)`
- Pairing: independent
- Solver: Euler, 5 steps
- Seeds: 0-9

| Method | Wasserstein | Hit rate | Coverage | Acceleration |
| --- | ---: | ---: | ---: | ---: |
| Standard CFM | 0.8025 +/- 0.0942 | 0.4644 +/- 0.0405 | 4.9/5 | 0.2162 +/- 0.0178 |
| Iso-FD w0.5 | 0.8190 +/- 0.1189 | 0.3897 +/- 0.0813 | 3.1/5 | 0.0871 +/- 0.0114 |
| Directional FD+FD w2 | 0.5125 +/- 0.1021 | 0.7616 +/- 0.0251 | 5.0/5 | 0.1042 +/- 0.0020 |
| Directional LVV w2 | 0.5339 +/- 0.1272 | 0.7486 +/- 0.0229 | 5.0/5 | 0.1105 +/- 0.0027 |

Takeaway: this is currently the strongest toy evidence for directional regularization. Standard CFM keeps some coverage but lands weakly; Iso-FD is smooth but loses coverage; directional FD+FD keeps full coverage, high hit rate, and low acceleration. LVV again tracks but does not beat plain directional FD+FD.

Artifacts:

- `results/five_modes_clumped015_multiseed/summary.csv`
- `results/five_modes_clumped015_multiseed/clumped015_seed8_comparison.png`
- `results/five_modes_clumped015_multiseed/clumped015_lvv_seed8_gate_only.png`

## Clumped Source, Minibatch OT Pairing

Setup:

- Dataset: `five_modes`
- Source: `0.15 * N(0, I)`
- Pairing: `minibatch_ot`
- Solver: Euler, 5 steps
- Seeds: 0-9

| Method | Wasserstein | Hit rate | Coverage | Acceleration |
| --- | ---: | ---: | ---: | ---: |
| Standard CFM + OT | 0.8058 +/- 0.0620 | 0.4219 +/- 0.0142 | 5.0/5 | 0.0575 +/- 0.0027 |
| Iso-FD w0.5 + OT | 1.1536 +/- 0.3214 | 0.1925 +/- 0.0665 | 1.9/5 | 0.0917 +/- 0.0179 |
| Directional FD+FD w2 + OT | 0.6708 +/- 0.0678 | 0.5261 +/- 0.0174 | 5.0/5 | 0.0479 +/- 0.0026 |

Takeaway: OT pairing makes trajectories very smooth but hurts endpoint concentration on this problem. Directional FD+FD remains compatible with OT and beats the OT baselines, but its best clumped-source result is still with independent pairing. Iso-FD + OT is brittle here and loses coverage.

Artifacts:

- `results/five_modes_clumped015_ot_comparison/summary.csv`
- `results/five_modes_clumped015_ot_comparison/clumped015_ot_seed8_comparison.png`

## Current Interpretation

The result should not be overclaimed. We may be measuring problem bias: the clumped ring favors methods that allow enough directional change to split mass, while the fan benchmark has different geometry. Still, the current evidence suggests:

- Directional FD+FD is the most promising main-method candidate among current variants.
- Iso-FD is a necessary baseline because it is smooth and close in spirit, but it can over-smooth multimodal allocation.
- Local velocity variance is scientifically sound as an ambiguity gate, but the present minibatch-kernel version has not yet improved over plain directional regularization.
- Minibatch OT pairing is an important control, not an automatic improvement.

Next comparisons should test whether these conclusions persist on additional geometries and higher-dimensional problems.
