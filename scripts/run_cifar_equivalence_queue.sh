#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-$HOME/upwind-cfm}"
python_bin="${PYTHON_BIN:-$HOME/upwind-cfm-venv/bin/python3}"
results_root="${RESULTS_ROOT:-$HOME/upwind-cfm-results}"
suite_root="${SUITE_ROOT:-$results_root/sctw_adjoint_suite}"
data_root="${DATA_ROOT:-$HOME/upwind-cfm-data}"
wait_pid="${WAIT_PID:-}"

done_marker="$suite_root/equivalence_image_queue.DONE"
failed_marker="$suite_root/equivalence_image_queue.FAILED"
rm -f "$done_marker" "$failed_marker"
trap 'touch "$failed_marker"' ERR

if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 60
  done
fi

cd "$repo_dir"

run_eval() {
  local run_dir="$1"
  local out_dir="$2"
  local adjoint_seed="$3"
  local estimators="$4"
  local equivalence_profiles="$5"

  "$python_bin" scripts/evaluate_cifar10_checkpoint.py \
    --run-dir "$run_dir" \
    --checkpoint model_ema.pt \
    --device cuda \
    --metric-device cuda \
    --data-root "$data_root" \
    --n-samples 32 \
    --batch-size 32 \
    --metric-batch-size 32 \
    --nfe-values 5,10,20,50 \
    --profile-samples 100 \
    --profile-batch-size 4 \
    --profile-fine-steps 50 \
    --profile-anchor-count 16 \
    --profile-estimators "$estimators" \
    --equivalence-profiles "$equivalence_profiles" \
    --equivalence-powers 0.5 \
    --pilot-seed 1311 \
    --adjoint-probes 2,4,8 \
    --adjoint-seed "$adjoint_seed" \
    --transport-perturbations 0.001,0.003 \
    --warp-powers 0.5 \
    --eval-seed 1234 \
    --methods adjoint_sctw \
    --reference-samples 32 \
    --reference-intervals 1000 \
    --reference-check-intervals 2000 \
    --reference-split train \
    --classifier-checkpoint artifacts/cifar10_metrics/classifier/cifar10_resnet18.pt \
    --inception-weights artifacts/cifar10_metrics/inception/weights-inception-2015-12-05-6726825d.pth \
    --skip-fid-kid \
    --skip-classifier \
    --out-dir "$out_dir"
}

conditional_run="$results_root/cifar10_coupling_large_100k/runs/cifar10_minibatch_ot_large_16x16_seed0"
unconditional_run="$results_root/cifar10_uncond_coupling_large_100k/runs/cifar10_uncond_sinkhorn_ot_large_16x16_seed0"
adjoint_estimators="transport_adjoint"
adjoint_profiles="transport_adjoint_r2,transport_adjoint_r8"
full_estimators="local,transport_fd,transport_jvp,transport_adjoint"
full_profiles="transport_fd_eps0p001,transport_fd_eps0p003,transport_jvp,transport_adjoint_r2,transport_adjoint_r8"

run_eval "$conditional_run" "$suite_root/equivalence/cifar10_conditional_probe7" \
  7 "$adjoint_estimators" "$adjoint_profiles"
run_eval "$conditional_run" "$suite_root/equivalence/cifar10_conditional_probe31415" \
  31415 "$adjoint_estimators" "$adjoint_profiles"
run_eval "$unconditional_run" "$suite_root/equivalence/cifar10_unconditional_probe2718" \
  2718 "$full_estimators" "$full_profiles"
run_eval "$unconditional_run" "$suite_root/equivalence/cifar10_unconditional_probe7" \
  7 "$adjoint_estimators" "$adjoint_profiles"
run_eval "$unconditional_run" "$suite_root/equivalence/cifar10_unconditional_probe31415" \
  31415 "$adjoint_estimators" "$adjoint_profiles"

touch "$done_marker"
