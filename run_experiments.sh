#!/usr/bin/env bash
# Jalankan dari akar codebase. Data mentah harus tersedia secara lokal.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)

"$PY" -m syscall_hids.data.adfa_ld_preparation
"$PY" -m syscall_hids.attacks.adversarial_mimicry
"$PY" -m syscall_hids.training.corrected_pipeline --seeds "${SEEDS[@]}" --out corrected_full.json --skip-adaptive
"$PY" -m syscall_hids.training.corrected_baselines
"$PY" -m syscall_hids.evaluation.make_legacy_shims
"$PY" -m syscall_hids.training.stage_legacy_models
"$PY" -m syscall_hids.evaluation.adaptive_eval --seeds "${SEEDS[@]}" --n 200 --out adaptive_full.json
"$PY" -m syscall_hids.diagnostics.analyze_corrected
"$PY" -m syscall_hids.diagnostics.diagnostic_e4_artifact_only
"$PY" -m syscall_hids.diagnostics.diagnostic_attribution
"$PY" -m syscall_hids.diagnostics.diagnostic_latent_viz
"$PY" -m syscall_hids.evaluation.injection_ratio_sweep
"$PY" -m syscall_hids.prevention.prevention_framework
"$PY" -m syscall_hids.prevention.multitier_policy
"$PY" -m syscall_hids.prevention.per_process_containment
"$PY" -m syscall_hids.prevention.latency_profiler
"$PY" -m syscall_hids.diagnostics.examiner_analyses
"$PY" -m syscall_hids.prevention.pareto_summary
"$PY" -m syscall_hids.export.export_onnx

"$PY" -m syscall_hids.data.lidds_extract
"$PY" -m syscall_hids.data.lidds_preparation
"$PY" -m syscall_hids.data.lidds_vocab_align
"$PY" -m syscall_hids.evaluation.lidds_eval_zeroshot
"$PY" -m syscall_hids.evaluation.lidds_eval_finetune
"$PY" -m syscall_hids.evaluation.lidds_eval_scratch
"$PY" -m syscall_hids.export.export_onnx_lidds Bruteforce_CWE-307
