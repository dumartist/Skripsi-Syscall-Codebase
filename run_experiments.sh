#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)

butuh_file() {
  for berkas in "$@"; do
    [[ -s "$berkas" ]] || { echo "File data olahan tidak ditemukan atau kosong: $berkas" >&2; exit 1; }
    if grep -aqm1 '^version https://git-lfs.github.com/spec/v1' < <(head -c 64 "$berkas"); then
      echo "File $berkas masih berupa pointer Git LFS. Jalankan git lfs pull." >&2
      exit 1
    fi
  done
}
adfa_mentah=false
if [[ -n ${ADFA_LD_ROOT:-} || -d archive-kaggle/ADFA-LD ]]; then
  adfa_mentah=true
else
  butuh_file prepared_data/{metadata.json,vocabulary.json,train.npz,test.npz,benign_trigrams.pkl} \
    adversarial_data/{distribution_aware,random,position_aware}_{train,test}.npz
  echo "Memakai prepared_data/ dan adversarial_data/. Uji rasio injeksi dan containment per proses memerlukan trace mentah dan akan dilewati."
fi
if [[ ! -d LID-DS && ! -d LID-DS_extracted ]]; then
  butuh_file prepared_data_lidds/{Bruteforce_CWE-307,CVE-2014-0160,CWE-89-SQL-injection}.npz \
    prepared_data_lidds/vocabulary_native.json
  echo "Memakai data LID-DS di prepared_data_lidds/."
fi
if $adfa_mentah; then
  "$PY" -m syscall_hids.data.adfa_ld_preparation
  "$PY" -m syscall_hids.attacks.adversarial_mimicry
fi
"$PY" -m syscall_hids.training.corrected_pipeline --seeds "${SEEDS[@]}" --out corrected_full.json --skip-adaptive
"$PY" -m syscall_hids.training.corrected_baselines
"$PY" -m syscall_hids.evaluation.make_legacy_shims
"$PY" -m syscall_hids.training.stage_legacy_models
"$PY" -m syscall_hids.evaluation.adaptive_eval --seeds "${SEEDS[@]}" --n 200 --out adaptive_full.json
"$PY" -m syscall_hids.diagnostics.analyze_corrected
"$PY" -m syscall_hids.diagnostics.diagnostic_e4_artifact_only
"$PY" -m syscall_hids.diagnostics.diagnostic_attribution
"$PY" -m syscall_hids.diagnostics.diagnostic_latent_viz
if $adfa_mentah; then "$PY" -m syscall_hids.evaluation.injection_ratio_sweep; fi
"$PY" -m syscall_hids.prevention.prevention_framework
"$PY" -m syscall_hids.prevention.multitier_policy
if $adfa_mentah; then "$PY" -m syscall_hids.prevention.per_process_containment; fi
"$PY" -m syscall_hids.prevention.latency_profiler
"$PY" -m syscall_hids.diagnostics.examiner_analyses
"$PY" -m syscall_hids.prevention.pareto_summary
"$PY" -m syscall_hids.export.export_onnx

if [[ -d LID-DS ]]; then "$PY" -m syscall_hids.data.lidds_extract; fi
if [[ -d LID-DS_extracted ]]; then "$PY" -m syscall_hids.data.lidds_preparation; fi
"$PY" -m syscall_hids.data.lidds_vocab_align
"$PY" -m syscall_hids.evaluation.lidds_eval_zeroshot
"$PY" -m syscall_hids.evaluation.lidds_eval_finetune
"$PY" -m syscall_hids.evaluation.lidds_eval_scratch
"$PY" -m syscall_hids.export.export_onnx_lidds Bruteforce_CWE-307
