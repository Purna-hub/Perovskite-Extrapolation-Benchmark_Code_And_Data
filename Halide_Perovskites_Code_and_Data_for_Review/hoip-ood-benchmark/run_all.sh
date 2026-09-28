#!/usr/bin/env bash
# Full reproduction of every number, table and figure in the manuscript.
# One CPU core, under one hour. No GPU, no API key, no network after stage 00.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
mkdir -p results/logs

bash src/00_fetch_source_data.sh

run () {                      # run <script> ; tee stdout to results/logs/
  local s="$1"; local log="results/logs/${s%.py}.log"
  echo "--- $s"
  python3 "src/$s" 2>&1 | tee "$log"
}

run 01_parse_structures.py            # CIF        -> data/hoip_structures.csv
run 02_assemble_dataset.py            # merge      -> data/hoip_master.csv
run 03_validate_llm_descriptors.py    # Section 3.6 memorisation control
run 04_protocol_audit.py              # Table 3, Tables S2-S3      (~25 min)
run 05_baselines_and_model_zoo.py     # Tables 4, 5, S4            (~25 min)
run 06_conformal_uncertainty.py       # Table 6, Table S5          (~4 min)
run 07_attribution_and_screening.py   # Figs 3b, 5; Table S9       (~6 min)
run 08_lookup_baselines.py            # Table 2, Table S1
run 09_permutation_control.py         # Table 7, Table S7
run 10_fold_level_metrics.py          # Table S6
run 11_seed_variability.py            # Table S8
run 12_make_figures.py                # Figures 1-5

echo
echo "Done. Compare results/tables/*.csv with the released copies."
