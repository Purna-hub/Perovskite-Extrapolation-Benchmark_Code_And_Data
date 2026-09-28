#!/usr/bin/env bash
# Stage 00 - retrieve the primary source data.
#
# Source: Kim, Huan, Krishnan and Ramprasad, "A hybrid organic-inorganic perovskite
# dataset", Scientific Data 4 (2017) 170057.  Please cite the original paper.
# The dataset is not redistributed in this archive; it is downloaded into data/raw/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"
cd "$RAW"
curl -sSL -o hoip_source.zip \
  "https://codeload.github.com/Abdulraheem6355/HOIPs-Bandgap-Data-set/zip/refs/heads/main"
unzip -oq hoip_source.zip
unzip -oq HOIPs-Bandgap-Data-set-main/HOIP_cif_merge.zip -d HOIPs-Bandgap-Data-set-main/cifs
n=$(find HOIPs-Bandgap-Data-set-main/cifs -name '*.cif' | wc -l)
echo "CIF files retrieved: $n  (expected 1346)"
echo "Properties table:    HOIPs-Bandgap-Data-set-main/HOIP_Extracted_dataset.xlsx"
