"""Single source of truth for every path used by the pipeline.

Resolved relative to the repository root, so the scripts run unchanged on any
machine and from any working directory.
"""
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
RAW      = ROOT / "data" / "raw"
DATA     = ROOT / "data"
TABLES   = ROOT / "results" / "tables"
LOGS     = ROOT / "results" / "logs"
FIGURES  = ROOT / "figures"

for _d in (RAW, DATA, TABLES, LOGS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
