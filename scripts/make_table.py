"""Build performance tables from ONE logged run. Usage:
  python scripts/make_table.py results/runs/<run_id>      (default: latest compare_* run)
Writes results/tables/<run_id>.csv (by qtype) and <run_id>_summary.csv (system x budget). run_compare.py
calls this automatically at the end of every run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402
from core.eval.table import build_tables  # noqa: E402

runs = REPO_ROOT / "results" / "runs"
run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(runs.glob("*_compare_*"))
run = run if run.is_absolute() else REPO_ROOT / run
for p in build_tables(run, REPO_ROOT / "results" / "tables"):
    print(f"wrote {p.relative_to(REPO_ROOT)}")
