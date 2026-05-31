from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

scaffold_cv = importlib.import_module("scripts.5_cross_validation")
scaffold_cv.MODEL_NAME = "Dual-FluoGPS"
scaffold_cv.DEFAULT_LOG_DIR = ROOT_DIR / "outputs" / "runs" / "dual_fluogps_scaffold_cv"
scaffold_cv.USE_DUAL_GRAPH = True
scaffold_cv.DEFAULT_DUAL_WEIGHT_MODE = "separate"


if __name__ == "__main__":
    print("Running Dual-FluoGPS scaffold 5-fold cross validation.")
    scaffold_cv.main()
