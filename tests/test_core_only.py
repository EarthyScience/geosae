"""The test suite and the training/inference path must run on the core install.

CI installs the core dependencies only, so importing a plotting module while
collecting the suite — or from any module the CLI imports eagerly — breaks CI
even though the same code passes locally.
"""

import subprocess
import sys

from conftest import VIZ_ONLY_MODULES

# Everything the train/predict path pulls in before any optional feature is used.
CORE_PATH_MODULES = (
    "geosae",
    "geosae.cli",
    "geosae.config",
    "geosae.run",
    "geosae.data.dataset",
    "geosae.models",
    "geosae.predict.writer",
    "geosae.predict.sparsify",
    "geosae.train.trainer",
    "geosae.train.loss",
    "geosae.eval.metrics",
)


def test_collection_imports_no_viz_only_module(modules_imported_at_collection):
    leaked = sorted(set(VIZ_ONLY_MODULES) & modules_imported_at_collection)
    assert not leaked, f"test modules import {leaked} at collection time; CI installs core only"


def test_core_path_imports_no_viz_only_module():
    imports = "; ".join(f"import {m}" for m in CORE_PATH_MODULES)
    script = f"{imports}\nimport sys\nprint([m for m in {VIZ_ONLY_MODULES!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"training/inference path imports {out.stdout.strip()}"
