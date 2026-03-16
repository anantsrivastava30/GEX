from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


CONFIG = load_config()

