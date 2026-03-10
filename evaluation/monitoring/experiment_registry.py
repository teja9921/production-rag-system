import json
from datetime import datetime
from pathlib import Path


REGISTRY_PATH = Path("evaluation/experiment_registry.json")


def _load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_registry(data):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def register_experiment(
    dataset_version: str,
    config_id: str,
    metrics: dict,
):
    registry = _load_registry()

    if dataset_version not in registry:
        registry[dataset_version] = {}

    registry[dataset_version][config_id] = {
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _save_registry(registry)
