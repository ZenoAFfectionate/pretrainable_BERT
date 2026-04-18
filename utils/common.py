"""Common helper utilities shared across the package."""

from typing import Dict, List, Optional

import yaml


def parse_cuda_devices(cuda_devices):
    """Normalize cuda_devices spec into list[int] or None.

    Accepts None, "0,1,2", [0, 1], 0, etc.
    """
    if cuda_devices is None:
        return None
    if isinstance(cuda_devices, str):
        parts = [x.strip() for x in cuda_devices.split(',') if x.strip()]
        return [int(x) for x in parts] if parts else None
    if isinstance(cuda_devices, (list, tuple)):
        return [int(x) for x in cuda_devices]
    return [int(cuda_devices)]


def extract_config_path(argv: List[str]) -> Optional[str]:
    """Return the value of ``--config`` in ``argv`` without calling argparse.

    Supports both ``--config path`` and ``--config=path`` styles. Returns
    ``None`` if absent. Used for the two-phase pre-scan that lets a YAML
    config replace argparse defaults before the full parse runs.
    """
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
        i += 1
    return None


def apply_yaml_defaults(parser, config_path: str) -> Dict:
    """Overlay values from a YAML file onto the given argparse parser.

    YAML keys are matched case-sensitively against argparse ``dest`` names.
    Unknown keys are logged and ignored (not an error — a single YAML can
    feed multiple commands that share most flags).

    Callers invoke this **before** ``parser.parse_args()``, so any flag the
    user provides on the command line still wins (argparse.set_defaults only
    replaces the default, not an explicit CLI value).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config file {config_path!r} must define a top-level YAML mapping "
            f"(got {type(cfg).__name__})"
        )

    known = {a.dest for a in parser._actions}
    applied = {}
    unknown = []
    for key, value in cfg.items():
        if key in known:
            applied[key] = value
        else:
            unknown.append(key)

    if unknown:
        print(f"[config] warning: ignoring unknown keys from {config_path}: {unknown}")
    if applied:
        parser.set_defaults(**applied)
    return applied
