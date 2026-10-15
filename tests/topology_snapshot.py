#!/usr/bin/env python3
"""Dump everything model-swap derives from its rig config, as stable JSON.

Refactor guard: run before and after a config/topology change and diff. Loads
the script as a module (no command runs), writes the contract to a temp file
instead of ~/gpu-contract.yaml, and strips only the timestamp. Live fields
(pid, state) are kept, so run before/after back to back with no loads between.

Run:  python3 tests/topology_snapshot.py > snap.json
"""
import json
import sys
import tempfile
from pathlib import Path

import yaml

from _rig import load


def plain(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in (sorted(x) if isinstance(x, set) else x)]
    return x


def main():
    ms = load()
    names = ["LLAMA_BIN_DEFAULT", "MODELS_DIR", "PROFILES_DIR", "STATE_FILE",
             "LOG_FILE", "HOST", "PORT", "DEVICE", "GPU_UUID", "HEALTH_URL",
             "MODELS_URL", "GPU_VRAM_TOTAL_MB", "DEFAULT_PROFILE", "VLLM_PYTHON",
             "DS4_BIN_DEFAULT", "FT_BIN_DEFAULT", "FT_PORT", "FT_LOG_FILE",
             "FT_DEVICES", "FT_DEVICE_ARCH", "FT_DEVICE_UUID", "_UUID_TO_PCI",
             "ALL_SLOTS", "ARCHIVE_REMOTE",
             "_LEASE_DEVICE_MAP", "_KNOWN_SERVICE_PATTERNS"]
    out = {n: plain(getattr(ms, n, "<MISSING>")) for n in names}
    # Lazily derived since the config refactor; same keys as the old constants.
    if hasattr(ms, "_ft_devices"):
        out["FT_DEVICES"] = plain(ms._ft_devices())
        out["FT_DEVICE_ARCH"] = {d: ms._gpu_arch(d) for d in ms.GPUS}
        out["FT_DEVICE_UUID"] = {d: g["uuid"] for d, g in ms.GPUS.items()}
        out["_UUID_TO_PCI"] = {g["uuid"]: ms._uuid_to_pci(g["uuid"]) for g in ms.GPUS.values()}

    with tempfile.TemporaryDirectory() as td:
        ms.CONTRACT_FILE = Path(td) / "contract.yaml"
        ms.write_contract()
        contract = yaml.safe_load(ms.CONTRACT_FILE.read_text())
    contract.pop("_updated", None)
    if (contract.get("lease") or {}).get("expires_in_s") is not None:
        contract["lease"]["expires_in_s"] = "<volatile>"
    if (contract.get("reservation") or {}).get("expires_in_s") is not None:
        contract["reservation"]["expires_in_s"] = "<volatile>"
    out["contract"] = plain(contract)
    json.dump(out, sys.stdout, indent=1, sort_keys=True, default=str)
    print()


if __name__ == "__main__":
    main()
