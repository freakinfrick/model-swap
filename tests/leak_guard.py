#!/usr/bin/env python3
"""Fail if a machine-specific fact has leaked into the shareable files.

The rule it enforces: all code is shareable, and every fact about one machine
lives in the rig config. A hit is fixed by moving the value to
config.example.yaml's schema and the private config, never by allow-listing.

Checks, over every file this repo tracks (it holds only shareable files):
  1. built-in patterns: home-dir paths, real GPU UUIDs, e-mail addresses,
     tailnet (100.64/10) addresses
  2. quoted GPU / slot names from your own rig config inside the script
     (e.g. "4090") — the code must take names from the config, not hardcode them
  3. private words, one regex per line, from ~/.config/model-swap/leak-denylist.txt
     (kept outside the repo, because the list itself would leak)

Run:      python3 tests/leak_guard.py            (exit 1 on any hit)
As hook:  ln -s ../../tests/leak_guard.py .git/hooks/pre-commit
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "model-swap"
DENYLIST = Path(os.path.expanduser("~/.config/model-swap/leak-denylist.txt"))

PATTERNS = {
    "home-dir path": r"/home/[a-z_][\w-]*|/Users/[A-Za-z][\w-]*",
    "GPU UUID": r"GPU-(?!0{8}-|1{8}-)[0-9a-f]{8}-[0-9a-f]{4}-",
    "e-mail": r"[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[\w.]+",
    "tailnet IP": r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b",
}


def shareable_files():
    """Everything this repo tracks or would track (untracked, not ignored)."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return sorted(ROOT / f for f in out if f and (ROOT / f).is_file())


def rig_names():
    """GPU and slot names from the private config, as quoted literals to forbid."""
    cfg_path = Path(os.environ.get("MODEL_SWAP_CONFIG")
                    or Path.home() / ".config/model-swap/config.yaml").expanduser()
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except OSError:
        return []
    names = {str(g) for g in cfg.get("gpus") or {}}
    names |= {str(s.get("name")) for s in (cfg.get("slots") or {}).values() if s.get("name")}
    names |= {str(s.get("device_alias")) for s in (cfg.get("slots") or {}).values()
              if s.get("device_alias")}
    return sorted(names)


def main():
    rules = dict(PATTERNS)
    if DENYLIST.exists():
        for i, line in enumerate(DENYLIST.read_text().splitlines()):
            if line.strip() and not line.startswith("#"):
                rules[f"denylist:{i + 1}"] = line.strip()
    compiled = {k: re.compile(v, re.I if k.startswith("denylist") else 0) for k, v in rules.items()}
    quoted = [re.compile(r"""["']%s["']""" % re.escape(n)) for n in rig_names()]

    hits = []
    for path in shareable_files():
        rel = path.relative_to(ROOT)
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(lines, 1):
            for label, rx in compiled.items():
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {label}: {line.strip()[:100]}")
            if path == SCRIPT and not line.lstrip().startswith("#"):
                for rx in quoted:
                    if rx.search(line):
                        hits.append(f"{rel}:{n}: rig name literal: {line.strip()[:100]}")
    for h in hits:
        print(h)
    if hits:
        print(f"\nleak_guard: {len(hits)} hit(s) — move the value into the rig config")
        return 1
    print(f"leak_guard: clean ({len(shareable_files())} files, {len(rules)} patterns, "
          f"{len(quoted)} rig names)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
