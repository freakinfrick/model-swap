#!/usr/bin/env python3
"""`model-swap backends` surface test — read-only, no GPU, no contract writes.

Runs the REAL model-swap. Every check is a table/JSON read, so this is safe
while the primary card is serving. The point of the test is anti-rot: the table and
the JSON payload must be DERIVED from the constants (FT_PORT, DS4_BIN_DEFAULT,
VLLM_PYTHON, PARAM_MAP, ALL_SLOTS, the timeout constants) and the profile
counts from <models_dir>/profiles/ — so it asserts against those sources, not
against retyped literals.

Run:  python3 tests/test_backends_cmd.py
"""
import json
import os
import subprocess
import sys

import yaml

from _rig import MS, ms as _ms

PROFILES = str(_ms.PROFILES_DIR)
BACKENDS = ("llama", "vllm", "ds4", "freetoken")

failures = []


def ms(*args):
    r = subprocess.run([sys.executable, MS, *args], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{'' if cond else '  — ' + detail}")
    if not cond:
        failures.append(label)


def constants():
    """Read the constants from the loaded CLI module (rig values come from the
    config file), so a changed constant fails this test instead of silently
    drifting from the surface."""
    return {name: getattr(_ms, name) for name in
            ("FT_PORT", "PORT", "VLLM_DEFAULT_TIMEOUT", "DS4_DEFAULT_TIMEOUT",
             "LLAMA_DEFAULT_TIMEOUT")}


def profile_counts():
    counts = {b: 0 for b in BACKENDS}
    for fn in sorted(os.listdir(PROFILES)):
        if not fn.endswith(".yaml"):
            continue
        try:
            data = yaml.safe_load(open(os.path.join(PROFILES, fn)).read())
        except Exception:
            continue
        if isinstance(data, dict):
            counts[str(data.get("backend", "llama"))] = \
                counts.get(str(data.get("backend", "llama")), 0) + 1
    return counts


def main():
    C = constants()

    # ── bare table ───────────────────────────────────────────────────────
    rc, out = ms("backends")
    check("backends exits 0", rc == 0, out)
    for b in BACKENDS:
        check(f"table lists {b}", b in out)

    # ── the gating contract ──────────────────────────────────────────────
    rc, out = ms("backends", "--json")
    check("backends --json exits 0", rc == 0, out)
    data = json.loads(out)
    check("--json has >= 4 backends", len(data) >= 4, str(list(data)))
    check("--json keys are the backend names", set(BACKENDS) <= set(data))

    required = {"name", "runs", "devices", "port", "mode", "status", "pick_when",
                "health", "param_block", "param_flags", "schema", "lifecycle",
                "gotchas", "profiles", "docs"}
    for b in BACKENDS:
        check(f"{b}: payload has every required key", required <= set(data[b]),
              str(sorted(required - set(data[b]))))
        check(f"{b}: schema is structured", isinstance(data[b]["schema"], list) and all(
            {"key", "type", "required", "default", "meaning"} <= set(f) for f in data[b]["schema"]))
        check(f"{b}: model is a required key", any(
            f["key"] == "model" and f["required"] for f in data[b]["schema"]))
        check(f"{b}: docs path resolves", data[b]["docs"] and os.path.exists(data[b]["docs"]),
              str(data[b]["docs"]))

    # ── derived, not retyped ─────────────────────────────────────────────
    check("freetoken port derives from FT_PORT", data["freetoken"]["port"] == C["FT_PORT"])
    check("llama/vllm/ds4 ports derive from PORT",
          all(data[b]["port"] == C["PORT"] for b in ("llama", "vllm", "ds4")))
    check("vllm timeout derives from VLLM_DEFAULT_TIMEOUT",
          data["vllm"]["health"]["default_timeout_s"] == C["VLLM_DEFAULT_TIMEOUT"])
    check("ds4 timeout derives from DS4_DEFAULT_TIMEOUT",
          data["ds4"]["health"]["default_timeout_s"] == C["DS4_DEFAULT_TIMEOUT"])
    check("llama timeout derives from LLAMA_DEFAULT_TIMEOUT",
          data["llama"]["health"]["default_timeout_s"] == C["LLAMA_DEFAULT_TIMEOUT"])
    check("llama param_flags come from PARAM_MAP",
          data["llama"]["param_flags"].get("ctx_size") == "--ctx-size")
    check("ds4 param_flags include the bool flags",
          data["ds4"]["param_flags"].get("ssd_streaming") == "--ssd-streaming")
    check("freetoken param_flags come from FT_PARAM_MAP",
          data["freetoken"]["param_flags"].get("moe_backend") == "--moe-backend")
    check("llama slots come from ALL_SLOTS",
          {s["slot"] for s in data["llama"]["slots"]} == set(_ms.ALL_SLOTS))

    # ── honest counts and honest status ──────────────────────────────────
    counts = profile_counts()
    for b in BACKENDS:
        check(f"{b}: profile count matches a scan of {PROFILES} ({counts[b]})",
              data[b]["profiles"] == counts[b], f"cli={data[b]['profiles']} scan={counts[b]}")
    check("ds4 is flagged experimental", "experimental" in data["ds4"]["status"].lower())
    check("freetoken is a borrow", data["freetoken"]["mode"] == "borrow")
    check("the other three are standing",
          all(data[b]["mode"] == "standing" for b in ("llama", "vllm", "ds4")))

    # ── deep view ────────────────────────────────────────────────────────
    for b in BACKENDS:
        rc, out = ms("backends", b)
        check(f"backends {b} exits 0", rc == 0, out)
        check(f"backends {b} prints the key schema", "Profile keys:" in out and "Gotchas:" in out)
        rc, out = ms("backends", b, "--json")
        check(f"backends {b} --json is that backend's object", rc == 0 and json.loads(out)["name"] == b)

    rc, out = ms("backends", "nope")
    check("unknown backend is rejected", rc != 0)

    # ── discoverability ──────────────────────────────────────────────────
    rc, out = ms("--help")
    check("--help epilog mentions backends", "backends" in out)

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
