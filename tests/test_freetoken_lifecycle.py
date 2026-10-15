#!/usr/bin/env python3
"""FreeToken backend lifecycle test (SPEC §5, covenant rule 11).

Drives the REAL model-swap against the fake `ft` (tests/ft-shim/ft), which
binds the real FreeToken port — so load/stop/reserve/contract round-trip is
proven without a 30GB model or a GPU. It borrows the first single-card slot's
card; the primary must stay bit-for-bit untouched (asserted before/after).
All single-card slots must be idle (aborts otherwise — the test would
stop/restore them). Card names, ports and paths come from the rig config.

Run:  python3 tests/test_freetoken_lifecycle.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import yaml

from _rig import MS, ms as _ms, solo_slots

PROFILES = str(_ms.PROFILES_DIR)
PROFILE_NAME = "zz-test-freetoken-shim"
PROFILE_PATH = os.path.join(PROFILES, PROFILE_NAME + ".yaml")
SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft-shim", "ft")
CONTRACT = str(_ms.CONTRACT_FILE)
STATE_FT = str(_ms.FT_STATE_FILE)
RESERVATION = str(_ms.RESERVATION_FILE)
PORT = _ms.FT_PORT
SOLO = solo_slots()                          # slot key -> card name
BORROW = next(iter(SOLO.values()), None)     # card the shim borrows
SERVED = "FT-SHIM-TEST"

failures = []


def ms(*args, timeout=300):
    r = subprocess.run([MS, *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + ("" if cond else f" — {detail}"))
    if not cond:
        failures.append(name)


def primary_identity(status_out):
    """(profile, model) from `model-swap status` text; ("", "") when idle."""
    def field(name):
        if f"{name}:" not in status_out:
            return ""
        return status_out.split(f"{name}:")[1].splitlines()[0].strip()
    return field("Profile"), field("Model")


def port_open(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True
    except Exception:
        return False


def contract_models():
    with open(CONTRACT) as f:
        return yaml.safe_load(f).get("models", {})


def cleanup_profile():
    try:
        os.unlink(PROFILE_PATH)
    except OSError:
        pass


def main():
    # ── preconditions ────────────────────────────────────────────────
    print("preconditions")
    if BORROW is None:
        print("SKIP: needs a single-card slot off the primary card in the config")
        return 0
    check(f"port {PORT} free", not port_open(PORT))
    check("no live freetoken state", not os.path.exists(STATE_FT))
    check("no active reservation", not os.path.exists(RESERVATION))
    for key, gpu in SOLO.items():
        r, out = ms("slot", key, "status")
        check(f"{gpu} idle", "stopped" in out, out.strip())
    if failures:
        print(f"ABORT: preconditions failed (single-card slots must be idle, {PORT} free)")
        return 1

    r, baseline = ms("status")
    base_profile, base_model = primary_identity(baseline)
    print(f"  baseline primary: profile={base_profile!r} model={base_model!r}")

    # ── write shim profile ───────────────────────────────────────────
    prof = {
        "model": "shim/test-model",  # shim ignores it; exercises the repo-ID branch
        "description": "freetoken lifecycle test (fake ft shim)",
        "backend": "freetoken",
        "device": BORROW,
        "served_name": SERVED,
        "ft_bin": SHIM,
        "reserve_timeout": 3600,
        "ft_params": {"moe_backend": "offload", "memory_ratio": 0.9},
    }
    with open(PROFILE_PATH, "w") as f:
        yaml.dump(prof, f)

    try:
        # ── dry run: no state change ─────────────────────────────────
        print("dry-run")
        r, out = ms("load", PROFILE_NAME, "--dry-run")
        check("dry-run rc 0", r == 0, out)
        check("dry-run shows shim cmd", SHIM in out and "ft serve" in out, out)
        check("dry-run left no state", not os.path.exists(STATE_FT))

        # ── load (borrow a slot card → serve → contract) ────────────────
        print("load")
        r, out = ms("load", PROFILE_NAME)
        check("load rc 0", r == 0, out)
        check("load reports served name", SERVED in out, out)
        models = contract_models()
        check("contract models entry", models.get(SERVED) == f"http://127.0.0.1:{PORT}",
              str(models))
        # pi-local resolves the endpoint by PROFILE STEM (gc.acquire →
        # endpoint_for_model), not by served name — both forms must resolve
        # to the ft port (SPEC §12 proxy-compat).
        check("contract models entry keyed by profile stem",
              models.get(PROFILE_NAME) == f"http://127.0.0.1:{PORT}", str(models))
        with open(RESERVATION) as f:
            res = json.load(f)
        check("reservation written",
              res.get("reason") == f"freetoken:{PROFILE_NAME}"
              and res.get("devices") == [_ms.SECONDARY_ALIAS], str(res))
        check(f"shim healthy on {PORT}", port_open(PORT))
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
        check("shim /v1/models lists served name",
              any(d.get("id") == SERVED for d in data.get("data", [])), str(data))
        r, out = ms("status")
        check("status reports FreeToken", "FreeToken" in out, out)

        # ── ensure proxy-compat (SPEC §12) ──────────────────────────
        # ensure must not fail while a freetoken borrow is live: the gateway
        # routes the served name, so the borrow IS the ready LLM. (With the
        # primary up, ensure returns via the primary branch; the ft
        # branch is exercised for real by the §7 borrow run's gpu-ensure.)
        print("ensure proxy-compat")
        r, out = ms("ensure", "--quiet")
        check("ensure rc 0 while ft borrow live", r == 0, out)
        req = urllib.request.Request(
            "http://127.0.0.1:8088/v1/chat/completions",
            data=json.dumps({"model": SERVED,
                             "messages": [{"role": "user",
                                           "content": "ping the shim"}],
                             "max_tokens": 32}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            check("gateway :8088 routes served name → shim",
                  data.get("model") == SERVED and "FT-SHIM-OK" in content,
                  str(data)[:200])
        except Exception as e:
            check("gateway :8088 routes served name → shim", False, str(e))

        # ── contract honesty: a slot-card borrow must not repoint `primary` ──
        # (the primary-card branch of the same rule is what lets a consumer fall
        # back to a LIVE endpoint during a primary borrow instead of a dead port)
        print("contract honesty")
        with open(CONTRACT) as f:
            _c = yaml.safe_load(f)
        check("slot-card borrow leaves primary port alone",
              _c.get("primary", {}).get("port") == _ms.PORT, str(_c.get("primary")))
        check("slot-card borrow does not mark primary borrowed",
              "borrowed_by" not in _c.get("primary", {}), str(_c.get("primary")))

        # ── stop (SIGTERM → release → card returned) ─────────────────
        print("stop")
        r, out = ms("stop")
        check("stop rc 0", r == 0, out)
        r = subprocess.run(["pgrep", "-f", SHIM], capture_output=True)
        check("shim process gone", r.returncode != 0)
        check("freetoken state cleared", not os.path.exists(STATE_FT))
        check("reservation released", not os.path.exists(RESERVATION))
        check("contract entry gone", SERVED not in contract_models())
        check(f"port {PORT} released", not port_open(PORT))
        # NOTE: no "second stop is a clean noop" check here. With the borrow gone,
        # a bare `stop` is not a noop — it falls through to the generic path and
        # stops whatever the primary is serving, which is exactly what the
        # "primary untouched" assertion below forbids.

        # ── crash recovery: engine dies, reservation must still clear ──
        # The original failure: the ft process
        # died and `stop` took the llama branch, printed "No server running",
        # exited 0, and left the card reserved until its 4h TTL.
        print("crash recovery")
        r, out = ms("load", PROFILE_NAME)
        check("reload for crash test rc 0", r == 0, out)
        with open(STATE_FT) as f:
            crash_pid = json.load(f)["pid"]
        os.kill(crash_pid, 9)
        for _ in range(20):
            if not port_open(PORT):
                break
            time.sleep(0.5)
        check("shim killed", not port_open(PORT))
        check("reservation still held after crash", os.path.exists(RESERVATION))
        r, out = ms("stop")
        check("stop rc 0 after crash", r == 0, out)
        check("crashed borrow released the card", not os.path.exists(RESERVATION), out)
        check("crashed borrow cleared its state", not os.path.exists(STATE_FT))

        # ── dead engine behind a green /health ───────────────────────
        # The defect in its purest form: frontend up, /health 200,
        # engine gone. The load must FAIL and hand the card back — not report
        # "Ready in 5.6s" and publish a corpse to the contract.
        print("dead engine behind green /health")
        prof["env"] = {"FT_SHIM_DEAD_ENGINE": 1}
        with open(PROFILE_PATH, "w") as f:
            yaml.dump(prof, f)
        r, out = ms("load", PROFILE_NAME, "--timeout", "20")
        check("load fails when the engine is dead", r != 0, out)
        check("failure names the completion probe, not /health",
              "serve a completion" in out, out)
        check("failed load published no contract entry",
              SERVED not in contract_models())
        check("failed load released the reservation",
              not os.path.exists(RESERVATION), out)
        check("failed load left no freetoken state", not os.path.exists(STATE_FT))
        check("failed load did not skip the restore",
              f"skipping {_ms.PRIMARY_GPU} restore" not in out, out)
        prof.pop("env")
        with open(PROFILE_PATH, "w") as f:
            yaml.dump(prof, f)

        # ── primary untouched ───────────────────────────────────────────
        print("primary integrity")
        r, out = ms("status")
        # Compare identities, so an idle primary (no Profile:/Model: lines)
        # counts as unchanged when it was idle at baseline too.
        same = primary_identity(out) == (base_profile, base_model)
        check("primary profile+model unchanged", same, out)
    finally:
        # belt and braces: never leave a shim or test profile behind
        if os.path.exists(STATE_FT):
            ms("stop")
        cleanup_profile()

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed: {failures}")
        return 1
    print("PASS: freetoken backend lifecycle (load/stop/reserve/contract/restore)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
