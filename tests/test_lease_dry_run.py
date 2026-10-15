#!/usr/bin/env python3
"""`model-swap lease --dry-run` must be side-effect free.

Regression test. This bug class has now landed twice in model-swap: `slot`
silently ignored --dry-run (so a dry-run `slot ... ensure` of a pool spanning
the primary really stopped the primary), and then `lease` did the same — `lease acquire
--dry-run` took the lease for real AND ran the eviction path, so it could stop a
serving slot or kill the primary server.

The invariants, asserted against bytes on disk rather than against what the
command prints:

  1. No mutating lease action alters ~/models/.gpu-lease.json or
     ~/gpu-contract.yaml under --dry-run.
  2. A dry run still runs the CHECKS — a conflicting acquire must still be
     denied, or the dry run is lying about what would happen.
  3. The real actions still work (guard against a fix that disables them).
  4. --json emits the {"dry_run": true, ...} envelope, matching `slot`.

Takes and releases a lease on a non-primary card (the last single-card slot's),
so anything that yields to leases (e.g. a background GPU job) will briefly
yield that card. No model is loaded and no server is started, so this is safe
while the primary card is serving.

Run:  python3 tests/test_lease_dry_run.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time

from _rig import MS, ms as _ms, solo_slots

LEASE_FILE = str(_ms.LEASE_FILE)
CONTRACT = str(_ms.CONTRACT_FILE)
# Never the primary card: that is the primary inference slot.
SLOT = (list(solo_slots().values()) or [g for g in _ms.GPUS if g != _ms.PRIMARY_GPU] or [None])[-1]
HOLDER = "test-dry-run"

failures = []


def ms(*args):
    r = subprocess.run([sys.executable, MS, "lease", *args],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def check(label, ok, detail=""):
    print(f"  {'✓' if ok else '✗'} {label}{'' if ok else f': {detail}'}")
    if not ok:
        failures.append(label)


def fingerprint():
    """Hash of the two files a lease action is allowed to touch."""
    out = []
    for p in (LEASE_FILE, CONTRACT):
        try:
            with open(p, "rb") as f:
                out.append(hashlib.md5(f.read()).hexdigest())
        except FileNotFoundError:
            out.append("absent")
    return tuple(out)


def main():
    if SLOT is None:
        print("SKIP: needs a GPU other than the primary in the config")
        return 0
    if os.path.exists(LEASE_FILE):
        print(f"SKIP: a lease is already held ({LEASE_FILE} exists) — "
              "refusing to disturb it")
        return 0

    print("1. dry-run acquire writes nothing")
    before = fingerprint()
    rc, out = ms("acquire", "--holder", HOLDER, "--devices", SLOT,
                 "--lease-ttl", "60", "--dry-run")
    check("dry acquire exits 0", rc == 0, out.strip())
    check("no lease file created", not os.path.exists(LEASE_FILE))
    check("lease file + contract unchanged", fingerprint() == before)

    print("2. dry-run acquire --json emits the dry_run envelope")
    rc, out = ms("acquire", "--holder", HOLDER, "--devices", SLOT,
                 "--dry-run", "--quiet", "--json")
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        payload = {}
        check("--quiet --json emits parseable JSON", False, out.strip()[:200])
    else:
        check("--quiet --json emits parseable JSON", True)
        check("payload marked dry_run", payload.get("dry_run") is True, str(payload))
        check("payload names the action", payload.get("action") == "acquire", str(payload))
        check("payload reports would_evict", "would_evict" in payload, str(payload))
    check("still no lease file", not os.path.exists(LEASE_FILE))

    print("3. real acquire still works")
    rc, out = ms("acquire", "--holder", HOLDER, "--devices", SLOT,
                 "--lease-ttl", "120", "--quiet", "--json")
    check("real acquire exits 0", rc == 0, out.strip())
    check("lease file created", os.path.exists(LEASE_FILE), out.strip())
    if not os.path.exists(LEASE_FILE):
        print("\nFAILED: cannot continue without a lease")
        return 1
    held = fingerprint()

    print("4. dry-run renew/release leave the held lease alone")
    time.sleep(1.1)   # a real renew would bump last_renew past md5 equality
    rc, out = ms("renew", "--holder", HOLDER, "--dry-run")
    check("dry renew exits 0", rc == 0, out.strip())
    rc, out = ms("release", "--holder", HOLDER, "--dry-run")
    check("dry release exits 0", rc == 0, out.strip())
    check("lease still held", os.path.exists(LEASE_FILE))
    check("lease file + contract unchanged by dry renew/release",
          fingerprint() == held)

    print("5. a dry run still enforces the conflict check")
    rc, out = ms("acquire", "--holder", "someone-else", "--devices", SLOT,
                 "--dry-run")
    check("conflicting dry acquire is denied", rc != 0, out.strip())
    check("denial names the holder", HOLDER in out, out.strip())
    check("denied dry acquire changed nothing", fingerprint() == held)

    print("6. real renew and release still work")
    rc, out = ms("renew", "--holder", HOLDER, "--quiet", "--json")
    check("real renew exits 0", rc == 0, out.strip())
    check("real renew wrote the lease", fingerprint() != held, "file identical")
    rc, out = ms("release", "--holder", HOLDER, "--quiet")
    check("real release exits 0", rc == 0, out.strip())
    check("lease file removed", not os.path.exists(LEASE_FILE))

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        # never leave a test lease behind
        if os.path.exists(LEASE_FILE):
            try:
                if json.load(open(LEASE_FILE)).get("holder") == HOLDER:
                    subprocess.run([sys.executable, MS, "lease", "release",
                                    "--holder", HOLDER, "--quiet"],
                                   capture_output=True, timeout=60)
            except Exception:
                pass
