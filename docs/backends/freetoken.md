# Backend: `freetoken`

FlashML **FreeToken** (Apache-2.0, PyPI `freetoken`) is a MoE-offload serving engine:
experts live in host RAM, an LRU expert-slot cache sits on the GPU, and cache misses stream
over PCIe. In model-swap it is **always a borrow**, never a standing model.

| | |
|---|---|
| **Dispatch** | `backend: freetoken` in a profile → `_detect_backend()` → `cmd_load` → `_cmd_load_freetoken()` |
| **Process** | `ft serve` from `paths.freetoken` (usually its own venv), on its own pinned port (`freetoken.port`, default **1919**), log `/tmp/freetoken-server.log` |
| **Lifecycle** | Borrow only: reserve the card → serve → readiness = a real one-token completion → publish contract. `stop` = SIGTERM → wait for VRAM drain → release restores the card's previous profile |
| **State** | `<models_dir>/.freetoken.json` (`FT_STATE_FILE`); reservation `<models_dir>/.reservation.json` with reason `freetoken:<profile>` |

Search the script by function name; no line numbers are given.

## 1. Dispatch and identity

`_detect_backend(profile)` returns `profile["backend"]`. FreeToken is the only backend with
its **own pinned port**, hence its own state file, log and contract handling.

| Name | Value / purpose |
|---|---|
| `FT_BIN_DEFAULT` | `MODELSWAP_FT_BIN` env, else config `paths.freetoken`, else `ft` on `PATH` |
| `FT_PORT` | config `freetoken.port`, default `1919`. No slot may use it |
| `FT_ENV` | config `freetoken.env` — machine-wide env applied to every `ft serve` |
| `FT_READY_TIMEOUT` | `300` s (first run = weight load + kernel JIT + graph capture) |
| `_ft_devices()` | `device → (CUDA_VISIBLE_DEVICES, reserve vocabulary, slot to evict)`, derived from `gpus:` and `slots:` |
| `FT_PARAM_MAP` | `ft_params` key → `ft serve` flag |

`_ft_devices()` makes every configured GPU borrowable. The primary card reserves under its
own name; any other card reserves the secondary alias and evicts the single-card slot on it.

**Liveness / identity detection:**

- `_ft_state()` — reads `FT_STATE_FILE`, checks the pid with `os.kill(pid, 0)`; a dead pid
  **deletes** the stale file and returns `None`. Not a read-only probe.
- `_read_reservation()` — read-only; `cmd_stop` uses it to find a reservation left by a
  dead engine.
- `get_current_state()` — with nothing on the primary port, a live primary-card borrow is
  reported as the primary server (`backend: "freetoken"`); else a `pgrep` marker
  (`<ft bin dir>/python.*ft serve.*--port <FT_PORT>`) reports a manual engine as
  `external: True`. Non-primary borrows are not reported here.

## 2. Binary and model resolution

`resolve_binary()`: `profile["ft_bin"]` if set, else `FT_BIN_DEFAULT`; a missing path makes
`cmd_load` exit 4.

`resolve_model_path()`, in order:

1. absolute path → used if it exists, else `None`;
2. `<models_dir>/<model>` if that exists (`model: hf/Org-Model` resolves to
   `<models_dir>/hf/Org-Model`);
3. a string containing `/` is treated as an **HF repo ID** and passed through for `ft` to
   fetch, same contract as the vLLM branch;
4. otherwise `None`.

So a missing local `hf/Org-Model` becomes a repo ID: the error is a Hugging Face resolution
failure, not "file not found". Check the weights are on disk first.

## 3. `_build_ft_cmd` and `FT_PARAM_MAP`

`build_command()` dispatches to `_build_ft_cmd()`:

```
<ft_bin> serve
  --model              <resolved model path or repo id>
  --served-model-name  <served_name | basename(model) without .gguf>
  --port               <ft_port | FT_PORT>
  [ft_params → flags]
  [extra_args…]
```

Only `FT_PARAM_MAP` keys are translated; other `ft_params` keys are silently ignored, so
other flags go through `extra_args`:

| `ft_params` key | `ft serve` flag |
|---|---|
| `moe_backend` | `--moe-backend` |
| `memory_ratio` | `--memory-ratio` |
| `max_seq_len` | `--max-seq-len-override` |
| `moe_cpu_threads` | `--moe-cpu-threads` |
| `tool_call_parser` | `--tool-call-parser` |
| `reasoning_parser` | `--reasoning-parser` |

Launch uses the shared `start_server()`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES` from `_ft_devices()`, then the profile's `env:` **after** pinning (so
it can override anything). It exits 3 if a live GPU lease holds the card.

## 4. Profile schema

Profiles live in `<models_dir>/profiles/*.yaml`. Fields the freetoken path reads:

| Field | Required | Meaning |
|---|---|---|
| `backend` | yes | must be `freetoken` |
| `model` | yes | `<models_dir>`-relative dir, absolute dir, or HF repo id |
| `device` | yes | a card name from `gpus:`. Normalized with `str()`, so a numeric name parsed by YAML as an int still matches |
| `served_name` | no | model id published to the contract; defaults to the model basename |
| `ft_port` | no | defaults to `FT_PORT` |
| `ft_bin` | no | alternate `ft` binary |
| `reserve_timeout` | no | reservation TTL in seconds, default 3600 |
| `ft_params` | no | mapped per `FT_PARAM_MAP` |
| `extra_args` | no | appended verbatim to the command |
| `env` | no | extra process env; merged after device pinning |

Example (`<models_dir>/profiles/ft-example-moe.yaml`):

```yaml
model: hf/Org-Example-MoE
description: "FreeToken borrow — example MoE model with experts offloaded to host RAM"
backend: freetoken
device: gpu0                  # a key under gpus: in the config
served_name: Example-MoE
reserve_timeout: 14400        # 4 h borrow window
ft_params:
  moe_backend: auto
  memory_ratio: 0.9
```

## 5. Borrow / return lifecycle

### 5.1 Load — `_cmd_load_freetoken()`

1. **Checks** — a live `_ft_state()` → rc 3; `device` (`str()`-normalized) not in
   `_ft_devices()` → rc 4.
2. **Env** — `TVM_FFI_CUDA_ARCH_LIST = _gpu_arch(device)` (nvidia-smi compute capability, or
   the card's `arch:`), then each `freetoken.env` entry; both via `setdefault`.
3. **Timeout** — `--timeout` if changed from the default 30, else `FT_READY_TIMEOUT`.
4. **`--dry-run`** — prints the command and stops; no state written.
5. **Load lock** — `_acquire_load_lock(profile, ttl_s=timeout + 300)`; competing load → rc 6.
6. **Evict the card's slot first**, so it joins the snapshot via `extra_stopped` and is
   restored on release.
7. **`do_reserve(res_devices, "freetoken:<profile>", reserve_timeout, extra_stopped)`** —
   stops inference on the card, snapshots it, writes reservation + contract.
8. **`start_server(...)`**, then readiness via `_ft_wait_ready()` (§6).
9. **Failure**, in this order: last 20 log lines → `_ft_stop(pid)` → `_ft_wait_vram_free()`
   → `_release_load_lock()` → `do_release(no_restore=False)` → rc 3. The lock must drop
   first: `do_release`'s primary restore takes it, and the reverse order prints "Load in
   progress — skipping … restore" and leaves the card empty.
10. **Success** — `_ft_save_state(...)` → `write_contract()` → `_release_load_lock()`.

### 5.2 Stop — `cmd_stop()` freetoken branches

**Live engine:** SIGTERM via `_ft_stop()` (SIGKILL after 60 s) → `_ft_wait_vram_free()`
(reports an error if the card still holds VRAM after 90 s) → delete `FT_STATE_FILE` →
`write_contract()` → `do_release(no_restore=False)`, which restores the snapshotted profile.

**Engine already crashed:** `_ft_state()` is `None` because the pid is dead, but the
reservation still holds the card. If `_read_reservation()` shows a `freetoken:` reason,
`stop` prints *"FreeToken reservation without a live process (engine crashed) — returning
the card"*, clears the state file and releases with restore.

**Anti-stacking:** `cmd_load`'s normal path refuses (rc 3) to start a server on a card
FreeToken is borrowing.

## 6. Readiness: a completion, never `/health`

`_ft_serving(port, model, timeout_s=5)` POSTs a one-token `/v1/chat/completions`. HTTP 200,
or an `HTTPError` below 500 (engine answered, rejected the probe), counts as alive; 5xx and
connection errors do not.

`_ft_wait_ready(pid, port, model, timeout_s)` polls it once a second, short-circuits if the
process dies, and returns `(ok, elapsed, reason)`, `reason` in
`healthy | process_died | timeout` (same shape as `wait_for_health`).

Why: `GET /health` is served by the HTTP frontend, which binds seconds in while the
scheduler is still loading weights. An engine that crashes after that (dtype error, CUDA
OOM, missing weight key) leaves `/health` green while completions return 503. Never
reintroduce `wait_for_health` on this path.

## 7. Contract and `ensure`

**`ensure`:** a live borrow **is** the ready LLM. `cmd_ensure` runs `_ft_serving()` on the
recorded port and served name; on success it returns 0 (JSON status `freetoken_serving`),
so launchers that call `ensure` keep working through a borrow. If the probe fails it falls
through and loads a real primary.

**`write_contract()`:**

- **`primary` override for a primary-card borrow:** `primary` gains `port=<ft port>,
  backend="freetoken", profile, model=served_name, pid, idle_timeout=None,
  borrowed_by="freetoken"`. `state` stays `"running"`, so consumers gating on it keep
  working, and one that falls back to the primary URL reaches the live engine instead of a
  dead port. **A borrow of any other card leaves `primary` untouched.**
- **`models` map entries:** `models[served_name] = http://127.0.0.1:<ft_port>`, plus
  `models[profile_name]` when it differs. While `primary.backend == "freetoken"` the primary
  llama entry is omitted. A proxy routing by this map needs no FreeToken-specific code.

A borrow never lands in a restore snapshot (`do_reserve` refuses while a reservation is
live), so `do_release`'s restore-timeout map has no `freetoken` entry.

## 8. Performance

Throughput depends on the card, PCIe and host RAM bandwidth, and how much of the model fits
in VRAM; measure on your own hardware. A model that fits entirely in VRAM never misses the
expert cache, so its numbers say nothing about offload. Decode speed tracks the
active-parameter fraction more than total size.

## 9. Gotchas

1. **The JIT arch must be pinned per device.** FreeToken's kernel JIT asks tvm-ffi for a
   target; tvm-ffi runs `nvidia-smi --query-gpu=compute_cap` and takes **row 0**, and
   nvidia-smi ignores `CUDA_VISIBLE_DEVICES`. On a mixed-GPU machine row 0 can be a different
   card (`nvcc fatal : Unsupported gpu architecture 'compute_120'`). model-swap sets
   `TVM_FFI_CUDA_ARCH_LIST` from `_gpu_arch()`; set the card's `arch:` if detection is wrong.
2. **Machine-wide env goes in `freetoken.env`.** E.g. when system nvcc and torch's CUDA build
   differ, FreeToken refuses to JIT without `FREETOKEN_ALLOW_CUDA_MISMATCH=1`. Per-profile
   copies get forgotten in the next new profile.
3. **Newer architectures need a newer toolkit.** sm_120 requires nvcc ≥ 12.8; no env
   override works around an older one.
4. **VRAM is reclaimed asynchronously; `_ft_wait_vram_free()` is load-bearing.** A dead
   `ft` pid does not mean free memory; restoring at once can OOM the restored server. The
   poll is by UUID, 90 s deadline, 1024 MiB floor, and never blocks if nvidia-smi fails.
5. **Check the checkpoint format before downloading.** One repo name can exist in
   incompatible quantized exports; fetch `model.safetensors.index.json` (no weights) first.
6. **Advertised context is not usable context.** `memory_ratio` budgets weights, MoE cache
   and KV cache together; a large MoE cache can leave little room for KV.
7. **Nothing supervises a healthy borrow.** An engine that dies later is noticed only when
   `ensure` or `stop` runs; `stop` recovers the card (§5.2).
8. **`primary.port` changes during a primary-card borrow.** Anything hardcoding the primary
   port fails while the borrow is live.
9. **Do not run an agent on the card it is reconfiguring.** A session served from the
   borrowed card is stopped by its own `load`.
10. **An orphan can keep holding VRAM.** If `stop` reports VRAM in use after 90 s, find it
    with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, kill it, and wait
    for the drain before the next load.

## 10. Tests

`tests/test_freetoken_lifecycle.py` is a standalone script, not pytest-collectable:

```bash
python3 tests/test_freetoken_lifecycle.py
```

It drives the real model-swap against your config and contract with the fake `ft` in
`tests/ft-shim/ft`, borrows the first single-card slot's card (skips if none), and asserts
the primary is unchanged. Preconditions: single-card slots idle, `FT_PORT` free, no
FreeToken state file, no live reservation. It covers dry run, load, `ensure`, contract
honesty, stop, crash recovery, a dead engine behind a green `/health`, and removes its
`zz-test-freetoken-shim` profile afterwards.

The shim serves `/health`, `/v1/models` and `/v1/chat/completions` on `--port` and exits on
SIGTERM. `FT_SHIM_DEAD_ENGINE=1` returns 200 on `/health` and 503 on completions, so the
failed-load path runs without a GPU.

Do not add a "second stop is a no-op" check: once the borrow is gone, a bare `stop` stops
whatever the primary card is serving.

## 11. Operating cheatsheet

```bash
model-swap backends freetoken                  # devices, port, param flags, lifecycle summary
model-swap load <profile> --dry-run            # print the exact ft serve command, no state change
model-swap load <profile> --timeout 900        # borrow (cold start can exceed 300 s)
model-swap status                              # 'freetoken' block: pid/port/model/device
curl -s localhost:<ft_port>/v1/models
model-swap stop                                # SIGTERM → VRAM drain → release → restore
```

Log: `/tmp/freetoken-server.log`.
