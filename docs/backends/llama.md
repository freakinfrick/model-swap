# Backend: `llama` (llama.cpp `llama-server`)

The default backend. Every profile without a `backend:` key is a llama profile
(`_detect_backend`). References below are function names in `model-swap`;
search for them rather than relying on line numbers.

| | |
|---|---|
| **What** | `llama-server` from llama.cpp (or a fork), GGUF weights, OpenAI-compatible API |
| **When** | Default for anything GGUF; the only backend a slot can run (`cmd_slot`) |
| **Where** | Primary card on `<host>:<primary.port>` (default `127.0.0.1:8012`); each configured slot on its own port |
| **Load** | `model-swap load <profile>` (primary) · `model-swap slot <key> ensure --profile <p>` (slot) |
| **Profile** | `<models_dir>/profiles/<name>.yaml`. Required: `model:`; everything else optional (`load_profile`) |

---

## 1. Dispatch and identity

**Backend selection.** `_detect_backend(profile)` is `profile.get("backend", "llama")`.
`build_command()` dispatches vllm / freetoken / ds4 first and falls through to
`_build_llama_cmd()`, so a typo such as `backend: llamacpp` silently builds a
llama command.

**Process recognition.** `PROC_PATTERNS` matches the command line by port
(`llama-server.*--port <primary port>`, plus vllm and ds4 patterns).
`find_server_process()` runs `pgrep -a -f` over them, so a `llama-server` started
by hand on the primary port is found and reported as `external: true`. Slots use
`slot_find_process()`, also by port; §4 explains why the slot state file decides
ownership.

`get_current_state()` takes the model from the cmdline
(`parse_model_from_cmdline`: `--model`, or `-m` only when the argument contains
`/` or ends in `.gguf`). The profile comes from `<models_dir>/.active.json` when
its pid or model matches; otherwise `match_profile_to_model()` scans profiles for
the same `model:` value, and no match means `external: true`. Two profiles that
share one GGUF are therefore indistinguishable once `.active.json` is gone.

**Health.** `check_health()` probes `/health`, then `/v1/models`.
`wait_for_health()` polls every 0.5 s and fast-fails with `process_died` via
`proc.poll()` on the launched process; `os.kill(pid, 0)` alone would succeed
against an unreaped zombie and wait out the full timeout.

---

## 2. Command construction

`_build_llama_cmd(profile)` starts with a fixed prefix:

```
<binary> --model <gguf> --host <host> --port <primary port> --device CUDA0 --metrics
```

`--metrics` is always present so the idle watchdog (`_get_predicted_total`,
metric `llamacpp:tokens_predicted_total`) can read `/metrics`. Then, in order:

1. `params:` value flags via `PARAM_MAP`:

   | profile key | flag | | profile key | flag |
   |---|---|---|---|---|
   | `ctx_size` | `--ctx-size` | | `n_cpu_moe` | `--n-cpu-moe` |
   | `n_gpu_layers` | `--n-gpu-layers` | | `flash_attn` | `--flash-attn` |
   | `threads` | `--threads` | | `spec_type` | `--spec-type` |
   | `threads_batch` | `--threads-batch` | | `spec_draft_model` | `--spec-draft-model` |
   | `batch_size` | `--batch-size` | | `spec_draft_n_max` | `--spec-draft-n-max` |
   | `ubatch_size` | `--ubatch-size` | | `spec_draft_p_min` | `--spec-draft-p-min` |
   | `cache_type_k` | `--cache-type-k` | | `tensor_split` | `--tensor-split` |
   | `cache_type_v` | `--cache-type-v` | | `split_mode` | `--split-mode` |
   | `cache_reuse` | `--cache-reuse` | | `n_parallel` | `--parallel` |

2. `params:` bare flags via `BOOL_FLAGS` (`mlock`, `jinja`, `metrics`), emitted
   when truthy and not already present.
3. `kv_persist: true` adds `--slot-save-path <models_dir>/.slot-kv` (see §5).
4. `extra_args:`, appended verbatim and **last**. llama.cpp argument parsing is
   last-wins, so `extra_args` overrides `params:`. Every flag not in `PARAM_MAP`
   goes here (`-ot`, `--no-mmap`, `--numa`, `--chat-template-file`, `-fit off`, ...).

Unknown `params:` keys are silently dropped; `model-swap validate` reports them.
`flash_attn` takes a value: write `flash_attn: "on"` quoted, because bare `on`
is YAML `True` and renders `--flash-attn True`.

**Pool loads** (`start_server` with a `pool_slot`) rewrite `--device CUDA0` to
`CUDA0,...,CUDA<n-1>` for the pool's card count (without it `--split-mode layer`
has one device and runs out of memory), then append the pool's `extra_args`,
skipping any `--flag` the profile already set together with its value.

**Environment.** `CUDA_DEVICE_ORDER=PCI_BUS_ID`; `CUDA_VISIBLE_DEVICES` is the
pool's UUID list, else the primary card's UUID from `gpus:`. Cards are pinned by
UUID, never by index. Profile `cuda_device_order:` and `env:` are applied after
pinning and can override either.

---

## 3. Profile schema (llama)

| key | type | default | notes |
|---|---|---|---|
| `model` | str | **required** | GGUF filename relative to `<models_dir>`, or an absolute path. |
| `binary` | str\|null | `paths.llama_server`, else `llama-server` on PATH | Absolute path, or a bare name resolved in the default binary's directory. Not found: `load` exits 4. |
| `params` | map | `{}` | `PARAM_MAP` + `BOOL_FLAGS` only. |
| `extra_args` | list | `[]` | Verbatim, appended last. |
| `vram_required_mb` | int | `0` | Static gate vs card/pool capacity, live gate vs free VRAM minus 300 MiB. `0` disables both. |
| `health_timeout` | int (s) | none (30 s effective) | Used when `--timeout` is left at its default. |
| `idle_stop` | bool | `true` | `false`: no primary watchdog; contract publishes `idle_timeout: null`. |
| `pool` | str | none | A key under `slots:`. Primary path only; serves on the primary port (§4). |
| `kv_persist` | bool | `false` | Save slot 0's KV cache on idle-stop, restore on next load. Use with `n_parallel: 1`. |
| `env` | map | `{}` | Extra process env, applied after device pinning. |
| `cuda_device_order` | str | `PCI_BUS_ID` | Overrides `CUDA_DEVICE_ORDER`. |
| `description` | str | `""` | Shown by `list` / `info`. |

```yaml
description: 8B instruct, Q4_K_M, 32k context
model: example-8b-instruct-Q4_K_M.gguf
vram_required_mb: 9000
params:
  ctx_size: 32768
  n_gpu_layers: 99
  flash_attn: "on"
  cache_type_k: q8_0
  cache_type_v: q8_0
  jinja: true
extra_args: ["--no-warmup"]
```

---

## 4. Slots and pools

Slots come from `slots:` in the config (`_slot_from_config`, collected in
`ALL_SLOTS`): a port, a `gpus:` list, optional `extra_args` and `label`. A slot
with several cards is a pool. Each slot has a log `/tmp/llama-<name>.log`, a
state file `<models_dir>/.active-<name>.json`, a last-ensure file
`<models_dir>/.last-<name>.json`, and an idle timeout `slot_idle_timeout_s`
(default 1800 s, per-server override `--idle-timeout`).

The `gpus:` order is the `CUDA_VISIBLE_DEVICES` order, so a `--tensor-split`
ratio in the slot's `extra_args` maps onto the cards as listed. Slots sharing a
card or port evict each other (`_slot_stop_conflicting`); a slot that includes
the primary card stops the primary server first. The `secondary` key also backs
the `secondary-*` commands and its `device_alias`.

**Two load paths, two ports.** The most common source of confusion:

- `load <profile>` with `pool: <key>`: `start_server` with the pool's cards
  visible, but **serves on the primary port**. Stops every slot first; state in
  `.active.json` and the contract's `primary`.
- `slot <key> ensure --profile <profile>`: `slot_ensure` / `slot_start`, serves on
  **the slot's port**; state in the slot's state file and `slots.<key>`.

The slot path reads only `model`, `binary`, `params` and `extra_args` from the
profile (`params.ctx_size` too, but an explicit `--ctx` wins). No `pool`,
`vram_required_mb`, `health_timeout`, `env` or `cuda_device_order`, and no VRAM
pre-flight. Its fixed prefix is `-ngl 99 --ctx-size N --flash-attn on
--cache-type-k q4_0 --cache-type-v q4_0 --jinja -np 1 --metrics -fit off`, then
the slot's `extra_args`, then profile `params`, then profile `extra_args` last.

A bare `slot <key> ensure` reloads the last explicit ensure (`.last-<name>.json`),
else the `slot_default_gguf` glob. `--dry-run slot <key> ensure` prints the plan only.

**Shared ports.** When two slots share a port, a healthy `/health` cannot prove
which owns it. The state file is the proof: `slot_ensure` treats "healthy and
state file present" as already running; `_slot_stop_conflicting` and
`_stop_slots_holding_primary` skip slots without one; `write_contract` reports a
slot running only if its state file exists.

`slot_ensure` with a different GGUF than the recorded one restarts the slot.
Its health timeout is fixed: 300 s for a multi-card slot, 120 s otherwise. On
timeout the process is **left running**, because killing it during a multi-GPU
CUDA load can wedge the driver and leave VRAM unreclaimable until reboot.

---

## 5. Lifecycle

`_cmd_load`, in order:

1. Resolve profile, binary, model path. If the GGUF is missing and
   `archive_remote` is set, `archived_entry()` looks it up in
   `<models_dir>/ARCHIVED.txt` and prints an `rclone copy` restore line.
2. `gpu_preflight()`: `cuInit`, primary card UUID present, temperature ≤ 85 °C.
3. Static VRAM gate; `--dry-run` prints the command and stops here.
4. Health timeout: profile `health_timeout` when `--timeout` is the default.
5. Load lock `<models_dir>/.load.lock` (TTL `timeout * 2 + 300`, cleared when the
   holder is dead or the TTL expired). While held for a non-pool load, the contract publishes
   `primary.state: loading`.
6. Evict slots holding the primary card (all slots for a pool load), refuse if a
   FreeToken borrow holds a target card, stop the current primary server.
7. Live VRAM gate (non-pool loads), after eviction so the outgoing server is not
   counted. If short, it waits up to 45 s for another GPU consumer to free memory.
8. Start and wait for health; two attempts, retrying only on `process_died`. On
   final failure it prints the tail of `/tmp/llama-server.log` and the last line
   matching a fatal pattern (`cudaMalloc failed`, `out of memory`, `CUDA error`, ...).
9. `save_state()` → `.active.json` → `write_contract()`; `_kv_restore()` for
   `kv_persist` profiles; idle watchdog unless `idle_stop: false`.

Exit codes: 2 pre-flight, 3 start/health failure or lease/borrow conflict,
4 profile/binary/model not found, 5 VRAM gate, 6 load lock held. A card leased
by another holder (`model-swap lease`) is refused on both paths.

**Contract.** `write_contract()` rewrites `paths.contract` atomically (temp file
+ `os.replace`); model-swap is its only writer. It publishes `primary`, one block
per slot, and a `models:` map keyed by served model id and by basename without
`.gguf`, for a gateway to route on.

**Idle watchdogs.**

- Primary: `_restart_idle_watchdog()` forks `_primary_idle_loop` (also run by
  `model-swap idle-stop`). Timeout `primary.idle_timeout_s` (default 1800 s). It
  watches `llamacpp:tokens_predicted_total`, exits if the server pid changes and
  defers while a lease covers the card. On stop it saves KV (`kv_persist`), kills
  the server and rewrites the contract; `.active.json` is kept so the next
  `ensure` reloads the same profile.
- Slot: `_start_slot_watchdog()` polls every 10 s. New tokens on the slot's
  `/metrics` and `slot_touch_activity()` both count as activity; a live lease or
  a `/slots` entry with `is_processing` postpones the stop.

**`ensure`** (`cmd_ensure`) means "something is serving". It returns without
loading when the primary is healthy, a load lock is held, any slot is healthy, or
a FreeToken borrow answers a generation probe. Otherwise it loads `--profile`,
else the profile in `.active.json`, else `primary.default_profile` (none: exit 4).
`stop` refuses while another process holds the load lock; `restart` reloads the
profile `get_current_state()` names.

---

## 6. Gotchas

- **Silent failures.** Unknown `params:` keys and `backend:` typos are not errors.
  Run `model-swap validate` after editing a profile. It checks that `binary`
  exists, not what it supports: a fork-only KV or spec type on the stock binary
  crashes at startup.
- **Too short a health window strands the server.** The load reports failure while
  the server is still starting; `save_state` never runs, so the process keeps
  running unregistered and the contract names the previous profile. Raise
  `health_timeout` for slow cold loads. An explicit `--timeout 30` equals the
  default, so `health_timeout` still applies.
- **Slow generation needs `idle_stop: false`.** At low tokens/s the token-delta
  watchdog can fire during a long request.
- **`-fit off` is set on the slot path only.** llama.cpp's automatic memory fit is
  redundant when `-ngl` / `--n-cpu-moe` / `--tensor-split` are pinned and can spin
  for minutes on a pool. On the primary path, add it to `extra_args`.
- **Tool calling depends on the chat template.** `--jinja` uses the GGUF's
  embedded template; forcing `--chat-template chatml` disables tool-call parsing.
  Replace a broken embedded template with `--chat-template-file` in `extra_args`.
- **Large MoE on CPU.** `n_cpu_moe: N` moves expert tensors of N layers to system
  RAM; `--no-mmap` and `--numa distribute` often matter more than the exact N.
- **Compute buffers grow with context depth.** A configuration that fits with a
  short prompt can run out of memory on a long one.
- **`info` previews `build_command()` only**, without pool rewrites. Use
  `--dry-run load` for the command that would run.

---

## Quick reference

```bash
model-swap list                       # profiles; [MISSING] / [ARCHIVED] weights
model-swap info <profile>             # params → flags, env, command preview
model-swap validate [<profile>]       # unknown params:, missing binary/GGUF, VRAM
model-swap --dry-run load <profile>   # print the exact command, change nothing
model-swap slot <key> ensure --profile <profile> [--ctx N] [--idle-timeout S]
model-swap status | current | health | gpu | ps | logs -n 50
```

Read-only: `list`, `info`, `validate`, `status`, `current`, `health`, `gpu`, `ps`,
`logs`, and anything with `--dry-run`. Everything else changes GPU state and the
contract.
