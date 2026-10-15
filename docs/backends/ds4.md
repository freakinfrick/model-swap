# Backend: `ds4` (`ds4-server`, SSD-streaming MoE)

| | |
|---|---|
| **Selector** | `backend: ds4` in the profile YAML (`_detect_backend`) |
| **Server** | `ds4-server`: config `paths.ds4_server`, else the binary on PATH (`DS4_BIN_DEFAULT`) |
| **Endpoint** | `HOST:PORT` of the primary server (default `127.0.0.1:8012`), same OpenAI-compatible API as llama and vLLM |
| **Model** | GGUF, resolved like a llama model (filename under `<models_dir>`, or a path) |
| **Health window** | `DS4_DEFAULT_TIMEOUT = 600` s; profile `health_timeout` overrides |
| **Idle watchdog** | none: `ds4-server` exposes no `/metrics` |

`ds4-server` streams MoE expert weights from SSD into host RAM and VRAM caches,
for models whose weights do not fit in VRAM plus host RAM offload. Reach for it
only when SSD-level streaming beats llama.cpp CPU expert offload; otherwise use
the `llama` backend.

**Do not confuse with a llama profile for a DeepSeek-V4 model.** A profile
that serves a DeepSeek-V4 GGUF through a llama.cpp fork (no `backend:` key,
`binary:` pointing at the fork's `llama-server`) is a llama profile. Its
`params:` mean nothing here: the ds4 path never reads `params:`, and
`extra_args` go straight to `ds4-server`, which exits on any option it does
not know.

## 1. Dispatch and identity

- **Backend selection.** `_detect_backend(profile)` returns
  `profile.get("backend", "llama")`. `build_command()` dispatches vllm →
  freetoken → ds4 (`_build_ds4_cmd`) → llama fallback.
- **Process pattern.** `PROC_PATTERNS["ds4"] = "ds4-server.*--port {PORT}"`.
  `find_server_process()` tries every pattern when no backend is given, so a
  running `ds4-server` is found by `status`, `stop` and `ensure` with no
  ds4-specific code. This entry is also the mutual exclusion: without it a
  ds4 server would read as "not running" and the next load would double-book
  the card.
- **Cmdline sniff.** `get_current_state()` sets `backend = "ds4"` when the live
  cmdline contains `ds4-server` (covers a server started by hand). The state
  file `<models_dir>/.active.json` wins when its PID or model matches.
- **Labels.** `_KNOWN_SERVICE_PATTERNS` maps `ds4-server` → `ds4` for the
  `ps`/`status` VRAM attribution; `load` and `stop` print "ds4-server".

## 2. Binary and model resolution

`resolve_binary()`, ds4 branch:

```python
b = Path(profile["binary"]) if profile.get("binary") else DS4_BIN_DEFAULT
return b if b.exists() else None
```

Unlike the llama branch there is no bin-dir name lookup and no fallback: a
profile `binary:` is taken as a path as written (`~` is not expanded, so use
an absolute path). A bad path makes `load` fail with `Binary not found:`.

`resolve_model_path()` has no ds4 branch; ds4 uses the GGUF path: `model:` is
a filename under `<models_dir>`, else an existing path, else `None`. On
`None`, `load` and `validate` check the archive manifest
`<models_dir>/ARCHIVED.txt` (only when `archive_remote:` is set in the config)
and print a restore hint; otherwise the model is reported missing.

## 3. Command construction

`_build_ds4_cmd(profile)`:

```python
cmd = [binary, "--model", model_path, "--host", HOST, "--port", str(PORT), "--cuda"]
# then DS4_PARAM_MAP value flags, DS4_BOOL_FLAGS presence flags, extra_args
```

The model, host, port and `--cuda` are fixed and cannot be changed through
`ds4_params`. Binding the primary port is what lets `stop`, `status`,
`health` and the contract work unchanged.

`DS4_PARAM_MAP`: value flags, emitted as `<flag> <value>` when the key is set.

| key | flag |
|---|---|
| `ctx` | `--ctx` |
| `tokens` | `--tokens` |
| `threads` | `--threads` |
| `ssd_streaming_cache_experts` | `--ssd-streaming-cache-experts` (a count or `<n>GB`) |
| `ssd_streaming_preload_experts` | `--ssd-streaming-preload-experts` |
| `ssd_streaming_full_layers` | `--ssd-streaming-full-layers` |
| `prefill_chunk` | `--prefill-chunk` |
| `power` | `--power` |
| `kv_disk_dir` | `--kv-disk-dir` |
| `kv_disk_space_mb` | `--kv-disk-space-mb` |
| `mtp` | `--mtp` (path to an MTP draft GGUF) |
| `mtp_draft` | `--mtp-draft` |
| `mtp_margin` | `--mtp-margin` |

`DS4_BOOL_FLAGS`: emitted only when truthy: `ssd_streaming`
(`--ssd-streaming`), `ssd_streaming_cold`, `warm_weights`, `quality`, `cors`.
Any other server flag goes in `extra_args`.

**Environment** (`start_server()`). The server inherits the parent env with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` pinned to the primary
card's UUID (CUDA accepts UUIDs; the vLLM UUID→index rewrite does not apply).
The profile's `env:` is applied last, so it can override anything; this is how
`DS4_CUDA_*` tuning variables reach the engine.

**KV checkpoint directory.** `start_server()`, not `_build_ds4_cmd`, creates
`ds4_params.kv_disk_dir`. `load --dry-run` returns before that point, so a
preview has no side effects. Keep `build_command()` pure: filesystem or GPU
side effects for any backend belong in `start_server()`.

**Pools and slots.** `start_server()` strips `--split-mode` / `--tensor-split`
from a pool's `extra_args` and skips the `--device` widening for every
non-llama backend. A `pool:` key only changes `CUDA_VISIBLE_DEVICES`; the server
still binds the primary port. `slot_start()` always builds a `llama-server`
command, so ds4 cannot run as a slot.

## 4. Profile schema

| key | type | default | effect |
|---|---|---|---|
| `backend` | str | `llama` | must be `ds4` |
| `model` | str | required | GGUF filename under `<models_dir>`, or a path |
| `binary` | str | `DS4_BIN_DEFAULT` | path to `ds4-server`; no name lookup |
| `description` | str | `""` | shown by `list` / `info` |
| `vram_required_mb` | int | `0` | gate vs card capacity and live free VRAM (300 MiB margin); `0` disables |
| `idle_stop` | bool | `true` | set `false` explicitly, or `validate` reports an issue |
| `health_timeout` | int (s) | unset | checked before `DS4_DEFAULT_TIMEOUT` |
| `ds4_params` | map | `{}` | §3 tables; unknown keys are dropped (`validate` warns) |
| `env` | map | `{}` | e.g. `DS4_CUDA_*`; applied after device pinning |
| `extra_args` | list | `[]` | appended verbatim; an unknown flag makes the server exit |
| `cuda_device_order` | str | `PCI_BUS_ID` | overrides `CUDA_DEVICE_ORDER` |
| `pool` | str | unset | a slot key; changes `CUDA_VISIBLE_DEVICES` only |
| `params` | map | | llama-only; ignored here, `validate` flags each key |

`model-swap create <name> <model.gguf> --backend ds4` writes a starter profile.
Its values (below) are template defaults; size the caches for your own RAM,
VRAM and disk:

```yaml
backend: ds4
model: Example-MoE-IQ2_XXS.gguf     # under <models_dir>
description: "Example large MoE, SSD expert streaming"
vram_required_mb: 22000
idle_stop: false                    # no /metrics, the watchdog cannot fire
ds4_params:
  ctx: 8192
  ssd_streaming: true
  ssd_streaming_cache_experts: 6GB  # VRAM expert cache: count or <n>GB
  warm_weights: true                # page-touch resident weights at engine open
env:
  DS4_CUDA_HOST_EXPERT_CACHE_GB: 72
  DS4_CUDA_PARALLEL_FETCH_THREADS: 16
  DS4_CUDA_ARENA_VRAM_RESERVE_GB: 3
  DS4_CUDA_NO_PINNED_ARENA_FALLBACK: 1
```

Check before loading: `model-swap validate <name>` (model resolves, binary
exists, every `ds4_params` key is known, no `params:` block, `idle_stop` is
`false`), then `model-swap load <name> --dry-run` to see the exact command.
`model-swap info <name>` shows the `ds4_params` → flag mapping.

## 5. Lifecycle quirks

**Health is reachability.** `ds4-server` has no `/health` (it 404s) but
serves `/v1/models`. `check_health()` tries `/health`, moves on to
`/v1/models` on an HTTP error, and returns "unreachable" at once on connection
refused. The server opens its socket only after the engine (weights, streaming
plan, preload) is up, so a 200 means the model is resident.

**Long timeout, fast failure.** `DS4_DEFAULT_TIMEOUT = 600` covers engine open
(mmap, pinning resident tensors, `warm_weights`, preload). `wait_for_health()`
polls `proc.poll()` every 0.5 s, so a crash (e.g. CUDA OOM) is reported in
seconds; the full 600 s is only spent on a slow load. Knock-on effects:

- `load` retries once, and only after a crash (`process_died`); a timeout
  fails at once. A server that hangs without crashing therefore fails after
  600 s, the worst case (late crash, then a hang) after about 2 × 600 s.
- The load lock TTL is `timeout * 2 + 300` (1500 s by default).
- `ensure` will not "recover" a server while a load lock is held; a ds4 engine
  open looks unhealthy for minutes, and loading over it would evict it.
- `release` restores a stopped primary profile with the backend default
  timeout, not the profile's `health_timeout`.

**No idle watchdog.** `_get_predicted_total()` returns `None` for ds4, and
`load` kills the watchdog instead of restarting it ("ds4 has no /metrics").
The check is `idle_stop and backend != "ds4"`, so `idle_stop: true` would not
enable it either; `validate` asks for `false` so the profile says what
happens. Consequences:

1. A ds4 server never frees the card by itself. It holds VRAM and host expert
   cache until `model-swap stop` or the next `load`.
2. `write_contract()` publishes `idle_timeout: null` for a profile with
   `idle_stop: false`, so contract readers do not expect a reaper.
3. `stop` sends SIGTERM, then SIGKILL after 10 s (`kill_server`). At low token
   rates a generation can still be running when SIGKILL lands, and killing
   mid-CUDA-op can leave VRAM held.

## 6. Gotchas

- An unknown flag in `extra_args` is fatal at the server, not ignored.
  Preview with `--dry-run` and run the printed command by hand once before
  trusting a new profile.
- The `<n>GB` form of `ssd_streaming_cache_experts` is passed through as
  written; confirm your `ds4-server` build accepts it.
- `binary: ~/...` does not work; `~` is not expanded on this path.

## Cross-references

- `model-swap backends ds4`: the same facts, generated from the script.
- Sibling backend docs: `docs/backends/llama.md`, `docs/backends/vllm.md`,
  `docs/backends/freetoken.md`.
