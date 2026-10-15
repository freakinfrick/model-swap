# Backend: `vllm`

| | |
|---|---|
| **Selector** | `backend: vllm` in the profile YAML (`_detect_backend`) |
| **Server** | `VLLM_PYTHON -m vllm.entrypoints.openai.api_server` |
| **Interpreter** | env `MODELSWAP_VLLM_PYTHON`, else config `paths.vllm_python`, else `python3` on PATH |
| **Endpoint** | `HOST:PORT` of the primary server (default `127.0.0.1:8012`), same API as llama; pooled loads too |
| **Model** | HF repo ID or absolute safetensors dir (`<models_dir>/hf/<name>`), never GGUF |
| **Health window** | `VLLM_DEFAULT_TIMEOUT = 120` s; profile `health_timeout` overrides |

## 1. When to pick it

vLLM is the continuous-batching, paged-attention server. It is the backend for
safetensors weights and non-GGUF quantization (GPTQ, AWQ, FP8,
compressed-tensors, NVFP4 W4A4 on Blackwell cards). Pick it when the weights
exist only as safetensors, or when you want real concurrency
(`max_num_seqs` > 1).

Pick `llama` instead for GGUF, CPU expert offload, or a single tight-fit load:
vLLM preallocates KV cache up to `gpu_memory_utilization` and is less forgiving
of a close fit.

## 2. Dispatch and identity

- `_detect_backend(profile)` returns `profile.get("backend", "llama")`;
  `build_command()` sends `vllm` to `_build_vllm_cmd`.
- `PROC_PATTERNS["vllm"] = "vllm.entrypoints.*--port {PORT}"`, used by
  `find_server_process()`, which tries every backend's pattern when none is given.
- `get_current_state()` sets `backend = "vllm"` when the live cmdline contains
  `vllm.entrypoints`; the state file `<models_dir>/.active.json` wins when its
  PID or model matches.
- `parse_model_from_cmdline()` prefers `--model` and only falls back to `-m`
  when its argument looks like a path, because `python -m vllm...` would
  otherwise read as the model.
- `_KNOWN_SERVICE_PATTERNS` labels `vllm` processes for `ps`/`status`;
  `load` and `stop` print "vLLM".

## 3. Interpreter and model resolution

`resolve_binary()` returns `VLLM_PYTHON` for every vllm profile. There is no
server binary; a `binary:` key is ignored. Install vLLM in its own venv
(it pins its own torch/CUDA build, which tends to conflict with other
packages) and point `paths.vllm_python` or `MODELSWAP_VLLM_PYTHON` at that
venv's `python`. The env var wins, so you can switch venvs per load.

`resolve_model_path()`, vllm branch: an absolute path that exists is returned
as a local dir; anything else is returned as-is and treated as an HF repo ID
for vLLM to fetch. So a vllm profile never fails the "model missing" check: a
mistyped repo ID only fails when vLLM tries to download it. `validate`
catches only a missing absolute dir.

**Download.** `model-swap download <org/repo> --backend vllm`:

- errors out if the repo has no safetensors;
- filters files through `_HF_IGNORE = ["*.gguf", "original/*", "*.pt", "*.bin"]`;
- writes to `<models_dir>/hf/<repo-basename>/`, biggest file first, via
  `_hf_curl_fetch` (curl with `-C -` range resume, so an interrupted download
  continues instead of restarting);
- skips GGUF conversion and prints the matching `create` command.

For vllm the safetensors dir *is* the served model; deleting it deletes the
model.

## 4. `CUDA_VISIBLE_DEVICES`: UUIDs in, PCI indices out

`start_server()` pins every backend by GPU UUID: `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and `CUDA_VISIBLE_DEVICES=<uuid[,uuid...]>` (the primary card, or a pool's
cards). UUIDs are stable across reseating; CUDA indices are not.

vLLM rejects UUIDs in `CUDA_VISIBLE_DEVICES`, so for vllm only,
`start_server()` rewrites the list just before `Popen` with `_uuid_to_pci()`:
each UUID maps to the card's `pci_index:` from the config, else the index
`nvidia-smi` reports (PCI bus order under `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
Everything upstream (lease checks, pool selection, the contract) still uses
UUIDs; the index exists only in the child env. Keep profiles and configs on
UUIDs and leave the conversion in place: it is the seam between the two.

## 5. Profile schema

| key | type | default | effect |
|---|---|---|---|
| `backend` | str | `llama` | must be `vllm` |
| `model` | str | required | HF repo ID or absolute local dir |
| `description` | str | `""` | shown by `list` / `info` |
| `vram_required_mb` | int | `0` | gate vs card/pool capacity, and vs live free VRAM (300 MiB margin) on non-pool loads |
| `pool` | str | unset | a slot key from the config; changes `CUDA_VISIBLE_DEVICES` only |
| `vllm_params` | map | `{}` | table below |
| `extra_args` | list | `[]` | appended verbatim, e.g. `--reasoning-parser <name>`, `--enforce-eager` |
| `env` | map | `{}` | exported after device pinning, so it can override anything |
| `cuda_device_order` | str | `PCI_BUS_ID` | overrides `CUDA_DEVICE_ORDER` |
| `health_timeout` | int (s) | unset | checked before `VLLM_DEFAULT_TIMEOUT` |
| `idle_stop` | bool | `true` | `false` disables the idle watchdog and publishes `idle_timeout: null` |
| `params` | map | | llama-only; ignored, `validate` reports each key |

`vllm_params` → flags, from `_build_vllm_cmd`:

| key | flag | default |
|---|---|---|
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `0.95` |
| `max_model_len` | `--max-model-len` | `131072` |
| `tensor_parallel_size` | `--tensor-parallel-size` | unset |
| `quantization` | `--quantization` | unset |
| `kv_cache_dtype` | `--kv-cache-dtype` | unset |
| `enable_chunked_prefill` | `--enable-chunked-prefill` (bool) | unset |
| `max_num_seqs` | `--max-num-seqs` | unset |
| `cpu_offload_gb` | `--cpu-offload-gb` | unset |
| `tool_call_parser` | `--tool-call-parser` | `hermes`, always emitted |

Unknown `vllm_params` keys are silently dropped; `validate` has no key check
for vllm. If a setting has no effect, check the table, then use `extra_args`.

Two things are always on the command line:

- **`--served-model-name` with up to three aliases**: profile name, the full
  `model` string, and its basename, deduped. vLLM checks the request's
  `model` field and returns 404 on a mismatch (llama.cpp ignores it), so
  clients addressing the server by profile name or by basename both work.
- **`--enable-auto-tool-choice --tool-call-parser <parser>`**: without them
  vLLM rejects every request with `tool_choice="auto"` (HTTP 400). `hermes`
  fits Qwen-style tool calls; set `vllm_params.tool_call_parser` for others.

`model-swap create <name> <repo-or-dir> --backend vllm` writes a starter:

```yaml
backend: vllm
model: org/Example-Model-AWQ      # or /abs/path/to/<models_dir>/hf/Example-Model-AWQ
description: ""
vram_required_mb: 24000
vllm_params:
  gpu_memory_utilization: 0.95
  max_model_len: 131072           # from --ctx-size when given
  tensor_parallel_size: 1
  enable_chunked_prefill: true
  max_num_seqs: 4
```

## 6. Lifecycle

`model-swap load <profile>` runs the same `cmd_load` path as llama:

1. GPU pre-flight, pool resolution, static VRAM-vs-capacity gate.
2. Health timeout: explicit `--timeout` → profile `health_timeout` →
   `VLLM_DEFAULT_TIMEOUT`. An explicit `--timeout 30` cannot be told apart
   from the default and is also raised to 120 s.
3. Load lock (TTL `timeout * 2 + 300`), evict slots holding the primary card,
   refuse if a FreeToken borrow holds a target card, stop the incumbent server.
4. Live free-VRAM gate (non-pool loads only).
5. `start_server()` → `wait_for_health()`, with one retry after a crash. On
   failure the tail of the server log (`/tmp/llama-server.log`, shared by all
   backends) is printed with the likely CUDA error.
6. `save_state(backend="vllm")` → `write_contract()` → idle watchdog.

vLLM needs CUDA init, weight load, graph capture and KV profiling before
`/health` answers, hence the 120 s floor. Large unquantized models and
first-run kernel compilation (e.g. FP4 JIT) can take far longer: set
`health_timeout`.

**A too-short window is the worst failure.** The health wait gives up while
the server keeps starting; `save_state` is never reached, so the process runs
unregistered and the contract still advertises the previous profile. The
watchdog then treats it under the old profile's policy and `ensure` may
reload the old profile.

Health, contract, watchdog:

- `check_health()` probes `/health` (vLLM answers it), falling back to `/v1/models`.
- The contract sets `primary.backend: vllm`; its `models` map holds both the
  raw model id and its normalized basename, the counterpart of
  `--served-model-name`.
- The idle watchdog reads `vllm:generation_tokens_total` from `/metrics`
  (`_get_predicted_total`).
- `release` restores a stopped vllm profile with `VLLM_DEFAULT_TIMEOUT`, not
  its `health_timeout`; a slow-loading profile fails its automatic restore
  and must be reloaded by hand.

## 7. Gotchas

- **Llama-only pool flags.** Slot `extra_args` often carry `--split-mode` /
  `--tensor-split`; vLLM exits on them, so `start_server()` strips them (flag
  and value) for non-llama backends and skips the `--device` widening.
  Multi-GPU vLLM is `pool:` plus `vllm_params.tensor_parallel_size`.
- **Pooled loads still own the primary port and contract entry.** `pool:`
  only changes `CUDA_VISIBLE_DEVICES`; the server binds `PORT`, the primary
  state file is written, and the contract's `primary.gpu` shows the primary
  card's label even when the model runs on other cards. `slot_start()` is
  llama-only: vLLM cannot run as a slot on its own port.
- **`validate` VRAM check ignores `pool:`.** It compares `vram_required_mb`
  with the primary card's capacity, so a pooled profile sized above that card
  is reported invalid even though `load` accepts it.
- **`validate` cannot tell whether vLLM is installed.** It only checks that
  `VLLM_PYTHON` exists; with the `python3` fallback that is almost always
  true. A missing `vllm` package shows up as a load failure in the log.
- **`params:` is ignored at runtime**; only `validate` mentions it.
- **Lower `gpu_memory_utilization` when a load dies at a fixed allocation.**
  vLLM fills its whole utilization budget with KV cache, and some allocations
  (e.g. CUDA graph capture) happen outside that budget. If a load fails at the
  same allocation size no matter what else you stop, go from 0.95 to 0.90.
  With the 0.95 default, any other process on the card can break a tight fit.
- **Do not judge quality on a cold MoE server.** The first responses after a
  cold start can be poor; send a few warm-up requests first.

## Cross-references

- `model-swap backends vllm`: the same facts, generated from the script.
- Sibling backend docs: `docs/backends/llama.md`, `docs/backends/ds4.md`,
  `docs/backends/freetoken.md`.
