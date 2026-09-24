# vNIM Builder

Pulls any NIM image, detects its structural family, and builds a vCOMP-patched variant with the eqty integrity middleware wired in.

Published vNIMs are https://github.com/orgs/eqtylab/packages?tab=packages&q=vcomp-nim+-eqty

## Layout

```
.
├── build.sh                            local: pull → detect → build
├── eqty_vcomp_middleware-0.0.10-...whl
```

## Local build

```bash
export NGC_API_KEY=nvapi-...
./build.sh nvcr.io/nim/meta/llama-3.2-3b-instruct
./build.sh nvcr.io/nim/google/gemma-4-31b-it
./build.sh nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:latest

# override output tag
IMAGE_REF=my-thing:dev ./build.sh nvcr.io/nim/...
```

What `build.sh` does:

1. `docker login nvcr.io` with `$NGC_API_KEY`.
2. `docker pull` the image.
3. Run the image briefly to inspect:
   - `id -un` → image user (`nim`, `ubuntu`, `nvs`, …) for the `chown` line.
   - presence of `/opt/nim/llm/nim_llm_sdk/` → **`nim_sdk` family** (older NIMs).
   - presence of `nimlib/nim_inference_api_builder/vllm_api.py` → **`nimlib` family** (newer vLLM-direct NIMs).
   - presence of `vllm_nvext/entrypoints/openai/api_server.py` → **`vllm_nvext` family** (pre-nim_sdk NIMs, e.g. phi-3-mini).
   - if none match, exits with a clear error.
4. `docker build` with `--build-arg PATCH_FAMILY=…` and `--build-arg IMAGE_USER=…`.
5. eqty pypi creds default to `accenture:eqtypartner` but can be overridden via `EQTY_PYPI_USER` / `EQTY_PYPI_PASSWORD`. They're written to a `mktemp -d` directory, mounted as build secrets, and wiped on exit.

## CI build (workflow_dispatch)

`.github/workflows/build-nim.yml` does the same detection on the runner:

1. Inputs: `nim_image`, `nim_tag`, `output_tag` (optional).
2. Logs into `nvcr.io` and `ghcr.io`.
3. Pulls the base, runs an inspection container, sets `patch_family` + `image_user` outputs.
4. `buildah build` with those build-args; `buildah push` to `ghcr.io/eqtylab/vcomp-nim/<short>-eqty:<tag>`.

Required secrets on the repo:

- `NGC_API_KEY`
- `EQTY_PYPI_USER`, `EQTY_PYPI_PASSWORD`

## CI build with vBuild attestation

`.github/workflows/build-nim-vbuild.yml` does the same detect → patch → push pipeline, but on the `vbuild` self-hosted runner with [`eqtylab/vbuild-action`](https://github.com/eqtylab/vbuild-action) wrapping the build. After `buildah build`, the image is exported to `image/<short>-eqty-<tag>.oci.tar` so vbuild can capture, sign (`vcomp-notary`), and produce SLSA + sigstore provenance for it. The same image is then pushed to `ghcr.io` as before.

Requirements on top of the standard build:

- A `vbuild` runner registered to the repo (or org).
- The job carries `id-token: write` for sigstore OIDC.
- Patterned after `../vbuild-examples/.github/workflows/oci-image.yaml`.

The web UI's **Build mode** selector toggles between this workflow and the plain `build-nim.yml`.

## How a NIM gets bucketed

| Family | Detection | Patch | Examples |
|---|---|---|---|
| `vllm_nvext` | `vllm_nvext/entrypoints/openai/api_server.py` exists | `patches/vllm_nvext.patch` (NVIDIA's pre-nim_sdk vLLM wrapper, run as `python -m vllm_nvext...`) | phi-3-mini-4k-instruct |
| `nim_sdk` | `/opt/nim/llm/nim_llm_sdk/` exists | `patches/nim_sdk_v{1,2}.patch` (Dockerfile sniffs `shutdown_task = await serve_http` vs bare `await serve_http` and applies the matching one) | llama-3.2-3b, llama-3.1-70b, deepseek-r1-distill, mistral-small-24b |
| `nimlib` | `nimlib/nim_inference_api_builder/vllm_api.py` exists | `patches/nimlib.patch` (patches NVIDIA's stable wrapper, more durable across vLLM revs) | llama-3.3-nemotron-super-49b, gemma-4-31b-it, gpt-oss-120b |
| `nimlib_sglang` | nimlib present **and** `/opt/nim/inference.py` imports `sglang_api` (SGLang-backed) | `patches/nimlib_sglang.patch` (patches nimlib's base FastAPI classes, runtime-agnostic) | qwen3.6-27b |
| (unknown) | none of the above | error: needs a new patch | — |

`build.sh` disambiguates `nimlib` vs `nimlib_sglang` by what `/opt/nim/inference.py`
imports (both `vllm_api.py` and `sglang_api.py` ship in newer nimlib), not by file
presence.

## Known NIMs

What we have inspected. Three structural families: **vllm_nvext** images (NVIDIA's pre-nim_sdk vLLM wrapper, run as `python -m vllm_nvext...`), **NIM-SDK** images (have `/opt/nim/llm/.venv` and a `nim_llm_sdk` package), and **vLLM-direct via nimlib** images (no NIM SDK, patched through NVIDIA's `nimlib` wrapper).

Sorted oldest → newest by image build date — the older the image, the older the SDK era it was packaged with.

| Image | Built | Family | User | nim_llm_sdk flavor | vLLM version |
|---|---|---|---|---|---|
| `nvcr.io/nim/microsoft/phi-3-mini-4k-instruct:latest` | 2024-10-18 | `vllm_nvext` | nim | n/a (pre-SDK) | `0.5.3.post1` |
| `nvcr.io/nim/deepseek-ai/deepseek-r1-distill-qwen-14b:latest` | 2025-03-05 | `nim_sdk` | nim | V2 (single-hunk variant) | `0.6.4.dev62+g899aa5d7.d20241030` |
| `nvcr.io/nim/mistralai/mistral-small-24b-instruct-2501:latest` | 2025-06-18 | `nim_sdk` | nim | V2 (line ~995) | `0.6.4.dev62+g899aa5d7.d20241030` (same build as DeepSeek) |
| `nvcr.io/nim/meta/llama-3.2-3b-instruct:latest` | 2025-07-25 | `nim_sdk` | nim | V2 (bare `await serve_http`) | `0.7.3+b3c2f048.nv25.03-nim-1.9-test` |
| `nvcr.io/nim/meta/llama-3.1-70b-instruct:1.13.1` | 2025-09-10 | `nim_sdk` (dual-location SDK) | nim | V1 (`shutdown_task = await serve_http`) | `0.9.0.pre1+1958ee56.nv25.06` |
| `nvcr.io/nim/google/gemma-4-31b-it:latest` | 2026-04-02 | `nimlib` | **`nvs`** | n/a | `0.1.dev15033+g2b557cb24` |
| `nvcr.io/nim/openai/gpt-oss-120b:latest` | 2026-04-17 | `nimlib` | nim (gid=root) | n/a | `0.19.0` |
| `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:latest` | 2026-04-17 | `nimlib` | nim (gid=root) | n/a | `0.19.0` (same as gpt-oss) |
| `nvcr.io/nim/qwen/qwen3.6-27b:latest` | 2026+ | `nimlib_sglang` | nim | n/a (SGLang backend) | SGLang `9.13.0.50-1` |

The family transition is visible in the date column: 2024 was `vllm_nvext`; 2025 is `nim_sdk`; 2026+ is `nimlib`. NVIDIA cut over from vllm_nvext → nim_sdk in early 2025, and from nim_sdk → nimlib roughly between Sept 2025 and April 2026.

Image users observed so far: `nim`, `ubuntu`, `nvs`. Only `nim` is in the active inventory above; `ubuntu` was the historical gpt-oss-120b user before its rebase.

The patch step also `grep`s the resulting image for `IntegrityFastAPI` and **fails the build** if the patch silently no-op'd, so a future NVIDIA release that shifts file structure will be loud rather than quiet.

### Why the loud-failure check matters

NIM images get rebased upstream without warning. Concrete example — `nvcr.io/nim/openai/gpt-oss-120b:latest`:

|                | Old `vcomp-nim/gpt-oss-120b/Dockerfile` assumed | Current image actually has               |
|----------------|--------------------------------------------------|------------------------------------------|
| Image user     | `ubuntu`                                         | `nim` (gid=root)                         |
| Patch targets  | `/usr/local/lib/python3.12/dist-packages/` and `/opt/vllm/vllm-src/` | `/opt/nim/.venv/.../nimlib/nim_inference_api_builder/` |
| Patch anchor   | `mount_metrics(app)`                             | `setup_middleware()` + `_initialization_complete` |

Running the old Dockerfile against today's image would have every patch hunk fail silently (the `\|\| true` was swallowing it) and ship a broken image. The new unified flow's `grep IntegrityFastAPI` check catches this immediately.

## Adding support for a new NIM layout

1. Inspect the new image:
   ```bash
   docker run --rm --entrypoint bash $IMG -c '
       id; echo ---
       find / -path /proc -prune -o -name api_server.py -print 2>/dev/null | grep -E "(vllm|vllm_nvext|nim_llm_sdk|nimlib)"
   '
   ```
2. Find the FastAPI app construction site and add a new `patches/<family>.patch`.
3. Add a new `case` in `Dockerfile`'s patch step and a new branch in `build.sh`'s and the workflow's detection logic.

The `vcomp-nim/` directory holds the original per-model Dockerfiles + patches we derived the unified approach from. It's reference material; the active build path is the root.
