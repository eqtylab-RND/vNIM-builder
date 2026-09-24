#!/usr/bin/env bash
# Build a vCOMP-patched image; eqty installs from the two local wheels, never the network.
#
#   ./vnim-build.sh                     # multinim vLLM image; pick model with -e MODEL_NAME at run time
#   ./vnim-build.sh MULTI               # same as above (explicit)
#   VNIM_CPU=1 ./vnim-build.sh          # CPU base (builds + runs locally); default base is GPU
#   NGC_API_KEY=nvapi-... ./vnim-build.sh nvcr.io/nim/meta/llama-3.2-3b-instruct   # a specific NIM
#
# Overrides: EQTY_SDK_WHL MIDDLEWARE_WHL VNIM_PLATFORM BASE_IMAGE BASE_TAG MODEL_NAME IMAGE_REF
set -euo pipefail

# no attestations: the manifest index they add won't load into the classic docker store
export BUILDX_NO_DEFAULT_ATTESTATIONS=1

probe() { docker run --rm --platform "$PLATFORM" --entrypoint bash "$1" -c "$2"; }

main() {
    cd "$(dirname "$0")"

    # amd64-only wheels → always build linux/amd64 (emulated on arm64 hosts)
    SDK_WHL="${EQTY_SDK_WHL:-eqty_sdk-2.3.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}"
    MW_WHL="${MIDDLEWARE_WHL:-eqty_vcomp_middleware-0.0.12-py3-none-any.whl}"
    PLATFORM="${VNIM_PLATFORM:-linux/amd64}"

    if [[ -n "${VNIM_CPU:-}" ]]; then
        DEFAULT_VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo; DEFAULT_VLLM_TAG=v0.27.0; VLLM_VARIANT=cpu
    else
        DEFAULT_VLLM_IMAGE=docker.io/vllm/vllm-openai; DEFAULT_VLLM_TAG=latest; VLLM_VARIANT=gpu
    fi

    command -v docker >/dev/null || { echo "error: docker not found on PATH" >&2; exit 1; }

    # the sdk wheel is git-ignored; pull it from the sibling checkout on demand
    if [[ ! -f "$SDK_WHL" && "$SDK_WHL" != */* && -f "../integrity-py/dist/linux-manylinux/$SDK_WHL" ]]; then
        echo ">>> vendoring $SDK_WHL"; cp "../integrity-py/dist/linux-manylinux/$SDK_WHL" ./
    fi
    [[ -f "$SDK_WHL" ]] || { echo "error: sdk wheel not found: $SDK_WHL" >&2; exit 1; }
    [[ -f "$MW_WHL"  ]] || { echo "error: middleware wheel not found: $MW_WHL" >&2; exit 1; }

    if [[ "$PLATFORM" == "linux/amd64" && "$(uname -m)" != "x86_64" ]]; then
        echo "note: $PLATFORM on $(uname -m) is QEMU-emulated (slower)" >&2
        [[ "$(uname -s)" == Darwin && "$VLLM_VARIANT" == gpu ]] && \
            echo "note: GPU image builds here but can't run locally (no GPU passthrough on macOS)" >&2
    fi

    COMMON_ARGS=(--platform "$PLATFORM" --build-arg "MIDDLEWARE_WHL=$(basename "$MW_WHL")" --build-arg "EQTY_SDK_WHL=$(basename "$SDK_WHL")")
    BUILD_DIR="$(mktemp -d)"; trap 'rm -rf "$BUILD_DIR"' EXIT
    render_context "$BUILD_DIR"
    cp "$SDK_WHL" "$MW_WHL" "$BUILD_DIR"/

    if [[ $# -eq 0 || -z "${1:-}" || "$1" == MULTI ]]; then
        build_multinim
    else
        build_nim "$1"
    fi
}

build_multinim() {
    BASE_IMAGE="${BASE_IMAGE:-$DEFAULT_VLLM_IMAGE}"; BASE_TAG="${BASE_TAG:-$DEFAULT_VLLM_TAG}"
    local IMG="${BASE_IMAGE}:${BASE_TAG}"
    local OUTPUT_REF="${IMAGE_REF:-ghcr.io/eqtylab/vvllm/vllm-${VLLM_VARIANT}-eqty:${BASE_TAG}}"

    echo ">>> multinim base: $IMG"
    docker pull --platform "$PLATFORM" "$IMG"

    local IMAGE_USER FLAVOR
    IMAGE_USER=$(docker run --rm --platform "$PLATFORM" --entrypoint id "$IMG" -un | tr -d '\r')
    # flavor from init_app_state: arity in api_server.py, or the launchers entry.py when it's a shim
    # shellcheck disable=SC2016  # $api/$entry expand in the container, not here
    FLAVOR=$(probe "$IMG" '
        api=$(find /opt /usr/local/lib -path "*/vllm/entrypoints/openai/api_server.py" -not -path "*vllm_nvext*" 2>/dev/null | head -1)
        [ -n "$api" ] || { echo MISSING; exit 0; }
        if grep -q "init_app_state(engine_client, app.state, args, supported_tasks)" "$api"; then echo vllm_serve_http
        elif grep -q "init_app_state(engine_client, app.state, args)" "$api"; then echo vllm_pooling
        else
            entry=$(find /opt /usr/local/lib -path "*/vllm/entrypoints/launchers/api_server/entry.py" 2>/dev/null | head -1)
            if [ -n "$entry" ] && grep -q "init_app_state(engine_client, app.state, args, supported_tasks)" "$entry"; then echo vllm_launchers
            else echo UNRECOGNIZED; fi
        fi' | tr -d '\r')
    case "$FLAVOR" in
        vllm_serve_http|vllm_pooling|vllm_launchers) ;;
        *) echo "error: vLLM api_server not found or unrecognized in $IMG ($FLAVOR)" >&2; exit 2 ;;
    esac

    echo "    user: ${IMAGE_USER:-root}  flavor: $FLAVOR  output: $OUTPUT_REF"
    local iid; iid="$(mktemp)"
    docker buildx build --load --iidfile "$iid" -t "$OUTPUT_REF" "${COMMON_ARGS[@]}" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "BASE_TAG=${BASE_TAG}" \
        --build-arg "PATCH_FLAVOR=${FLAVOR}" \
        --build-arg "IMAGE_USER=${IMAGE_USER:-root}" \
        --build-arg "MODEL_NAME=${MODEL_NAME:-}" \
        -f "$BUILD_DIR/Dockerfile.vllm" "$BUILD_DIR"
    # containerd-store buildx may not apply -t; re-tag by built id if so
    docker image inspect "$OUTPUT_REF" >/dev/null 2>&1 || docker tag "$(cat "$iid")" "$OUTPUT_REF"
    rm -f "$iid"

    echo; echo "Built: $OUTPUT_REF"
    if [[ "$VLLM_VARIANT" == gpu ]]; then
        echo "Run:   docker run --rm --gpus all --shm-size=2g -p 8080:8080 -e MODEL_NAME=<org/model> $OUTPUT_REF"
    else
        # on the CPU backend --gpu-memory-utilization caps the RAM fraction reserved
        echo "Run:   docker run --rm --shm-size=2g -p 8080:8080 -e MODEL_NAME=<org/model> $OUTPUT_REF --gpu-memory-utilization 0.5"
    fi
}

build_nim() {
    local IMAGE BASE_IMAGE BASE_TAG OUTPUT_REF IMAGE_USER PATCH_FAMILY
    case "$1" in *:*) IMAGE="$1" ;; *) IMAGE="$1:latest" ;; esac
    BASE_IMAGE="${IMAGE%:*}"; BASE_TAG="${IMAGE##*:}"
    OUTPUT_REF="${IMAGE_REF:-ghcr.io/eqtylab/vnim/${BASE_IMAGE##*/}-eqty:${BASE_TAG}}"

    if ! docker manifest inspect "$IMAGE" >/dev/null 2>&1; then
        [[ -n "${NGC_API_KEY:-}" ]] || { echo "error: cannot reach $IMAGE; set NGC_API_KEY or 'docker login nvcr.io'" >&2; exit 1; }
        echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
    fi

    echo ">>> pulling $IMAGE"
    docker pull --platform "$PLATFORM" "$IMAGE"
    IMAGE_USER=$(docker run --rm --platform "$PLATFORM" --entrypoint id "$IMAGE" -un | tr -d '\r')

    local HAS_NIM_LLM_SDK HAS_NIMLIB_VLLM HAS_NIMLIB_SGLANG HAS_VLLM_NVEXT NIMLIB_BACKEND
    HAS_NIM_LLM_SDK=$(probe "$IMAGE" 'find /opt -maxdepth 6 -type d -name nim_llm_sdk 2>/dev/null | head -1')
    HAS_NIMLIB_VLLM=$(probe "$IMAGE" 'find /opt /usr/local -path "*nimlib/nim_inference_api_builder/vllm_api.py" 2>/dev/null | head -1')
    HAS_NIMLIB_SGLANG=$(probe "$IMAGE" 'find /opt /usr/local -path "*nimlib/nim_inference_api_builder/sglang_api.py" 2>/dev/null | head -1')
    HAS_VLLM_NVEXT=$(probe "$IMAGE" 'find /opt -path "*/vllm_nvext/entrypoints/openai/api_server.py" 2>/dev/null | head -1')
    # newer nimlib ships both vllm_api.py and sglang_api.py; inference.py's import picks the backend
    NIMLIB_BACKEND=$(probe "$IMAGE" 'grep -hoE "from nimlib\.nim_inference_api_builder\.(vllm|sglang)_api" /opt/nim/inference.py 2>/dev/null | head -1')

    if [[ -n "$HAS_NIM_LLM_SDK" ]]; then
        PATCH_FAMILY=nim_sdk
    elif [[ -n "$HAS_NIMLIB_VLLM" || -n "$HAS_NIMLIB_SGLANG" ]]; then
        case "$NIMLIB_BACKEND" in
            *sglang_api) PATCH_FAMILY=nimlib_sglang ;;
            *vllm_api)   PATCH_FAMILY=nimlib ;;
            *) [[ -n "$HAS_NIMLIB_SGLANG" && -z "$HAS_NIMLIB_VLLM" ]] && PATCH_FAMILY=nimlib_sglang || PATCH_FAMILY=nimlib ;;
        esac
    elif [[ -n "$HAS_VLLM_NVEXT" ]]; then
        PATCH_FAMILY=vllm_nvext
    else
        echo "error: unrecognized NIM layout (no nim_llm_sdk, nimlib/{vllm,sglang}_api.py, or vllm_nvext/api_server.py)" >&2
        exit 2
    fi

    echo "    user: ${IMAGE_USER:-root}  family: $PATCH_FAMILY  output: $OUTPUT_REF"
    local iid; iid="$(mktemp)"
    docker buildx build --load --iidfile "$iid" -t "$OUTPUT_REF" "${COMMON_ARGS[@]}" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "BASE_TAG=${BASE_TAG}" \
        --build-arg "PATCH_FAMILY=${PATCH_FAMILY}" \
        --build-arg "IMAGE_USER=${IMAGE_USER:-root}" \
        -f "$BUILD_DIR/Dockerfile" "$BUILD_DIR"
    # containerd-store buildx may not apply -t; re-tag by built id if so
    docker image inspect "$OUTPUT_REF" >/dev/null 2>&1 || docker tag "$(cat "$iid")" "$OUTPUT_REF"
    rm -f "$iid"

    echo; echo "Built: $OUTPUT_REF"
}

render_context() {
    local d="$1"; mkdir -p "$d/patches"

cat > "$d/Dockerfile" <<'__DOCKERFILE__'
ARG BASE_IMAGE=nvcr.io/nim/meta/llama-3.2-3b-instruct
ARG BASE_TAG=latest

FROM ${BASE_IMAGE}:${BASE_TAG}

USER root

ARG PATCH_FAMILY=nim_sdk
ARG IMAGE_USER=nim
ARG MIDDLEWARE_WHL=eqty_vcomp_middleware-0.0.11-py3-none-any.whl
# set to a context wheel → install eqty from local wheels, no network; empty → private index
ARG EQTY_SDK_WHL=
ARG EQTY_PYPI_HOST=eqty-pypi.westus2.cloudapp.azure.com

RUN apt-get update \
 && apt-get install -y --no-install-recommends patch python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY patches/ /tmp/patches/
COPY *.whl /tmp/wheels/

# Apply the patch for this NIM's structural family.
#   nim_sdk: V1/V2 differ by the serve_http call shape; detect per-file, apply the matching patch.
RUN set -eux; \
    case "${PATCH_FAMILY}" in \
        nim_sdk) \
            for d in $(find /opt/nim/llm -maxdepth 6 -type d -name nim_llm_sdk 2>/dev/null); do \
                f="$d/entrypoints/openai/api_server.py"; \
                [ -f "$f" ] || continue; \
                if grep -q "shutdown_task = await serve_http" "$f"; then \
                    flavor=v1; \
                elif grep -qE "^            await serve_http\(" "$f"; then \
                    flavor=v2; \
                else \
                    echo "FATAL: $f matches neither V1 nor V2 NIM SDK shape" >&2; exit 1; \
                fi; \
                echo "patching $d (nim_sdk ${flavor})"; \
                (cd "$d" && patch -p0 -F0 < "/tmp/patches/nim_sdk_${flavor}.patch"); \
            done; \
            ;; \
        nimlib) \
            # vllm-backed nimlib: nim_inference_api_builder drives the app, so patch both the
            # vllm entrypoint and the nimlib base classes.
            for vllm_root in $(find /usr/local/lib /opt -path '*/vllm/entrypoints/openai/api_server.py' -printf '%h\n' 2>/dev/null | sed 's|/vllm/entrypoints/openai$||' | sort -u); do \
                echo "patching vllm at $vllm_root"; \
                (cd "$vllm_root" && patch -p1 -F10 < /tmp/patches/nimlib.patch); \
            done; \
            for nimlib_root in $(find /usr/local/lib /opt -path '*/nimlib/nim_inference_api_builder/vllm_api.py' -printf '%h\n' 2>/dev/null | sed 's|/nimlib/nim_inference_api_builder$||' | sort -u); do \
                echo "patching nimlib base (vllm-backed) at $nimlib_root"; \
                (cd "$nimlib_root" && patch -p1 -F10 < /tmp/patches/nimlib_sglang.patch); \
            done; \
            ;; \
        nimlib_sglang) \
            # sglang-backed nimlib (no vllm package): the app comes from nimlib's base classes,
            # so patch those (api.py / http_api.py) instead of a vllm entrypoint.
            for nimlib_root in $(find /usr/local/lib /opt -path '*/nimlib/nim_inference_api_builder/sglang_api.py' -printf '%h\n' 2>/dev/null | sed 's|/nimlib/nim_inference_api_builder$||' | sort -u); do \
                echo "patching nimlib base at $nimlib_root"; \
                (cd "$nimlib_root" && patch -p1 -F10 < /tmp/patches/nimlib_sglang.patch); \
            done; \
            ;; \
        vllm_nvext) \
            # pre-nim_sdk NIMs (e.g. phi-3-mini): patch vllm_nvext's own api_server.
            for nvext_root in $(find /opt -path '*/vllm_nvext/entrypoints/openai/api_server.py' -printf '%h\n' 2>/dev/null | sed 's|/vllm_nvext/entrypoints/openai$||' | sort -u); do \
                echo "patching vllm_nvext at $nvext_root"; \
                (cd "$nvext_root" && patch -p1 -F10 < /tmp/patches/vllm_nvext.patch); \
            done; \
            ;; \
        *) \
            echo "unknown PATCH_FAMILY=${PATCH_FAMILY}" >&2; exit 1 ;; \
    esac; \
    if ! grep -rq "IntegrityFastAPI" /opt/nim 2>/dev/null && \
       ! grep -rq "IntegrityFastAPI" /usr/local/lib/python3.12 2>/dev/null; then \
        echo "FATAL: patch did not land - IntegrityFastAPI not found" >&2; exit 1; \
    fi

# eqty install: EQTY_SDK_WHL set → both wheels, no network; else middleware wheel +
# sdk from the private index. NIM-SDK NIMs use the /opt/nim/llm/.venv, others system Python.
RUN --mount=type=secret,id=eqty_pypi_user \
    --mount=type=secret,id=eqty_pypi_password \
    set -eux; \
    if [ -d /opt/nim/llm/.venv ]; then \
        SITE_PACKAGES="$(ls -d /opt/nim/llm/.venv/lib/python*/site-packages)"; \
        TARGET="--target ${SITE_PACKAGES}"; \
    else \
        TARGET=""; \
    fi; \
    unset PIP_CONSTRAINT; \
    if [ -n "${EQTY_SDK_WHL}" ]; then \
        pip3 install ${TARGET} \
            "/tmp/wheels/${EQTY_SDK_WHL}" \
            "/tmp/wheels/${MIDDLEWARE_WHL}"; \
    else \
        PYPI_USER="$(cat /run/secrets/eqty_pypi_user)"; \
        PYPI_PASSWORD="$(cat /run/secrets/eqty_pypi_password)"; \
        pip3 install ${TARGET} \
            "/tmp/wheels/${MIDDLEWARE_WHL}" \
            --extra-index-url "http://${PYPI_USER}:${PYPI_PASSWORD}@${EQTY_PYPI_HOST}/simple/" \
            --trusted-host "${EQTY_PYPI_HOST}"; \
    fi; \
    rm -rf /tmp/patches /tmp/wheels

# infra/probe/metadata routes are never attested
ENV EQTY_MIDDLEWARE_EXCLUDE_PATHS="/health*,/v1/models,/v1/health/*,/v1/metrics,/v1/version,/v1/license,/v1/manifest,/v1/metadata,/docs*,/openapi*,/redoc*"
ENV EQTY_MIDDLEWARE_ENABLE_COSAI="true"

# NIM 1.x downloads weights on a tokio runtime sized by hwloc, which under-detects on TDX
# PodVMs and deadlocks the download; force enough workers.
ENV NIM_RUNTIME_MAX_WORKER_THREADS="16"

# base NIMs ship no /opt/nim/.cache, so a fresh named volume mounts as root and the runtime
# user can't write weights; pre-create it owned by IMAGE_USER.
RUN mkdir -p /.eqty_sdk /opt/nim/.cache \
 && chown ${IMAGE_USER} /.eqty_sdk /opt/nim/.cache

# nimlib NIMs front the backend with nginx: expose X-EQTY-Request-ID and route /integrity through.
RUN if [ -f /opt/nim/scripts/nginx_env_vars.sh ]; then \
        sed -i \
            -e 's|"X-Request-Id"|"X-Request-Id, X-EQTY-Request-ID"|' \
            -e "s|'\\^/v1/(license|'^(/integrity\\|/v1/(license|" \
            -e "s|unload_lora_adapter)'|unload_lora_adapter))'|" \
            /opt/nim/scripts/nginx_env_vars.sh; \
        grep -q "X-EQTY-Request-ID" /opt/nim/scripts/nginx_env_vars.sh \
            || { echo "FATAL: bake X-EQTY-Request-ID failed" >&2; exit 1; }; \
        grep -q "/integrity" /opt/nim/scripts/nginx_env_vars.sh \
            || { echo "FATAL: bake /integrity into mgmt pattern failed" >&2; exit 1; }; \
    fi

USER ${IMAGE_USER}
__DOCKERFILE__

cat > "$d/Dockerfile.vllm" <<'__DOCKERFILE_VLLM__'
ARG BASE_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo
ARG BASE_TAG=v0.27.0

FROM ${BASE_IMAGE}:${BASE_TAG}

USER root

# Patch shape, detected by the caller: vllm_serve_http | vllm_pooling | vllm_launchers.
ARG PATCH_FLAVOR=vllm_serve_http
ARG IMAGE_USER=root
ARG MODEL_NAME=""
ARG MIDDLEWARE_WHL=eqty_vcomp_middleware-0.0.12-py3-none-any.whl
ARG EQTY_SDK_VERSION=2.3.0
# set to a context wheel → install eqty from local wheels, no network; empty → private index
ARG EQTY_SDK_WHL=
ARG EQTY_PYPI_HOST=eqty-pypi.westus2.cloudapp.azure.com

RUN apt-get update \
 && apt-get install -y --no-install-recommends patch python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY patches/vllm_serve_http.patch patches/vllm_pooling.patch patches/vllm_launchers.patch /tmp/patches/
COPY *.whl /tmp/wheels/
COPY vllm-entrypoint.sh /opt/eqty/entrypoint.sh

# patch every vllm install and verify the middleware landed
RUN set -eux; \
    patched=""; \
    for vllm_root in $(find /opt /usr/local/lib -path '*/vllm/entrypoints/openai/api_server.py' -not -path '*vllm_nvext*' -printf '%h\n' 2>/dev/null | sed 's|/vllm/entrypoints/openai$||' | sort -u); do \
        echo "patching vllm at $vllm_root (${PATCH_FLAVOR})"; \
        (cd "$vllm_root" && patch -p1 -F10 < "/tmp/patches/${PATCH_FLAVOR}.patch"); \
        grep -rq "IntegrityFastAPI" "$vllm_root/vllm/entrypoints/" \
            || { echo "FATAL: patch did not land in $vllm_root" >&2; exit 1; }; \
        patched="yes"; \
    done; \
    [ -n "$patched" ] || { echo "FATAL: no vllm installation found" >&2; exit 1; }

# install into the env vllm lives in: venv pip if present, else system pip3 --target
RUN --mount=type=secret,id=eqty_pypi_user \
    --mount=type=secret,id=eqty_pypi_password \
    set -eux; \
    SITE_PACKAGES="$(find /opt /usr/local/lib -maxdepth 6 -type d -name vllm -path '*-packages/vllm' 2>/dev/null | head -1 | xargs -r dirname)"; \
    [ -n "$SITE_PACKAGES" ] || { echo "FATAL: could not locate vllm site-packages" >&2; exit 1; }; \
    VENV_ROOT="${SITE_PACKAGES%%/lib/*}"; \
    if [ -x "${VENV_ROOT}/bin/pip3" ]; then \
        PIP="${VENV_ROOT}/bin/pip3"; TARGET=""; \
    elif [ -x "${VENV_ROOT}/bin/python3" ]; then \
        PIP="${VENV_ROOT}/bin/python3 -m pip"; TARGET=""; \
    else \
        PIP="pip3"; TARGET="--target ${SITE_PACKAGES}"; \
    fi; \
    unset PIP_CONSTRAINT; \
    if [ -n "${EQTY_SDK_WHL}" ]; then \
        ${PIP} install ${TARGET} \
            "/tmp/wheels/${EQTY_SDK_WHL}" \
            "/tmp/wheels/${MIDDLEWARE_WHL}"; \
    else \
        PYPI_USER="$(cat /run/secrets/eqty_pypi_user)"; \
        PYPI_PASSWORD="$(cat /run/secrets/eqty_pypi_password)"; \
        ${PIP} install ${TARGET} \
            "eqty-sdk==${EQTY_SDK_VERSION}" \
            "/tmp/wheels/${MIDDLEWARE_WHL}" \
            --extra-index-url "http://${PYPI_USER}:${PYPI_PASSWORD}@${EQTY_PYPI_HOST}/simple/" \
            --trusted-host "${EQTY_PYPI_HOST}"; \
    fi; \
    rm -rf /tmp/patches /tmp/wheels

# infra/probe/metadata routes are never attested
ENV EQTY_MIDDLEWARE_EXCLUDE_PATHS="/health*,/ping,/version,/metrics,/v1/models,/docs*,/openapi*,/redoc*"
ENV EQTY_MIDDLEWARE_ENABLE_COSAI="true"

RUN mkdir -p /.eqty_sdk /tmp/api_logs \
 && chown ${IMAGE_USER} /.eqty_sdk /tmp/api_logs \
 && chmod +x /opt/eqty/entrypoint.sh

# baked default model (optional); override at run with -e MODEL_NAME
ENV MODEL_NAME="${MODEL_NAME}"

EXPOSE 8080

USER ${IMAGE_USER}

ENTRYPOINT ["/opt/eqty/entrypoint.sh"]
CMD []
__DOCKERFILE_VLLM__

cat > "$d/vllm-entrypoint.sh" <<'__ENTRYPOINT__'
#!/bin/bash
# serve $MODEL_NAME (baked, or -e at run); extra args pass through to vllm
set -euo pipefail

if [ -z "${MODEL_NAME:-}" ]; then
    echo "FATAL: MODEL_NAME is not set." >&2
    echo "Bake it at build time or run with -e MODEL_NAME=<org/model>." >&2
    exit 1
fi

exec python3 -m vllm.entrypoints.openai.api_server \
    --host "${VLLM_HOST:-0.0.0.0}" \
    --port "${VLLM_PORT:-8080}" \
    --model "${MODEL_NAME}" \
    "$@"
__ENTRYPOINT__

cat > "$d/patches/nim_sdk_v1.patch" <<'__NIM_SDK_V1_PATCH__'
--- entrypoints/openai/api_server.py    2025-11-29 21:47:14.114410292 -0500
+++ entrypoints/openai/api_server.py    2025-11-29 21:48:12.606537108 -0500
@@ -1229,6 +1229,9 @@
                 set_prompt_telemetry_webhooks(envs.NIM_PROMPT_TELEMETRY_EXPORT_WEBHOOKS)
                 if is_prompt_telemetry_enabled():
                     app_interface.app.add_middleware(PromptTelemetryMiddleware)
+                from eqty_middleware.fastapi import IntegrityFastAPI #V1
+                app_interface.app.add_middleware(IntegrityFastAPI, fastapi_app=app_interface.app, log_directory="/tmp/api_logs")
+

                 shutdown_task = await serve_http(
                     my_app,
__NIM_SDK_V1_PATCH__

cat > "$d/patches/nim_sdk_v2.patch" <<'__NIM_SDK_V2_PATCH__'
--- entrypoints/openai/api_server.py    2025-11-29 21:47:14.114410292 -0500
+++ entrypoints/openai/api_server.py    2025-11-29 21:48:12.606537108 -0500
@@ -1060,6 +1060,8 @@
             set_prompt_telemetry_webhooks(envs.NIM_PROMPT_TELEMETRY_EXPORT_WEBHOOKS)
             if is_prompt_telemetry_enabled():
                 app_interface.app.add_middleware(PromptTelemetryMiddleware)
+            from eqty_middleware.fastapi import IntegrityFastAPI #V2
+            app_interface.app.add_middleware(IntegrityFastAPI, fastapi_app=app_interface.app, log_directory="/tmp/api_logs")

             await serve_http(
                 my_app,
__NIM_SDK_V2_PATCH__

cat > "$d/patches/nimlib_sglang.patch" <<'__NIMLIB_SGLANG_PATCH__'
--- a/nimlib/nim_inference_api_builder/api.py
+++ b/nimlib/nim_inference_api_builder/api.py
@@ -879,6 +879,9 @@
                     f"Invalid middleware {middleware_path}. Must be a function or a class."
                 )
 
+        from eqty_middleware.fastapi import IntegrityFastAPI
+        self.app.add_middleware(IntegrityFastAPI, fastapi_app=self.app, log_directory="/tmp/api_logs", delay_start=True)
+
     async def health_live(self):
         """
         Handler for liveness endpoint.
--- a/nimlib/nim_inference_api_builder/http_api.py
+++ b/nimlib/nim_inference_api_builder/http_api.py
@@ -209,6 +209,15 @@
         """
         self.log_routes()
         self.logger.info("Welcome! Application is ready to receive API requests.")
+        # Initialize eqty integrity middleware now that the model is ready.
+        self.app.middleware_stack = self.app.build_middleware_stack()
+        current = self.app.middleware_stack
+        while hasattr(current, "app"):
+            from eqty_middleware.fastapi import IntegrityFastAPI
+            if isinstance(current, IntegrityFastAPI):
+                current.initalize_model()
+                break
+            current = current.app
 
     def emit_server_ready(self):
         """
__NIMLIB_SGLANG_PATCH__

cat > "$d/patches/nimlib.patch" <<'__NIMLIB_PATCH__'
--- a/vllm/entrypoints/openai/api_server.py
+++ b/vllm/entrypoints/openai/api_server.py
@@ -250,4 +250,10 @@
     app.root_path = args.root_path
+
+    print("EQTY-MARKER: build_app reached, importing middleware", flush=True)
+    from eqty_middleware.fastapi import IntegrityFastAPI
+    app.add_middleware(IntegrityFastAPI, fastapi_app=app, log_directory="/tmp/api_logs", delay_start=True)
+    print("EQTY-MARKER: IntegrityFastAPI added", flush=True)
+
     app.add_middleware(
         CORSMiddleware,
         allow_origins=args.allowed_origins,
@@ -593,5 +598,19 @@
     await init_app_state(engine_client, app.state, args, supported_tasks)

+    # Build the middleware stack eagerly so we can find the IntegrityFastAPI
+    # instance and tell it the engine is ready to register the base model.
+    print("EQTY-MARKER: post-init_app_state, building middleware stack", flush=True)
+    app.middleware_stack = app.build_middleware_stack()
+    current = app.middleware_stack
+    while hasattr(current, "app"):
+        from eqty_middleware.fastapi import IntegrityFastAPI
+        if isinstance(current, IntegrityFastAPI):
+            print("EQTY-MARKER: found IntegrityFastAPI, calling initalize_model", flush=True)
+            current.initalize_model()
+            break
+        current = current.app
+    print("EQTY-MARKER: stack walk done", flush=True)
+
     logger.info("Starting vLLM server on %s", listen_address)

     return await serve_http(
__NIMLIB_PATCH__

cat > "$d/patches/vllm_nvext.patch" <<'__VLLM_NVEXT_PATCH__'
--- a/vllm_nvext/entrypoints/openai/api_server.py
+++ b/vllm_nvext/entrypoints/openai/api_server.py
@@ -674,4 +674,7 @@
         api_compat_single_error_field = args.api_compat_single_error_field

+        from eqty_middleware.fastapi import IntegrityFastAPI
+        app.add_middleware(IntegrityFastAPI, fastapi_app=app, log_directory="/tmp/api_logs", delay_start=True)
+
         app.add_middleware(
             CORSMiddleware,
@@ -796,4 +796,13 @@
         log_served_endpoints(app, args.host, args.port)
         log_example_curl_request(served_model_names[0], args.host, args.port)
+        # Initialize eqty integrity middleware now that the engine is ready.
+        app.middleware_stack = app.build_middleware_stack()
+        current = app.middleware_stack
+        while hasattr(current, "app"):
+            from eqty_middleware.fastapi import IntegrityFastAPI
+            if isinstance(current, IntegrityFastAPI):
+                current.initalize_model()
+                break
+            current = current.app
         uvicorn.run(
             app,
__VLLM_NVEXT_PATCH__

cat > "$d/patches/vllm_pooling.patch" <<'__VLLM_POOLING_PATCH__'
--- a/vllm/entrypoints/openai/api_server.py
+++ b/vllm/entrypoints/openai/api_server.py
@@ -540,6 +540,9 @@

     register_pooling_api_routers(app)

+    from eqty_middleware.fastapi import IntegrityFastAPI
+    app.add_middleware(IntegrityFastAPI, fastapi_app=app, log_directory="/tmp/api_logs", delay_start=True)
+
     app.add_middleware(
         CORSMiddleware,
         allow_origins=args.allowed_origins,
@@ -996,6 +999,17 @@
         app = build_app(args)

         await init_app_state(engine_client, app.state, args)
+        # Initialize the eqty integrity middleware now that the engine is
+        # ready: build the middleware stack eagerly, find the IntegrityFastAPI
+        # instance and let it register the base model.
+        app.middleware_stack = app.build_middleware_stack()
+        current = app.middleware_stack
+        while hasattr(current, "app"):
+            from eqty_middleware.fastapi import IntegrityFastAPI
+            if isinstance(current, IntegrityFastAPI):
+                current.initalize_model()
+                break
+            current = current.app

         logger.info(
             "Starting vLLM API server %d on %s",
__VLLM_POOLING_PATCH__

cat > "$d/patches/vllm_serve_http.patch" <<'__VLLM_SERVE_HTTP_PATCH__'
--- a/vllm/entrypoints/openai/api_server.py
+++ b/vllm/entrypoints/openai/api_server.py
@@ -250,4 +250,8 @@
     app.root_path = args.root_path
+
+    from eqty_middleware.fastapi import IntegrityFastAPI
+    app.add_middleware(IntegrityFastAPI, fastapi_app=app, log_directory="/tmp/api_logs", delay_start=True)
+
     app.add_middleware(
         CORSMiddleware,
         allow_origins=args.allowed_origins,
@@ -593,5 +597,17 @@
     await init_app_state(engine_client, app.state, args, supported_tasks)

+    # Initialize the eqty integrity middleware now that the engine is
+    # ready: build the middleware stack eagerly, find the IntegrityFastAPI
+    # instance and let it register the base model.
+    app.middleware_stack = app.build_middleware_stack()
+    current = app.middleware_stack
+    while hasattr(current, "app"):
+        from eqty_middleware.fastapi import IntegrityFastAPI
+        if isinstance(current, IntegrityFastAPI):
+            current.initalize_model()
+            break
+        current = current.app
+
     logger.info("Starting vLLM server on %s", listen_address)

     return await serve_http(
__VLLM_SERVE_HTTP_PATCH__

cat > "$d/patches/vllm_launchers.patch" <<'__VLLM_LAUNCHERS_PATCH__'
--- a/vllm/entrypoints/launchers/app.py
+++ b/vllm/entrypoints/launchers/app.py
@@ -42,6 +42,9 @@
     app.state.args = args
     app.root_path = args.root_path
 
+    from eqty_middleware.fastapi import IntegrityFastAPI
+    app.add_middleware(IntegrityFastAPI, fastapi_app=app, log_directory="/tmp/api_logs", delay_start=True)
+
     register_api_routers(args, app, supported_tasks, model_config)
 
     # Endpoint plugins are attached last so their routes are registered after all core
--- a/vllm/entrypoints/launchers/api_server/entry.py
+++ b/vllm/entrypoints/launchers/api_server/entry.py
@@ -136,6 +136,15 @@
     app = build_app(args, supported_tasks, model_config)
     await init_app_state(engine_client, app.state, args, supported_tasks)
 
+    app.middleware_stack = app.build_middleware_stack()
+    current = app.middleware_stack
+    while hasattr(current, "app"):
+        from eqty_middleware.fastapi import IntegrityFastAPI
+        if isinstance(current, IntegrityFastAPI):
+            current.initalize_model()
+            break
+        current = current.app
+
     logger.info("Starting vLLM server on %s", listen_address)
 
     return await serve_http(
__VLLM_LAUNCHERS_PATCH__

}

main "$@"
