# Stage-2 derive: layer the cuAttest container *client* onto a stage-1 eqty image.
# The client (cuattest-src) is fetched from cuAttest branch fix/signed-model-asset-cid,
# which provides validate_model_cid (the notary-rnd worker imports it) and matches the
# deployed notary's signed-model-asset-CID protocol. It never builds or runs the signing
# kernel, so the B200 kernel fix does not belong here (that ships in the podvm notary).
#
# BASE = the stage-1 output, so one file serves both images:
#   docker build --build-arg BASE=ghcr.io/eqtylab/vvllm/vllm-gpu-eqty:latest \
#     -t nvidiademoacr.azurecr.io/vcomp-nim/vllm-gpu-eqty:cuattest .
#   docker build --build-arg BASE=ghcr.io/eqtylab/vnim/llama-3.1-nemotron-nano-8b-v1-eqty:latest \
#     --build-arg APP_USER=nim \
#     -t nvidiademoacr.azurecr.io/vcomp-nim/llama-3.1-nemotron-nano-8b-v1-eqty:eqty-cuattest .
ARG BASE=ghcr.io/eqtylab/vvllm/vllm-gpu-eqty:latest
FROM ${BASE}

# APP_USER restores the base image's runtime user after the root-only install
# (root for the vLLM image, nim for the NIM). Redeclared here because an ARG
# before FROM is only visible to FROM.
ARG APP_USER=root

USER root
COPY cuattest-src /tmp/cuattest-src
# Install the cuAttest client into the SAME env eqty_middleware lives in, so the
# middleware's vllm_worker can import it. Mirror vnim-build.sh's installer choice,
# but from one file for both bases: prefer the base env's own pip, and because the
# NIM venv (/opt/nim/llm/.venv) ships a python with no pip module, PROBE that pip
# actually works before using it, else fall back to system pip3 --target that
# site-packages (which also sidesteps PEP 668 externally-managed on py3.12 bases).
RUN set -eux; \
    command -v pip3 >/dev/null && command -v g++ >/dev/null && dpkg -s python3-dev >/dev/null 2>&1 \
      || (apt-get update && apt-get install -y --no-install-recommends g++ python3-dev python3-pip && rm -rf /var/lib/apt/lists/*); \
    SITE="$(find /opt /usr/local/lib /usr/lib -maxdepth 8 -type d -path '*-packages/eqty_middleware' 2>/dev/null | head -1 | xargs -r dirname)"; \
    [ -n "$SITE" ] || { echo "FATAL: eqty_middleware not found in base image (build the stage-1 eqty base first)" >&2; exit 1; }; \
    VENV_ROOT="${SITE%%/lib/*}"; \
    if [ -x "${VENV_ROOT}/bin/pip3" ]; then \
        PIP="${VENV_ROOT}/bin/pip3"; TARGET=""; \
    elif [ -x "${VENV_ROOT}/bin/python3" ] && "${VENV_ROOT}/bin/python3" -m pip --version >/dev/null 2>&1; then \
        PIP="${VENV_ROOT}/bin/python3 -m pip"; TARGET=""; \
    else \
        PIP="pip3"; TARGET="--target ${SITE}"; \
    fi; \
    ${PIP} install ${TARGET} --no-cache-dir "/tmp/cuattest-src[client]"; \
    rm -rf /tmp/cuattest-src; \
    VERIFY_PY="${VENV_ROOT}/bin/python3"; [ -x "$VERIFY_PY" ] || VERIFY_PY="python3"; \
    PYTHONPATH="${SITE}" "$VERIFY_PY" -c "import cuattest.client, cuattest.registered, cuattest.ids; print(\"cuattest import OK\")"

USER ${APP_USER}
