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
RUN set -eux; \
    (command -v g++ >/dev/null && dpkg -s python3-dev >/dev/null 2>&1) \
      || (apt-get update && apt-get install -y --no-install-recommends g++ python3-dev && rm -rf /var/lib/apt/lists/*); \
    python3 -m pip install --no-build-isolation --no-cache-dir "/tmp/cuattest-src[client]" \
      || pip3 install --no-build-isolation --no-cache-dir "/tmp/cuattest-src[client]"; \
    rm -rf /tmp/cuattest-src; \
    python3 -c "import cuattest.client, cuattest.registered, cuattest.ids; print(\"cuattest import OK\")"

USER ${APP_USER}
