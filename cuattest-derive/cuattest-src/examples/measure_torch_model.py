#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure a model's resident weights against a running notary.

    cuattest serve &                       # process A: holds the key
    python examples/measure_torch_model.py   # process B: holds the model

Process B loads a model into its own CUDA context and hands the notary IPC
handles to the live allocations. Nothing is copied, and the two processes
never share a secret: B cannot sign, A cannot see the model.

Requires torch, transformers, and the ``verify`` extra (cryptography); the
notary process itself requires none of those client-side packages.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuattest.client import Client, NotaryClientError
from cuattest.expect import verify_evidence
from cuattest.ipc import share_model

URL = os.environ.get("CUATTEST_URL", "http://127.0.0.1:8077")
MODEL_ID = os.environ.get("MODEL_ID", "openai-community/gpt2")


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    client = Client(URL)
    try:
        info = client.info()
    except NotaryClientError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"notary  {info['gpu_did']}")
    print(f"        {info['device']} ({info['arch']}), pid {info['pid']}")
    print(f"        kernel {info['kernel_cid']}")
    print(f"        cubin  {info['cubin_cid']}  [{info['compiler']}]")

    print(f"\nloading {MODEL_ID} into this process...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda").eval()
    _names, refs, keepalive = share_model(model)
    print(f"sharing {len(refs)} tensors over CUDA IPC (zero copy)")

    started = time.perf_counter()
    signed = client.sign(refs, MODEL_ID)
    elapsed = time.perf_counter() - started
    del keepalive          # safe only now that the atomic request has returned

    # measurementDocument is hex-encoded: those are the exact bytes the
    # detached signature covers. verify_evidence checks that signature, the
    # signer DID, and the digest fold before exposing the authenticated root.
    verified = verify_evidence(signed, trusted_pubkey=info["gpu_pubkey_uncompressed"])
    document = json.loads(bytes.fromhex(signed["measurementDocument"]))
    print(f"\nmeasured {verified.tensor_count} tensors in {elapsed:.3f}s")
    print(f"  model_root {verified.model_root}")
    print(f"  vram_cid   {verified.vram_cid}")
    statements = signed.get("statements", {})
    print(f"\nmeasurement document: modelHash {document['modelHash'][:24]}…")
    print(f"                      modelCID  {document['modelCID']}")
    print(f"\nsigned in-kernel: {len(statements)} statements")
    for key, st in statements.items():
        print(f"  {st['@type']:<26} {key}")
    print("\nthe signature and per-tensor digest fold verify")
    # Tear down while torch is fully alive. Client.sign() has already retired
    # cuAttest's process-owned storage leases after the notary acknowledged
    # that it closed every imported mapping.
    del model
    gc.collect()
    torch.cuda.ipc_collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
