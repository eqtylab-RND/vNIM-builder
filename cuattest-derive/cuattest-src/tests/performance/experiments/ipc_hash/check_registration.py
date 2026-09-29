"""Live fresh-byte, producer-lease and stale-token checks for the prototype."""

import argparse
import json
from pathlib import Path
import sys

from resident import owned_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root / "src"))
    import torch
    from blake3 import blake3
    from cuattest.ipc import (
        share_tensors,
        claim_ipc_refs,
        wire_refs,
        _complete_ipc_refs,
        IpcTensorMutatedError,
    )
    from cuattest.expect import verify_evidence

    refs = None
    keepalive = None
    try:
        with owned_server(
            args.root, args.output, {"label": "fresh", "registered": True}
        ) as (client, info):
            keys = {
                d["device_uuid"]: d["gpu_pubkey_uncompressed"] for d in info["devices"]
            }
            cpu = torch.arange(131073, dtype=torch.int32).view(torch.uint8)
            tensor = cpu.to("cuda:0")
            _, refs, keepalive = share_tensors({"weight": tensor})
            refs = claim_ipc_refs(refs)
            registration = client._rpc(
                "/experiment/register", {"tensors": wire_refs(refs)}
            )
            roots = []
            for generation in range(3):
                if generation:
                    tensor.add_(generation)
                    cpu.add_(generation)
                    # Producer readiness is explicit. No mutation may overlap
                    # a sign; sign completion alone does not release storage.
                    torch.cuda.synchronize(0)
                receipt = client._rpc(
                    "/experiment/sign",
                    {"token": registration["token"], "model": "fresh-byte-test"},
                )
                verified = verify_evidence(receipt, trusted_pubkeys=keys)
                expected = blake3(
                    (1).to_bytes(4, "little") + blake3(cpu.numpy()).digest()
                ).hexdigest()
                assert verified.model_root == expected
                roots.append(expected)
                try:
                    keepalive.release()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(
                        "sign prematurely released a registered producer lease"
                    )
            assert len(set(roots)) == 3
            assert client._rpc(
                "/experiment/unregister", {"token": registration["token"]}
            ) == {"released": True}
            _complete_ipc_refs(refs)
            try:
                keepalive.release()
            except IpcTensorMutatedError:
                pass  # Deliberate between-request writes, now safely unmapped.
            refs = None
            # A released generation must not be reusable, even on the same PID.
            try:
                client._rpc(
                    "/experiment/sign",
                    {"token": registration["token"], "model": "stale"},
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("stale registration was accepted")
            print(
                json.dumps(
                    {
                        "fresh_roots": roots,
                        "unique_handles": registration["unique_handles"],
                        "lease_guard": "passed",
                        "stale_token": "rejected",
                    }
                ),
                flush=True,
            )
    finally:
        # If the test fails before a confirmed unregister, retain the process
        # registry lease. This standalone test exits; it never guesses an ACK.
        pass


if __name__ == "__main__":
    main()
