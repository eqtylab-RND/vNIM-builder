# SPDX-License-Identifier: Apache-2.0
"""`cuattest` command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__, ids
from . import expect as expect_mod
from . import kernel as kmod
from ._hosthash import blake3_digest
from ._build_options import BUILD_TYPES
from ._protocol import parse_kernel_receipt, parse_measurement_document
from ._statements import MANIFEST_VERSION, validate_statement_graph_structure
from .notary import (
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_REQUEST_TENSORS,
    DEFAULT_MAX_REQUEST_TILES,
    GpuCleanupUncertainError,
    GpuSessionAbortedError,
    Notary,
    NotaryError,
    _QueuedGpuWorkUnconfirmedError,
)


def _hash_owned_device_buffer(
    notary: Notary, buffer, nbytes: int, *, offset: int = 0
) -> bytes:
    """Hash a CLI-owned buffer without freeing it after context quarantine."""
    try:
        return notary.hash_dptr(buffer.ptr + offset, nbytes)
    except (GpuSessionAbortedError, GpuCleanupUncertainError):
        # hash_dptr already attempted context destruction. Ownership has moved
        # to that context, even when the driver could not confirm destruction.
        buffer.ptr = 0
        raise


def _launch_owned_fused_buffers(
    notary: Notary, buffers, sizes: list[int], measured_at: str, model: bytes
):
    """Launch the private self-test path with context-owned failure cleanup."""
    try:
        return notary._launch_fused_active(
            [(buffer.ptr, size) for buffer, size in zip(buffers, sizes)],
            measured_at,
            model,
        )
    except _QueuedGpuWorkUnconfirmedError as error:
        # The fused self-test bypasses public hash_dptr so it must perform the
        # same context-level quarantine itself. Never let the outer cleanup
        # issue cuMemFree while queued work may still reference these buffers.
        for buffer in buffers:
            buffer.ptr = 0
        notary._abort_unconfirmed_gpu_work(error)


def _log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [cuattest] %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def cmd_info(args) -> int:
    with_notary(args, lambda n: print(json.dumps(n.info.as_dict(), indent=1)))
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    selection = getattr(args, "devices", None)
    factory = Notary
    device_args = {"device": args.device}
    if selection is not None:
        from .multigpu import MultiGpuNotary

        factory = MultiGpuNotary
        try:
            devices = (
                None if selection == "all" else [int(d) for d in selection.split(",")]
            )
        except ValueError as error:
            raise NotaryError(
                "--devices must be all or comma-separated CUDA ordinals"
            ) from error
        device_args = {"devices": devices}
    notary = factory(
        **device_args,
        artifact_dir=args.artifacts,
        max_request_tensors=args.max_request_tensors,
        max_request_bytes=args.max_request_bytes,
        max_request_tiles=args.max_request_tiles,
    )
    try:
        serve(notary, args.host, args.port)
    finally:
        notary.close()
    return 0


def cmd_build_kernel(args) -> int:
    """Compile the kernel to CUBINs. Needs NVRTC; needs no GPU."""
    arches = args.arch or _detect_arch_or_default(args.device)
    out = Path(args.out) if args.out else kmod.cache_dir()
    built, failed = [], []
    for arch in arches:
        try:
            options = {}
            if getattr(args, "build_type", None) is not None:
                options["build_type"] = args.build_type
            cubin, version, path = kmod.build_cubin(arch, out, **options)
            digest_note = f"{len(cubin)} bytes"
            built.append((arch, path, digest_note, version))
            print(f"  {arch:<8} {digest_note:>12}  -> {path}")
        except Exception as e:  # noqa: BLE001 - one arch failing must not stop the rest
            failed.append((arch, e))
            print(f"  {arch:<8} {'SKIPPED':>12}  {e}", file=sys.stderr)
    if not built:
        print("no architecture compiled", file=sys.stderr)
        return 1
    print(f"\nNVRTC {built[0][3]}: built {len(built)}/{len(arches)} into {out}")
    if failed:
        print(f"skipped: {', '.join(a for a, _ in failed)}", file=sys.stderr)
    return 0


def _detect_arch_or_default(device: int = 0) -> list[str]:
    """The selected GPU's architecture, or a broad set when CUDA is unavailable."""
    try:
        from ._cuda import Cuda

        cu = Cuda()
        cu.init()
        # Ordinals can have different capabilities on a heterogeneous host;
        # build for the same --device that will later load the artifact.
        dev = cu.device(device)
        major, minor = cu.compute_capability(dev)
        return [f"sm_{major}{minor}"]
    except Exception:  # noqa: BLE001 - unavailable CUDA falls back to build targets
        return ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"]


def cmd_measure(args) -> int:
    """Measure tensors described by a JSON file (as /v1/measure would take)."""
    body = json.loads(Path(args.tensors).read_text())
    from .notary import TensorRef

    refs = [TensorRef.from_dict(t) for t in body["tensors"]]

    def run(n: Notary):
        if args.sign:
            print(json.dumps(n.sign(refs, args.sign), indent=1))
        else:
            print(json.dumps(n.measure(refs).as_dict(), indent=1))

    with_notary(args, run)
    return 0


def cmd_expect(args) -> int:
    """What CID would the notary report for this on-disk checkpoint?"""
    compare_path = args.compare
    trusted_pubkey = getattr(args, "trusted_pubkey", None)
    key_file = getattr(args, "trusted_pubkeys", None)
    trusted_pubkeys = None
    if key_file:
        if trusted_pubkey:
            raise NotaryError("use either --trusted-pubkey or --trusted-pubkeys")
        from ._protocol import loads_no_duplicate_fields
        from .multigpu import _UUID

        try:
            with Path(key_file).open("rb") as keys:
                encoded = keys.read(128 * 1024 + 1)
            if len(encoded) > 128 * 1024:
                raise ValueError("trusted key file is too large")
            trusted_pubkeys = loads_no_duplicate_fields(encoded)
            if not isinstance(trusted_pubkeys, dict) or not trusted_pubkeys:
                raise ValueError("expected a nonempty UUID-to-key map")
            for uid, pin in trusted_pubkeys.items():
                if (
                    _UUID.fullmatch(uid) is None
                    or type(pin) is not str
                    or len(pin) != 130
                ):
                    raise ValueError("expected GPU UUIDs and 65-byte P-256 keys")
                raw = bytes.fromhex(pin)
                if len(raw) != 65 or raw[0] != 4:
                    raise ValueError("expected uncompressed P-256 public keys")
        except (OSError, ValueError) as error:
            raise NotaryError(f"invalid --trusted-pubkeys file: {error}") from error
    # Pins must be supplied independently before opening CUDA. A receipt's own
    # public keys establish self-consistency only, for one GPU or many.
    if compare_path is not None and not trusted_pubkey and not trusted_pubkeys:
        raise NotaryError(
            "--compare requires --trusted-pubkey or --trusted-pubkeys from a trusted channel; "
            "a receipt cannot authenticate the public key it supplies itself"
        )
    if compare_path is not None and trusted_pubkey:
        try:
            trusted_pubkey_bytes = bytes.fromhex(trusted_pubkey)
        except (TypeError, ValueError) as e:
            raise NotaryError("--trusted-pubkey must be hexadecimal") from e
        if len(trusted_pubkey_bytes) != 65 or trusted_pubkey_bytes[0] != 0x04:
            raise NotaryError(
                "--trusted-pubkey must be a 65-byte uncompressed P-256 public key"
            )
        trusted_pubkey = trusted_pubkey_bytes.hex()

    def progress(i, n, name):
        if not args.quiet:
            print(
                f"\r  hashing {i}/{n}  {name[:56]:<56}",
                end="",
                file=sys.stderr,
                flush=True,
            )

    def run(n: Notary):
        exp = expect_mod.compute(
            n, args.model, progress=progress, tied=not args.no_tied
        )
        if not args.quiet:
            print("\r" + " " * 72 + "\r", end="", file=sys.stderr)

        if compare_path is not None:
            measured = expect_mod.load_measurement(compare_path)
            if trusted_pubkeys is not None:
                result = expect_mod.compare(
                    exp, measured, trusted_pubkeys=trusted_pubkeys
                )
            else:
                result = expect_mod.compare(
                    exp, measured, trusted_pubkey=trusted_pubkey
                )
            if args.json:
                print(
                    json.dumps(
                        {
                            "expected": exp.as_dict(),
                            "matches": result.matches,
                            "reason": result.reason,
                            "differing": result.differing,
                        },
                        indent=1,
                    )
                )
            else:
                print(result.report())
            # A mismatch is a finding, not a crash — but it must not exit 0.
            raise SystemExit(0 if result.matches else 2)

        if args.json:
            print(json.dumps(exp.as_dict(), indent=1))
        else:
            gb = exp.total_bytes / 1e9
            print(f"  tensors    {exp.tensor_count}  ({gb:.2f} GB on disk)")
            print(f"  model_root {exp.model_root}")
            print(f"  vram_cid   {exp.vram_cid}")
            for t in exp.tied:
                print(f"  tied       {t}  (restored from explicit schema/model config)")
            print(
                "\n  This is the CID for these checkpoint bytes in canonical span order."
            )
            print(
                "  It matches equivalent submitted VRAM spans; it does not prove those"
            )
            print("  spans were used for inference — see docs/expected-cid.md.")

    with_notary(args, run)
    return 0


def cmd_selftest(args) -> int:
    """Prove the whole path end to end: keygen, hash, measure, sign."""
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(
            f"  {'ok  ' if cond else 'FAIL'} {label}"
            + (f"  {detail}" if detail else "")
        )

    # Host-side encoders, against a fixed vector - no GPU needed.
    empty_b3 = bytes.fromhex(
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    check(
        "CID encoding matches the reference vector",
        ids.raw_cid(empty_b3)
        == "bafkr4ifpcne3t5pzugtkaqcn5i3nzskjtpfslsnnyejlpte2spfoihzsmi",
    )
    check(
        "did:key is well formed",
        ids.did_key_p256(bytes(32), bytes(32)).startswith("did:key:z"),
    )

    def run(n: Notary):
        nonlocal ok
        check(
            "session key generated",
            n.info.gpu_did.startswith("did:key:z"),
            n.info.gpu_did,
        )
        check(
            "kernel + cubin registered",
            n.info.kernel_cid.startswith("bafkr4")
            and n.info.cubin_cid.startswith("bafkr4"),
        )

        # The GPU's BLAKE3 must equal the reference BLAKE3 of the same bytes.
        digest = n.hash_bytes(b"")
        check(
            "in-GPU BLAKE3 of empty input matches the known digest",
            digest == empty_b3,
            digest.hex(),
        )
        # And it must be size-independent: a multi-chunk input exercises the
        # tree reduction rather than the single-chunk path.
        big = bytes(range(256)) * 40  # 10240 bytes -> 10 chunks
        d1 = n.hash_bytes(big)
        # Determinism alone would let a consistently wrong tree reduction pass.
        # Compare across the trust boundary to the independent host library.
        check(
            "multi-chunk hashing matches host BLAKE3",
            d1 == blake3_digest(big),
            d1.hex()[:32] + "…",
        )

        # Exercise every vector-load boundary and large reduction widths around
        # 128 KiB. These values keep the aligned 1 KiB fast path honest without
        # changing BLAKE3's fixed 1 KiB cryptographic chunk format.
        boundary_sizes = (
            1,
            3,
            63,
            64,
            65,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
            127 * 1024,
            128 * 1024,
            129 * 1024,
            255 * 1024,
            256 * 1024,
            257 * 1024,
            300 * 1024 + 17,
            383 * 1024,
            384 * 1024,
            385 * 1024,
        )
        boundary_ok = True
        for size in boundary_sizes:
            payload = bytes((index * 131 + 17) & 0xFF for index in range(size))
            boundary_ok &= n.hash_bytes(payload) == blake3_digest(payload)
        check("aligned fast path matches host BLAKE3 at chunk boundaries", boundary_ok)

        # Measure this process's own device buffer through the same path a
        # remote caller would use, minus IPC.
        from ._cuda import DeviceBuffer

        with n._activate():
            buf = DeviceBuffer.from_bytes(n.cu, big)
            try:
                check(
                    "device-range hash equals host-uploaded hash",
                    _hash_owned_device_buffer(n, buf, len(big)) == d1,
                )
            finally:
                buf.close()

        # Tensor views need not begin at an address suitable for uint4 loads.
        # Cover every non-zero alignment modulo 16 so the byte-safe fallback
        # cannot silently diverge from the optimized path.
        unaligned_payload = bytes((index * 29 + 7) & 0xFF for index in range(2049))
        unaligned_digest = blake3_digest(unaligned_payload)
        unaligned_ok = True
        with n._activate():
            for offset in range(1, 16):
                buf = DeviceBuffer.from_bytes(n.cu, bytes(offset) + unaligned_payload)
                try:
                    unaligned_ok &= (
                        _hash_owned_device_buffer(
                            n, buf, len(unaligned_payload), offset=offset
                        )
                        == unaligned_digest
                    )
                finally:
                    buf.close()
        check("unaligned device spans match host BLAKE3", unaligned_ok)

        # Exercise the production cooperative path with tensors whose BLAKE3
        # trees have different depths and tile counts. Keeping one request on
        # both sides of 128 KiB catches descriptor-base and cross-tile merges,
        # not merely the single-tensor boundary cases above.
        payloads = [
            b"",
            bytes(range(256)) * 12,
            bytes((index * 67 + 11) & 0xFF for index in range(129 * 1024 + 3)),
            bytes((index * 43 + 19) & 0xFF for index in range(300 * 1024 + 17)),
            big,
        ]
        expected_roots = b"".join(blake3_digest(payload) for payload in payloads)
        expected_model_root = blake3_digest(
            len(payloads).to_bytes(4, "little") + expected_roots
        )
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with n._activate():
            buffers = [DeviceBuffer.from_bytes(n.cu, payload) for payload in payloads]
            try:
                fused = _launch_owned_fused_buffers(
                    n,
                    buffers,
                    [len(payload) for payload in payloads],
                    ts,
                    b"selftest",
                )
            finally:
                for buffer in buffers:
                    buffer.close()
        check(
            "fused multi-tensor roots match host BLAKE3", fused.roots == expected_roots
        )
        check(
            "fused model-root fold matches host BLAKE3",
            fused.model_root == expected_model_root,
        )

        text = fused.receipt or ""
        try:
            doc = parse_kernel_receipt(text)
        except (TypeError, ValueError) as error:
            check("fused kernel returned an exact JSON receipt", False, str(error))
            return
        check("fused kernel returned parseable JSON", isinstance(doc, dict))
        check("measurement document present", "measurementDocument" in doc)
        check(
            "detached signature present",
            len(doc.get("measurementSignature", "")) == 128,
        )
        try:
            document_bytes = bytes.fromhex(doc["measurementDocument"])
            signed_doc = parse_measurement_document(document_bytes)
        except (KeyError, TypeError, ValueError) as error:
            check("measurement document matches its exact schema", False, str(error))
            return
        check(
            "selected device is signed",
            signed_doc.get("device") == f"cuda:{n.device_ordinal}",
        )
        check(
            "source and CUBIN identities are signed",
            signed_doc.get("kernelCID") == f"urn:cid:{n.info.kernel_cid}"
            and signed_doc.get("cubinCID") == f"urn:cid:{n.info.cubin_cid}",
        )
        manifest = doc.get("manifest", {})
        st = manifest.get("statements", {}) if isinstance(manifest, dict) else {}
        types = sorted(v.get("@type") for v in st.values())
        subject_types = sorted(
            tuple(v.get("credential", {}).get("type", ())) for v in st.values()
        )
        check(
            "manifest carries the registered state attestation",
            isinstance(manifest, dict)
            and manifest.get("version") == MANIFEST_VERSION
            and len(st) == 1
            and types == ["CredentialRegistration"]
            and subject_types == [("VerifiableCredential", "StateAttestation")],
            f"{types} {subject_types}",
        )
        try:
            validate_statement_graph_structure(manifest, signed_doc)
        except ValueError as error:
            check(
                "statement graph matches the signed measurement exactly",
                False,
                str(error),
            )
        else:
            check("statement graph matches the signed measurement exactly", True)
            # cryptography is a base dependency now that the service issues
            # its own IdentityAttestation, so selftest always crosses the
            # consumer trust boundary and authenticates the kernel's JWS.
            evidence = dict(doc)
            evidence["gpu_pubkey_uncompressed"] = n.info.gpu_pubkey_uncompressed
            try:
                verified = expect_mod.verify_evidence(
                    evidence,
                    trusted_pubkey=n.info.gpu_pubkey_uncompressed,
                )
            except expect_mod.VerificationUnavailableError:
                print("  skip credential proof signatures (install cuattest[verify])")
            except expect_mod.EvidenceError as error:
                check("all credential proof signatures verify", False, str(error))
            else:
                check(
                    "all credential proof signatures verify",
                    verified.model_root == fused.model_root.hex(),
                )

    try:
        with_notary(args, run)
    except NotaryError as e:
        print(f"  FAIL {e}")
        ok = False
    print("\n  ALL CHECKS PASS" if ok else "\n  SOME CHECKS FAILED")
    return 0 if ok else 1


def with_notary(args, fn):
    notary = Notary(
        device=getattr(args, "device", 0), artifact_dir=getattr(args, "artifacts", None)
    )
    try:
        fn(notary)
    finally:
        notary.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="cuattest",
        description="A P-256 notary whose signing state is retained inside the GPU.",
    )
    p.add_argument("--version", action="version", version=f"cuattest {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--device", type=int, default=0, help="CUDA device ordinal (default 0)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="print this session's identity and registered code")
    s.add_argument(
        "--artifacts", help="directory to write the kernel source and CUBIN to"
    )
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("serve", help="run the HTTP notary")
    s.add_argument(
        "--devices",
        help="serve multiple GPUs: all or comma-separated ordinals (overrides --device)",
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8077)
    s.add_argument(
        "--max-request-tensors",
        type=int,
        default=DEFAULT_MAX_REQUEST_TENSORS,
        help=f"maximum spans per request (default {DEFAULT_MAX_REQUEST_TENSORS})",
    )
    s.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BYTES,
        help=f"maximum aggregate logical bytes (default {DEFAULT_MAX_REQUEST_BYTES})",
    )
    s.add_argument(
        "--max-request-tiles",
        type=int,
        default=DEFAULT_MAX_REQUEST_TILES,
        help=f"maximum aggregate 128 KiB scheduling tiles (default {DEFAULT_MAX_REQUEST_TILES})",
    )
    s.add_argument(
        "--artifacts", help="directory to write the kernel source and CUBIN to"
    )
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser(
        "build-kernel", help="compile the kernel to CUBINs (no GPU required)"
    )
    s.add_argument(
        "--arch", action="append", help="e.g. sm_90; repeatable. Default: this GPU's"
    )
    s.add_argument("--out", help=f"output directory (default {kmod.cache_dir()})")
    s.add_argument("--build-type", choices=BUILD_TYPES,
                   help="CUBIN configuration (default: installed package's build type)")
    s.set_defaults(fn=cmd_build_kernel)

    s = sub.add_parser("measure", help="measure tensors listed in a JSON file")
    s.add_argument(
        "tensors",
        help='JSON: {"tensors": [{"handle": "...", "nbytes": N, "device": D}, ...]}',
    )
    s.add_argument("--sign", metavar="MODEL", help="also sign, naming this model")
    s.add_argument("--artifacts", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_measure)

    s = sub.add_parser(
        "expect", help="compute the canonical span CID for an on-disk checkpoint"
    )
    s.add_argument(
        "model",
        help="a .safetensors file, a *.safetensors.index.json, or a directory",
    )
    s.add_argument(
        "--compare",
        metavar="FILE",
        help="a signed receipt to verify and compare (exit 2 on mismatch)",
    )
    s.add_argument(
        "--trusted-pubkey",
        metavar="HEX",
        help="trusted 65-byte P-256 signer key for single-GPU --compare",
    )
    s.add_argument(
        "--trusted-pubkeys",
        metavar="FILE",
        help="trusted JSON map of GPU UUID to P-256 key for multi-GPU --compare",
    )
    s.add_argument("--json", action="store_true")
    s.add_argument("--quiet", action="store_true", help="no progress output")
    s.add_argument(
        "--no-tied",
        action="store_true",
        help="do not restore full-span aliases from schema/model config",
    )
    s.add_argument("--artifacts", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_expect)

    s = sub.add_parser(
        "selftest", help="exercise keygen, hashing, measurement and signing"
    )
    s.add_argument("--artifacts", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_selftest)

    args = p.parse_args(argv)
    _log(args.verbose)
    try:
        return args.fn(args)
    except NotaryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
