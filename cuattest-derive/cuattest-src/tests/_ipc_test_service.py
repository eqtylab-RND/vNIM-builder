# SPDX-License-Identifier: Apache-2.0
"""A separate-process HTTP notary for single- and multi-GPU client tests."""

import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from threading import Thread

import pytest

from cuattest.client import Client


@contextmanager
def ipc_test_service(backend, *, minimum_devices=1):
    import torch

    count = torch.cuda.device_count()
    if count < minimum_devices:
        pytest.skip(f"at least {minimum_devices} visible CUDA GPUs are required")
    from cuattest._cuda import Cuda

    cuda = Cuda()
    cuda.init()
    # UUID-based visibility also works when the parent itself is restricted
    # or reordered. Reverse devices to exercise producer/consumer ordinal
    # mismatches; with one GPU this remains a real cross-process IPC test.
    uuids = [cuda.device_uuid(cuda.device(i)) for i in range(count)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(reversed(uuids)),
        CUATTEST_DISABLE_NATIVE_HOST="1" if backend == "fallback" else "0",
    )
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("_multigpu_server.py"))],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    ready = Queue()
    Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
    try:
        message = ready.get(timeout=300)
        assert message, f"notary startup failed (exit {proc.poll()})"
        setup = json.loads(message)
        assert [entry["device_uuid"] for entry in setup["info"]["devices"]] == list(
            reversed(uuids)
        )
        expected_backend = "Python" if backend == "fallback" else "C++"
        assert all(
            entry["host_backend"] == expected_backend
            for entry in setup["info"]["devices"]
        )
        keys = {
            entry["device_uuid"]: entry["gpu_pubkey_uncompressed"]
            for entry in setup["info"]["devices"]
        }
        yield Client(f"http://127.0.0.1:{setup['port']}"), keys, torch
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        proc.stdout.close()
