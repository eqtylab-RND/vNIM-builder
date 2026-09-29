# SPDX-License-Identifier: Apache-2.0
"""cuattest — a P-256 notary whose signing state is retained inside the GPU.

The notary generates a key in a CUDA kernel, hashes another process's live
VRAM over CUDA IPC, and signs a measurement document and a set of EQTY
statements entirely on the device. The private scalar is not copied back by
cuAttest and dies with the process; the trusted host that supplies its CSPRNG
seed can reproduce it.

    from cuattest import Notary
    n = Notary()
    print(n.info.gpu_did)

or over HTTP:

    $ cuattest serve
    $ curl -s localhost:8077/v1/info
"""

from .ids import did_key_p256, raw_cid
from .notary import GpuInfo, Notary, Measurement, NotaryError, TensorRef
from .multigpu import MultiGpuNotary

__all__ = [
    "GpuInfo", "Notary", "MultiGpuNotary", "Measurement", "NotaryError", "TensorRef",
    "did_key_p256", "raw_cid",
]

__version__ = "0.1.0"
