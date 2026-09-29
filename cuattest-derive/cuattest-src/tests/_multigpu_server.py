# SPDX-License-Identifier: Apache-2.0
"""Isolated HTTP/CUDA consumer used by the opt-in multi-GPU integration test."""

import json

from cuattest.multigpu import MultiGpuNotary
from cuattest.server import _NotaryHTTPServer, make_handler


def main():
    notary = MultiGpuNotary()
    server = None
    try:
        server = _NotaryHTTPServer(("127.0.0.1", 0), make_handler(notary))
        # stdout is the test's trusted setup channel; pins are captured before
        # any untrusted receipt is received, not inferred from signed evidence.
        print(
            json.dumps({"port": server.server_port, "info": notary.info.as_dict()}),
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        notary.close()


if __name__ == "__main__":
    main()
