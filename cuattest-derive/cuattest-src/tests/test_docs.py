# SPDX-License-Identifier: Apache-2.0
"""Documentation contracts that affect offline use and privacy."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_illustrated_page_fetches_no_external_resources():
    html = (ROOT / "docs" / "index.html").read_text()

    # README promises that opening this file needs nothing from the network.
    # Guard resource attributes and CSS imports/URLs so a web-font or tracking
    # asset cannot silently turn that promise into an outbound request.
    assert re.search(
        r"\b(?:href|src)\s*=\s*['\"]https?://", html, re.IGNORECASE
    ) is None
    assert re.search(
        r"(?:@import|url\s*\()\s*['\"]?https?://", html, re.IGNORECASE
    ) is None
