# SPDX-License-Identifier: Apache-2.0
"""Keep archived experiments from silently relabeling an edited baseline."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def variants():
    path = Path(__file__).parent / "performance/experiments/variants.py"
    spec = importlib.util.spec_from_file_location("performance_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_baseline_requires_exact_source(variants, monkeypatch):
    source = "public test source\n"
    monkeypatch.setattr(variants, "BASELINE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    assert variants.require_baseline(source) == source
    for different in (source.rstrip(), source + "// edit\n", ""):
        with pytest.raises(ValueError, match="pristine 4708698"):
            variants.require_baseline(different)


@pytest.mark.parametrize("modulus_name", ["P", "N"])
def test_archived_public_chains_prove_their_own_exponents(variants, modulus_name):
    # validate_chain executes the generator's exact square/multiply schedule
    # and asserts that its exponent is modulus-2, not inverse-square modulus-3.
    counts = variants.validate_chain(getattr(variants, modulus_name))
    assert counts["squares"] + counts["multiplies"] < 320
