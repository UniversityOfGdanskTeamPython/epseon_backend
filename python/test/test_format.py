"""Test utilities from epseon_backend.format module."""
from __future__ import annotations

import pytest
from epseon_backend.format import convert_size_in_bytes_to_adaptive_unit


@pytest.mark.parametrize(
    ("value", "expect"),
    [
        (0, "0.000B"),
        (15, "15.000B"),
        (1024, "1.000KiB"),
        (256 * 1024**2, "256.000MiB"),
        (1024**11, "1073741824.000YiB"),
    ],
)
def test_convert_size_in_bytes_to_adaptive_unit(value: int, expect: str) -> None:
    """Check behavior of convert_size_in_bytes_to_adaptive_unit() with different input values."""
    assert convert_size_in_bytes_to_adaptive_unit(value) == expect
