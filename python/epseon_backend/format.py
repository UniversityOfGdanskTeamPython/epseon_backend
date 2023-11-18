"""Format utilities."""
from __future__ import annotations

from decimal import Decimal

_BINARY_PREFIX_RANGES = [
    (1, ""),
    (1024**1, "Ki"),
    (1024**2, "Mi"),
    (1024**3, "Gi"),
    (1024**4, "Ti"),
    (1024**5, "Pi"),
    (1024**6, "Ei"),
    (1024**7, "Zi"),
    (1024**8, "Yi"),
    (1024**9, ""),
]
_IF_NONE_MATCHES_PREFIX = _BINARY_PREFIX_RANGES[8]


def convert_size_in_bytes_to_adaptive_unit(value: int) -> str:
    """Return size in bytes with highest possible prefix."""
    if value < 0:
        msg = f"Values below 0 are not allowed, got {value}"
        raise ValueError(msg)

    decimal_value = Decimal(value)

    for i, (multiplier, name) in enumerate(_BINARY_PREFIX_RANGES[:-1]):
        next_multiplier, _ = _BINARY_PREFIX_RANGES[i + 1]

        if (multiplier - 1) <= decimal_value < next_multiplier:
            converted_value = decimal_value / multiplier
            return f"{converted_value:.3f}{name}B"

    multiplier, name = _IF_NONE_MATCHES_PREFIX
    converted_value = decimal_value / multiplier
    return f"{converted_value:.3f}{name}B"
