"""Subpackage for interacting with GPU compute capabilities."""
from __future__ import annotations

from typing import Any

from epseon_backend.device.gpu._libepseon_gpu import greet


def get_device_info() -> Any:
    """Get all accessible information about GPU properties."""
    greet()
