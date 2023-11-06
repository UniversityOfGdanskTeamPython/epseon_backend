"""Test components of `epseon_backend.device.gpu._libepseon_gpu` submodule."""
from __future__ import annotations

import re


def test_get_vulkan_version() -> None:
    """Check if temporary `greet()` method exported from _libepseon_gpu is available."""
    from epseon_backend.device.gpu._libepseon_gpu import get_vulkan_version

    assert re.match(r"\d+\.\d+\.\d+\.\d+", get_vulkan_version()) is not None
