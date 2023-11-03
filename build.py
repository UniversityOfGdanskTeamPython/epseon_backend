from __future__ import annotations

import platform
import sys
from pathlib import Path

import requests

VULKAN_SDK_LINUX_URL = "https://sdk.lunarg.com/sdk/download/1.3.268.0/linux/vulkansdk-linux-x86_64-1.3.268.0.tar.xz"


class Builder:
    """Class responsible for building epseon_backend binaries."""

    def __init__(self, working_directory: Path) -> None:
        """Initialize builder object."""

    def download_vulkan_sdk(self) -> None:
        if platform.system() == "Linux":
            response = requests.get(
                VULKAN_SDK_LINUX_URL,
                allow_redirects=True,
                timeout=300,  # 300 seconds.
            )
            response.content

    def build(self) -> None:
        """Build extension module."""


Builder(Path.cwd() / "build").build()

print(sys.argv)
