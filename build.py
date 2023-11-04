"""Build script for epseon_backend package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class Builder:
    """Class responsible for building epseon_backend binaries."""

    def __init__(self) -> None:
        """Initialize builder object."""

    def build(self) -> None:
        """Build extension module."""
        self.cmake("-S", ".", "-B", "build")
        self.cmake("--build", "build", "--target", "epseon_cpu")
        self.cmake("--build", "build", "--target", "epseon_gpu")

    def cmake(self, *arg: str) -> None:
        """Run cmake command. If fails, raises CalledProcessError."""
        try:
            subprocess.run(
                executable=sys.executable,
                args=[
                    sys.executable,
                    "-c",
                    "import cmake;cmake.cmake()",
                    *arg,
                ],
                cwd=Path.cwd().as_posix(),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            sys.stdout.write(e.stdout.decode("utf-8"))
            sys.stderr.write(e.stderr.decode("utf-8"))
            raise


Builder().build()
