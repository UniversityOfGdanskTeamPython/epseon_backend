"""Build script for epseon_backend package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent


class Builder:
    """Class responsible for building epseon_backend binaries."""

    DEPS: tuple[tuple[str, str], ...] = (
        ("googletest", "release-1.12.1"),
        ("spdlog", "v1.12.0"),
        ("pybind11", "v2.11.1"),
        ("fmt", "10.1.1"),
        ("vma_hpp", "v3.0.1-3"),
        ("glslang", "13.1.1"),
        ("shaderc", "v2023.7"),
    )

    def __init__(self) -> None:
        """Initialize builder object."""
        self.repo_path = Path(__file__).parent
        self.prepare_submodules()

    def prepare_submodules(self) -> None:
        """Prepare dependency submodules."""
        # Skip submodule initialization when not a git repo, eg. when building from
        # sdist.
        if not (THIS_DIR / ".git").exists():
            return

        self.git("submodule", "update", "--init", "--recursive")
        for dependency_name, dependency_tag in self.DEPS:
            self.git(
                "-C",
                f"{self.repo_path.as_posix()}/external/{dependency_name}",
                "fetch",
                "--tags",
                "--force",
            )
            if dependency_name == "glslang":
                self.python(
                    f"{self.repo_path.as_posix()}/external/glslang/update_glslang_sources.py",
                    cwd=self.repo_path / "external" / "glslang",
                )

            self.git(
                "-C",
                f"{self.repo_path.as_posix()}/external/{dependency_name}",
                "checkout",
                dependency_tag,
            )

    def git(self, *arg: str) -> None:
        """Run git command."""
        return self.shell(["git", *arg])

    def python(self, *arg: str, cwd: Path | None = None) -> None:
        """Run git command."""
        return self.shell([sys.executable, *arg], cwd=cwd)

    def shell(self, args: list[str], cwd: Path | None = None) -> None:
        """Run git command."""
        try:
            r = subprocess.run(
                args=args,
                cwd=self.repo_path.as_posix() if cwd is None else cwd.as_posix(),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            sys.stdout.write(e.stdout.decode("utf-8"))
            sys.stderr.write(e.stderr.decode("utf-8"))
            raise
        else:
            sys.stdout.write(r.stdout.decode("utf-8"))
            sys.stderr.write(r.stderr.decode("utf-8"))

    def build(self) -> None:
        """Build extension module."""
        self.cmake("-S", ".", "-B", "build")
        self.cmake("--build", "build", "--target", "epseon_cpu")
        self.cmake("--build", "build", "--target", "epseon_gpu")

    def cmake(self, *arg: str) -> None:
        """Run cmake command. If fails, raises CalledProcessError."""
        return self.shell(
            [
                sys.executable,
                "-c",
                "import cmake;cmake.cmake()",
                *arg,
            ],
        )


Builder().build()
