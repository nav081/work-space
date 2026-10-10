"""Manifest-aware Python, Node.js, and .NET dependency installation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool

from threadline_backend.core.constants import (
    COMMAND_ERROR_OUTPUT_LIMIT,
    DEPENDENCY_ERROR_OUTPUT_LIMIT,
    DEPENDENCY_SETUP_TIMEOUT_SECONDS,
    PYTHON_MANIFEST_PRIORITY,
    PYTHON_ENV_CREATE_TIMEOUT_SECONDS,
    SUPPORTED_DEPENDENCY_ECOSYSTEMS,
    SUPPORTED_DOTNET_SUFFIXES,
    SUPPORTED_PYTHON_MANIFESTS,
)
from threadline_backend.infrastructure.command_tools import launch_command, project_python
from threadline_backend.infrastructure.project_paths import safe_project_file


def build_dependency_tools(root: Path, enabled: set[str]) -> list[StructuredTool]:
    """Expose the dependency installer only when the agent has permission."""
    if "Install dependencies" not in enabled:
        return []

    def install_project_dependencies(ecosystem: str = "auto", manifest_name: str = "") -> str:
        """Install dependencies described by a supported project manifest."""
        ecosystem = ecosystem.casefold().strip()
        if ecosystem not in SUPPORTED_DEPENDENCY_ECOSYSTEMS:
            return "Install rejected: ecosystem must be auto, python, node, or dotnet."
        manifest = _select_manifest(root, ecosystem, manifest_name)
        if isinstance(manifest, str):
            return manifest
        ecosystem = _detect_ecosystem(root, ecosystem, manifest)
        if ecosystem not in SUPPORTED_DEPENDENCY_ECOSYSTEMS:
            return ecosystem
        try:
            command, cwd, manifest_name, environment = _install_command(root, ecosystem, manifest, bool(manifest_name))
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            return f"Dependency setup failed: {error}"
        try:
            result = subprocess.run(launch_command(command), cwd=cwd, shell=False, capture_output=True, text=True, timeout=DEPENDENCY_SETUP_TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"Dependency setup failed for {manifest_name or ecosystem}: {error}"
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"Dependency setup failed (exit {result.returncode}) for {manifest_name or ecosystem}:\n{output[-DEPENDENCY_ERROR_OUTPUT_LIMIT:]}"
        return f"Dependencies installed into {environment} from {manifest_name or ecosystem}.\n{output[-DEPENDENCY_ERROR_OUTPUT_LIMIT:]}"

    return [StructuredTool.from_function(install_project_dependencies, name="install_project_dependencies")]


def _select_manifest(root: Path, ecosystem: str, manifest_name: str) -> Path | str:
    if not manifest_name:
        return root
    try:
        manifest = safe_project_file(root, manifest_name)
    except ValueError as error:
        return f"Install rejected: {error}"
    if not manifest.is_file():
        return f"Install failed: {manifest_name} was not found in the project."
    if ecosystem == "auto":
        if manifest.name.casefold() == "package.json":
            return manifest
        if manifest.name.casefold() in SUPPORTED_PYTHON_MANIFESTS:
            return manifest
        if manifest.suffix.casefold() in SUPPORTED_DOTNET_SUFFIXES:
            return manifest
    return manifest


def _detect_ecosystem(root: Path, ecosystem: str, manifest: Path) -> str:
    if ecosystem != "auto":
        return ecosystem
    if manifest != root:
        name = manifest.name.casefold()
        if name == "package.json":
            return "node"
        if name in SUPPORTED_PYTHON_MANIFESTS:
            return "python"
        if manifest.suffix.casefold() in SUPPORTED_DOTNET_SUFFIXES:
            return "dotnet"
    if (root / "package.json").is_file():
        return "node"
    if any(root.glob("*.sln")) or any(root.glob("*.slnx")) or any(root.glob("*.csproj")):
        return "dotnet"
    if any((root / name).is_file() for name in SUPPORTED_PYTHON_MANIFESTS):
        return "python"
    return "No supported dependency manifest found. Expected Python requirements/pyproject, Node package.json, or a .NET solution/project."


def _install_command(root: Path, ecosystem: str, manifest: Path, has_manifest: bool) -> tuple[list[str], Path, str, str]:
    if ecosystem == "python":
        return _python_install(root, manifest, has_manifest)
    if ecosystem == "node":
        return _node_install(root, manifest, has_manifest)
    return _dotnet_install(root, manifest, has_manifest)


def _python_install(root: Path, manifest: Path, has_manifest: bool) -> tuple[list[str], Path, str, str]:
    selected_name = manifest.name if has_manifest else next(
        (name for name in PYTHON_MANIFEST_PRIORITY if (root / name).is_file()),
        "",
    )
    if selected_name not in SUPPORTED_PYTHON_MANIFESTS:
        raise ValueError(f"Install rejected: choose one of {', '.join(sorted(SUPPORTED_PYTHON_MANIFESTS))}.")
    manifest = safe_project_file(root, str(manifest.relative_to(root)) if has_manifest else selected_name)
    if not manifest.is_file():
        raise ValueError(f"Install failed: {manifest.relative_to(root)} was not found in the project.")
    environment_path = root / ".venv"
    if environment_path.is_symlink():
        raise ValueError("Install rejected: the project .venv must not be a symlink.")
    interpreter = project_python(root)
    if interpreter is None:
        if environment_path.exists():
            raise ValueError("Install failed: .venv exists but has no Python executable; repair or remove it first.")
        result = subprocess.run([sys.executable, "-m", "venv", str(environment_path)], cwd=root, shell=False, capture_output=True, text=True, timeout=PYTHON_ENV_CREATE_TIMEOUT_SECONDS, check=False)
        if result.returncode != 0:
            output = (result.stderr or result.stdout)[-COMMAND_ERROR_OUTPUT_LIMIT:]
            raise ValueError(f"Could not create project .venv (exit {result.returncode}): {output}")
        interpreter = project_python(root)
        if interpreter is None:
            raise ValueError("Could not find the Python executable in the newly created .venv.")
    relative_manifest = str(manifest.relative_to(root))
    if selected_name == "pyproject.toml":
        args = [str(interpreter), "-m", "pip", "install", "-e", str(manifest.parent.relative_to(root) or ".")]
    else:
        args = [str(interpreter), "-m", "pip", "install", "-r", relative_manifest]
    return args, root, relative_manifest, "project .venv"


def _node_install(root: Path, manifest: Path, has_manifest: bool) -> tuple[list[str], Path, str, str]:
    package_dir = manifest.parent if has_manifest else root
    if has_manifest and manifest.name.casefold() != "package.json":
        raise ValueError("Install rejected: Node.js installs must target a package.json manifest.")
    package_json = safe_project_file(root, str(package_dir.relative_to(root) / "package.json"))
    if not package_json.is_file():
        raise ValueError("Install failed: package.json was not found in the project.")
    manager, args = _node_manager(package_dir)
    executable = shutil.which(manager)
    if not executable and manager in {"pnpm", "yarn"}:
        corepack = shutil.which("corepack")
        if not corepack:
            raise ValueError(f"{manager} is not installed or available through Corepack.")
        command = [corepack, manager, *args]
    else:
        if not executable:
            raise ValueError(f"{manager} is not installed or available on PATH.")
        command = [executable, *args]
    return command, package_dir, str(package_json.relative_to(root)), "project node_modules"


def _node_manager(package_dir: Path) -> tuple[str, list[str]]:
    if (package_dir / "pnpm-lock.yaml").is_file():
        return "pnpm", ["install", "--frozen-lockfile"]
    if (package_dir / "yarn.lock").is_file():
        return "yarn", ["install", "--frozen-lockfile"]
    if (package_dir / "package-lock.json").is_file() or (package_dir / "npm-shrinkwrap.json").is_file():
        return "npm", ["ci"]
    return "npm", ["install"]


def _dotnet_install(root: Path, manifest: Path, has_manifest: bool) -> tuple[list[str], Path, str, str]:
    if has_manifest:
        if manifest.suffix.casefold() not in SUPPORTED_DOTNET_SUFFIXES:
            raise ValueError("Install rejected: .NET installs must target a .sln, .slnx, or .csproj manifest.")
        selected = manifest
    else:
        solutions = sorted([*root.glob("*.sln"), *root.glob("*.slnx")])
        projects = sorted(root.glob("*.csproj"))
        if solutions:
            selected = solutions[0]
        elif len(projects) == 1:
            selected = projects[0]
        elif len(projects) > 1:
            raise ValueError("Multiple .NET projects found. Specify the .sln or .csproj manifest to restore.")
        else:
            raise ValueError("No .NET .sln, .slnx, or .csproj manifest found.")
    dotnet = shutil.which("dotnet")
    if not dotnet:
        raise ValueError("dotnet was not found on PATH. Install the .NET SDK, then retry.")
    relative_manifest = str(selected.relative_to(root))
    return [dotnet, "restore", relative_manifest], root, relative_manifest, "project NuGet restore"
