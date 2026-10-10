"""Project command adapters with bounded and opt-in unrestricted policies."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool

from threadline_backend.core.constants import (
    ALLOWED_CARGO_OPERATIONS,
    ALLOWED_DOTNET_OPERATIONS,
    ALLOWED_EXECUTABLES,
    ALLOWED_GIT_OPERATIONS,
    ALLOWED_PACKAGE_SCRIPTS,
    COMMAND_ERROR_OUTPUT_LIMIT,
    PROJECT_COMMAND_OUTPUT_LIMIT,
    PROJECT_COMMAND_TIMEOUT_SECONDS,
    PROJECT_TEST_TIMEOUT_SECONDS,
    PROJECT_TOOL_ARGUMENT_LENGTH_LIMIT,
    PROJECT_TOOL_ARGUMENT_LIMIT,
    UNRESTRICTED_ARGUMENT_LENGTH_LIMIT,
    UNRESTRICTED_ARGUMENT_LIMIT,
)

EXIT_CODE_PATTERN = re.compile(r"(?:^|\n)Exit code: (-?\d+)(?:\n|$)")
PYTHON_TEST_ARGS = ["-m", "pytest"]


def project_python(root: Path) -> Path | None:
    """Find a Python interpreter contained by the selected project's venv."""
    candidates = (
        root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


def is_test_command(arguments: list[str]) -> bool:
    """Recognize supported Python, Node, .NET, Cargo, and Go test commands."""
    if not arguments:
        return False
    executable = Path(arguments[0]).name.casefold()
    operation = arguments[1].casefold() if len(arguments) > 1 else ""
    if executable in {"python", "python3"}:
        return arguments[1:3] == PYTHON_TEST_ARGS
    if executable == "pytest":
        return True
    if executable in {"npm", "pnpm", "yarn"}:
        script = arguments[2].casefold() if operation == "run" and len(arguments) > 2 else operation
        return script == "test" and operation in {"test", "run"}
    return executable in {"dotnet", "cargo", "go"} and operation == "test"


def test_exit_code(result: str) -> int | None:
    """Extract a process exit code from a command-tool result."""
    match = EXIT_CODE_PATTERN.search(result)
    return int(match.group(1)) if match else None


def launch_command(command: list[str]) -> list[str]:
    """Launch Windows batch files through cmd while preserving argument quoting."""
    if os.name == "nt" and Path(command[0]).suffix.casefold() in {".cmd", ".bat"}:
        command_line = subprocess.list2cmdline(command)
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", command_line]
    return command


def build_command_tools(root: Path, enabled: set[str]) -> list[StructuredTool]:
    """Build the command tools granted to one workflow agent."""
    tools: list[StructuredTool] = []
    if "Run commands" in enabled:
        def run_command(arguments: list[str]) -> str:
            """Run a restricted executable or project check without a shell."""
            validation = _allowlisted_command(arguments, root)
            if isinstance(validation, str):
                return validation
            timeout = PROJECT_TEST_TIMEOUT_SECONDS if is_test_command(arguments) else PROJECT_COMMAND_TIMEOUT_SECONDS
            return _run_process(validation, root, timeout)

        tools.append(StructuredTool.from_function(run_command, name="run_project_command"))
    if "Run any command" in enabled:
        def run_any_command(arguments: list[str]) -> str:
            """Run any executable in the selected project without a shell."""
            if not _bounded_arguments(arguments, UNRESTRICTED_ARGUMENT_LIMIT, UNRESTRICTED_ARGUMENT_LENGTH_LIMIT):
                return f"Command rejected: provide between 1 and {UNRESTRICTED_ARGUMENT_LIMIT} bounded arguments."
            return _run_process(launch_command(arguments), root, PROJECT_TEST_TIMEOUT_SECONDS)

        tools.append(StructuredTool.from_function(run_any_command, name="run_any_project_command"))
    return tools


def _allowlisted_command(arguments: list[str], root: Path) -> list[str] | str:
    if not _bounded_arguments(arguments, PROJECT_TOOL_ARGUMENT_LIMIT, PROJECT_TOOL_ARGUMENT_LENGTH_LIMIT):
        return f"Command rejected: provide between 1 and {PROJECT_TOOL_ARGUMENT_LIMIT} bounded arguments."
    executable = Path(arguments[0]).name.casefold()
    if executable not in ALLOWED_EXECUTABLES:
        return f"Command rejected: {executable} is not allowlisted."
    operation = arguments[1].casefold() if len(arguments) > 1 else ""
    project_interpreter = str(project_python(root) or Path(sys.executable))
    if executable in {"python", "python3"}:
        if arguments[1:3] != PYTHON_TEST_ARGS:
            return "Command rejected: Python is restricted to `python -m pytest`."
        return [project_interpreter, *arguments[1:]]
    if executable == "pytest":
        return [project_interpreter, *PYTHON_TEST_ARGS, *arguments[1:]]
    if executable == "git" and operation not in ALLOWED_GIT_OPERATIONS:
        return "Command rejected: Git is restricted to status, diff, log, and show."
    if executable in {"npm", "pnpm", "yarn"} and not _allowed_package_operation(arguments, operation):
        return "Command rejected: package managers are restricted to test, build, lint, and check scripts."
    if executable == "dotnet" and operation not in ALLOWED_DOTNET_OPERATIONS:
        return "Command rejected: dotnet is restricted to test and build."
    if executable == "cargo" and operation not in ALLOWED_CARGO_OPERATIONS:
        return "Command rejected: Cargo is restricted to test, check, and build."
    if executable == "go" and operation != "test":
        return "Command rejected: Go is restricted to `go test`."
    return launch_command(arguments)


def _allowed_package_operation(arguments: list[str], operation: str) -> bool:
    script = arguments[2].casefold() if operation == "run" and len(arguments) > 2 else operation
    return operation in {"test", "run"} and script in ALLOWED_PACKAGE_SCRIPTS


def _bounded_arguments(arguments: list[str], count_limit: int, length_limit: int) -> bool:
    return bool(arguments) and len(arguments) <= count_limit and all(len(part) <= length_limit for part in arguments)


def _run_process(command: list[str], root: Path, timeout: int) -> str:
    try:
        result = subprocess.run(command, cwd=root, shell=False, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Command failed: {error}"
    output = (result.stdout + result.stderr).strip()
    return f"Exit code: {result.returncode}\n{output[:PROJECT_COMMAND_OUTPUT_LIMIT]}"
