import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from fastapi.testclient import TestClient
from unittest.mock import Mock
import main as api
import workflow as workflow_module
from settings_store import ModelSettingsInput, ModelSettingsStore
from run_store import RunStore
from threadline_backend.infrastructure import command_tools, dependency_tools

from workflow import AgentConfig, EdgeConfig, RunRequest, WorkflowState, _agent_result_decision, _project_tools, _review_targets, _routing_targets, _test_targets, build_workflow, create_chat_model, get_model_api_key, missing_model_settings, required_model_settings, resolve_project_path, safe_project_file


def make_settings_store(tmp_path: Path) -> ModelSettingsStore:
    keyring_backend = Mock()
    secrets: dict[tuple[str, str], str] = {}
    keyring_backend.get_password.side_effect = lambda service, username: secrets.get((service, username))
    keyring_backend.set_password.side_effect = lambda service, username, password: secrets.__setitem__((service, username), password)
    keyring_backend.delete_password.side_effect = lambda service, username: secrets.pop((service, username), None)
    return ModelSettingsStore(tmp_path / "user-settings.json", keyring_backend, import_legacy=False)


def configure_gpt4o(store: ModelSettingsStore, *, endpoint: str = "https://unit-test.openai.azure.com/", api_version: str = "2024-10-21", api_key: str = "test-key") -> dict:
    model = store.get_model("azure-gpt-4o")
    return store.save(ModelSettingsInput(
        id=model["id"], name=model["name"], model=model["model"], deployment="project-gpt4o",
        endpoint=endpoint, api_version=api_version, api_key=api_key,
    ))


def make_agent(agent_id: str, kind: str = "developer", model: str = "azure-gpt-4o") -> AgentConfig:
    return AgentConfig(id=agent_id, kind=kind, name=agent_id, instruction="Do assigned work", model=model)


def make_state(*, decisions: dict[str, str] | None = None, review_counts: dict[str, int] | None = None, requirement: str = "Test requirement", project_path: str = ".") -> WorkflowState:
    return {"requirement": requirement, "project_path": project_path, "messages": [], "decisions": decisions or {}, "review_counts": review_counts or {}, "handoffs": {}, "test_commands": {}}


def test_reviewer_routes_revision_and_approval() -> None:
    reviewer = make_agent("critic", "critic")
    links = [
        EdgeConfig(source="critic", target="developer", label="changes requested"),
        EdgeConfig(source="critic", target="tester", label="approved"),
    ]
    revise_state = make_state(decisions={"critic": "revise"}, review_counts={"critic": 1})
    approved_state = make_state(decisions={"critic": "approved"}, review_counts={"critic": 1})

    assert _review_targets(reviewer, links, revise_state) == ["developer"]
    assert _review_targets(reviewer, links, approved_state) == ["tester"]


def test_review_loop_stops_after_configured_retry_limit() -> None:
    reviewer = make_agent("critic", "critic")
    links = [EdgeConfig(source="critic", target="developer", label="changes requested")]

    assert _review_targets(reviewer, links, make_state(decisions={"critic": "revise"}, review_counts={"critic": 5})) == ["developer"]
    with pytest.raises(RuntimeError, match=r"configured retry limit \(5\)"):
        _review_targets(reviewer, links, make_state(decisions={"critic": "revise"}, review_counts={"critic": 6}))


def test_agent_retry_limit_is_configurable() -> None:
    reviewer = make_agent("critic", "critic").model_copy(update={"max_retries": 2})
    links = [EdgeConfig(source="critic", target="developer", label="changes requested")]

    assert _review_targets(reviewer, links, make_state(decisions={"critic": "revise"}, review_counts={"critic": 2})) == ["developer"]
    with pytest.raises(RuntimeError, match=r"configured retry limit \(2\)"):
        _review_targets(reviewer, links, make_state(decisions={"critic": "revise"}, review_counts={"critic": 3}))


def test_tester_routes_failure_to_developer_and_pass_to_next() -> None:
    tester = make_agent("tester", "tester")
    links = [
        EdgeConfig(source="tester", target="developer", label="Test failed - fix and rerun"),
        EdgeConfig(source="tester", target="summary", label="Test suite passed"),
    ]

    assert _test_targets(tester, links, make_state(decisions={"tester": "fail"}, review_counts={"tester": 1})) == ["developer"]
    assert _test_targets(tester, links, make_state(decisions={"tester": "pass"}, review_counts={"tester": 1})) == ["summary"]


def test_tester_uses_wire_direction_when_custom_label_has_no_route_keyword() -> None:
    tester = make_agent("tester", "tester")
    custom_next = [EdgeConfig(source="tester", target="summary", label="QA completed - hand off findings")]
    developer_return = [EdgeConfig(source="tester", target="developer", label="Please investigate this report")]

    assert _test_targets(tester, custom_next, make_state(decisions={"tester": "pass"}), {"summary": "summary"}) == ["summary"]
    assert _test_targets(tester, developer_return, make_state(decisions={"tester": "fail"}, review_counts={"tester": 1}), {"developer": "developer"}) == ["developer"]


def test_tester_uses_target_roles_to_route_custom_fail_and_pass_wires() -> None:
    tester = make_agent("tester", "tester")
    outgoing = [
        EdgeConfig(source="tester", target="developer", label="Review this report and fix the issue"),
        EdgeConfig(source="tester", target="summary", label="Continue with release notes"),
    ]
    target_kinds = {"developer": "developer", "summary": "summary"}

    assert _test_targets(tester, outgoing, make_state(decisions={"tester": "fail"}, review_counts={"tester": 1}), target_kinds) == ["developer"]
    assert _test_targets(tester, outgoing, make_state(decisions={"tester": "pass"}, review_counts={"tester": 1}), target_kinds) == ["summary"]


def test_tester_stops_after_retry_limit_and_ends_on_unrouted_pass() -> None:
    tester = make_agent("tester", "tester")
    failure_edge = [EdgeConfig(source="tester", target="developer", label="Failed: return to developer")]

    assert _test_targets(tester, failure_edge, make_state(decisions={"tester": "fail"}, review_counts={"tester": 5})) == ["developer"]
    with pytest.raises(RuntimeError, match=r"configured retry limit \(5\)"):
        _test_targets(tester, failure_edge, make_state(decisions={"tester": "fail"}, review_counts={"tester": 6}))
    assert _test_targets(tester, failure_edge, make_state(decisions={"tester": "pass"}, review_counts={"tester": 1})) == "__end__"


def test_configured_agent_outcomes_follow_only_the_selected_wire() -> None:
    developer = make_agent("developer").model_copy(update={"success_edge_id": "dev-test", "failure_edge_id": "dev-retry"})
    outgoing = [
        EdgeConfig(id="dev-test", source="developer", target="tester", label="Run tests"),
        EdgeConfig(id="dev-retry", source="developer", target="retry", label="Investigate failure"),
    ]

    assert _routing_targets(developer, outgoing, "pass", 0, {}) == ["tester"]
    assert _routing_targets(developer, outgoing, "fail", 0, {}) == ["retry"]


def test_configured_agent_retry_limit_controls_failure_wire() -> None:
    developer = make_agent("developer").model_copy(update={"success_edge_id": "dev-test", "failure_edge_id": "dev-retry", "max_retries": 1})
    outgoing = [
        EdgeConfig(id="dev-test", source="developer", target="tester"),
        EdgeConfig(id="dev-retry", source="developer", target="retry"),
    ]

    assert _routing_targets(developer, outgoing, "fail", 1, {}) == ["retry"]
    with pytest.raises(RuntimeError, match=r"configured retry limit \(1\)"):
        _routing_targets(developer, outgoing, "fail", 2, {})


def test_agent_result_markers_and_failed_commands_select_failure() -> None:
    assert _agent_result_decision("Changes complete. [AGENT_RESULT: SUCCESS]", False) == "pass"
    assert _agent_result_decision("Could not complete. [AGENT_RESULT: FAILURE]", False) == "fail"
    assert _agent_result_decision("Changes complete. [AGENT_RESULT: SUCCESS]", True) == "fail"
    assert _agent_result_decision("No outcome marker", False) == "fail"


def test_configured_generic_failure_loop_stops_after_five_failures() -> None:
    developer = make_agent("developer").model_copy(update={"success_edge_id": "dev-test", "failure_edge_id": "dev-retry"})
    outgoing = [
        EdgeConfig(id="dev-test", source="developer", target="tester"),
        EdgeConfig(id="dev-retry", source="developer", target="retry"),
    ]

    assert _routing_targets(developer, outgoing, "fail", 5, {}) == ["retry"]
    with pytest.raises(RuntimeError, match=r"configured retry limit \(5\)"):
        _routing_targets(developer, outgoing, "fail", 6, {})


def test_compiled_graph_accepts_configured_developer_retry_routes(tmp_path: Path) -> None:
    developer = make_agent("developer").model_copy(update={"success_edge_id": "dev-test", "failure_edge_id": "dev-retry"})
    retry = make_agent("retry").model_copy(update={"success_edge_id": "retry-test", "failure_edge_id": "retry-dev"})
    request = RunRequest(
        project_path=str(tmp_path),
        requirement="Implement and verify a change",
        agents=[make_agent("analyst", "understand"), developer, retry, make_agent("tester", "tester")],
        edges=[
            EdgeConfig(id="analyst-dev", source="analyst", target="developer", label="Implementation brief"),
            EdgeConfig(id="dev-test", source="developer", target="tester", label="Run tests"),
            EdgeConfig(id="dev-retry", source="developer", target="retry", label="Investigate failure"),
            EdgeConfig(id="retry-test", source="retry", target="tester", label="Retest after fix"),
            EdgeConfig(id="retry-dev", source="retry", target="developer", label="Try another fix"),
        ],
    )

    assert build_workflow(request, tmp_path, asyncio.Queue()) is not None


def test_configured_outcome_wire_must_be_outgoing_from_its_agent(tmp_path: Path) -> None:
    developer = make_agent("developer").model_copy(update={"success_edge_id": "not-connected"})
    request = RunRequest(
        project_path=str(tmp_path),
        requirement="Implement a change",
        agents=[developer, make_agent("tester", "tester")],
        edges=[EdgeConfig(id="dev-test", source="developer", target="tester")],
    )

    with pytest.raises(ValueError, match="not connected from that agent"):
        build_workflow(request, tmp_path, asyncio.Queue())


def test_configured_tester_outcomes_follow_selected_wires_and_keep_retry_limit() -> None:
    tester = make_agent("retry", "tester").model_copy(update={"success_edge_id": "retry-test", "failure_edge_id": "retry-dev"})
    outgoing = [
        EdgeConfig(id="retry-test", source="retry", target="tester", label="Retest"),
        EdgeConfig(id="retry-dev", source="retry", target="developer", label="Fix again"),
    ]

    assert _routing_targets(tester, outgoing, "pass", 0, {}) == ["tester"]
    assert _routing_targets(tester, outgoing, "fail", 1, {}) == ["developer"]
    assert _routing_targets(tester, outgoing, "fail", 5, {}) == ["developer"]
    with pytest.raises(RuntimeError, match=r"configured retry limit \(5\)"):
        _routing_targets(tester, outgoing, "fail", 6, {})


def test_compiled_graph_accepts_review_cycle(tmp_path: Path) -> None:
    request = RunRequest(
        project_path=str(tmp_path),
        requirement="Add a feature",
        agents=[make_agent("analyst"), make_agent("developer"), make_agent("critic", "critic"), make_agent("tester")],
        edges=[
            EdgeConfig(source="analyst", target="developer"),
            EdgeConfig(source="developer", target="critic"),
            EdgeConfig(source="critic", target="developer", label="changes requested"),
            EdgeConfig(source="critic", target="tester", label="approved"),
        ],
    )

    graph = build_workflow(request, tmp_path, asyncio.Queue())

    assert graph is not None


def test_graph_executes_revision_loop_then_continues(tmp_path: Path, monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.reviews = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages):
            system = messages[0].content
            if "role: critic" in system:
                self.reviews += 1
                decision = "REVISE" if self.reviews == 1 else "APPROVED"
                return AIMessage(content=f"Review complete. [DECISION: {decision}]")
            return AIMessage(content="Implementation updated.")

    fake_model = FakeModel()
    test_settings = make_settings_store(tmp_path)
    configure_gpt4o(test_settings)
    monkeypatch.setattr(workflow_module, "model_settings", test_settings)
    request = RunRequest(
        project_path=str(tmp_path),
        requirement="Implement a small feature",
        agents=[make_agent("analyst"), make_agent("developer"), make_agent("critic", "critic"), make_agent("tester")],
        edges=[
            EdgeConfig(source="analyst", target="developer"),
            EdgeConfig(source="developer", target="critic"),
            EdgeConfig(source="critic", target="developer", label="changes requested"),
            EdgeConfig(source="critic", target="tester", label="approved"),
        ],
    )
    event_queue: asyncio.Queue[dict] = asyncio.Queue()
    graph = build_workflow(
        request,
        tmp_path,
        event_queue,
        model_factory=lambda _config, _api_key: fake_model,
        api_key_resolver=lambda _model_id: "test-key",
    )

    result = asyncio.run(graph.ainvoke(make_state(requirement=request.requirement, project_path=str(tmp_path))))

    assert fake_model.reviews == 2
    assert [message["agent_id"] for message in result["messages"]].count("developer") == 2
    assert [message["agent_id"] for message in result["messages"]][-1] == "tester"
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    assert sum(event["type"] == "agent_started" for event in events) == 6
    assert all(event["phase"] == "model" for event in events if event["type"] == "agent_progress")


def test_graph_reruns_failed_test_after_developer_fix(tmp_path: Path, monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.tester_calls = 0
            self.developer_runs = 0
            self.developer_prompts: list[str] = []

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages):
            system = messages[0].content
            if "role: test" in system:
                self.tester_calls += 1
                if any(isinstance(message, ToolMessage) for message in messages):
                    result = "PASS" if "Exit code: 0" in str(messages[-1].content) else "FAIL"
                    return AIMessage(content=f"Project pytest result: {result}. [TEST_RESULT: {result}]")
                if self.tester_calls > 1:
                    return AIMessage(content="The fix looks good. [TEST_RESULT: PASS]")
                return AIMessage(content="", tool_calls=[{"name": "run_project_command", "args": {"arguments": ["pytest", "tests", "-q"]}, "id": f"test-call-{self.developer_runs}"}])
            if "role: developer" in system:
                self.developer_runs += 1
                self.developer_prompts.append(messages[-1].content)
                return AIMessage(content="Applied test fix.")
            return AIMessage(content="Workflow step complete.")

    store = make_settings_store(tmp_path)
    configure_gpt4o(store)
    monkeypatch.setattr(workflow_module, "model_settings", store)
    fake_model = FakeModel()
    test_command_runs = 0

    def fake_pytest(command, **_kwargs):
        nonlocal test_command_runs
        test_command_runs += 1
        if test_command_runs == 1:
            return subprocess.CompletedProcess(command, 1, stdout="ImportError: No module named PySide6", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="All focused and full-suite tests passed", stderr="")

    monkeypatch.setattr(command_tools.subprocess, "run", fake_pytest)
    request = RunRequest(
        project_path=str(tmp_path),
        requirement="Run a focused test, fix failures, then rerun the full suite",
        agents=[make_agent("analyst", "understand"), make_agent("developer"), AgentConfig(id="tester", kind="tester", name="tester", instruction="Run tests", model="azure-gpt-4o", tools=["Run commands"]), make_agent("summary", "summary")],
        edges=[
            EdgeConfig(source="analyst", target="developer", label="Implementation plan from analyst"),
            EdgeConfig(source="developer", target="tester"),
            EdgeConfig(source="tester", target="developer", label="Failed PySide6 test - fix missing dependency and rerun full suite"),
            EdgeConfig(source="tester", target="summary", label="Full suite passed"),
        ],
    )
    graph = build_workflow(request, tmp_path, asyncio.Queue(), model_factory=lambda _config, _key: fake_model, api_key_resolver=lambda _model_id: "test-key")

    result = asyncio.run(graph.ainvoke(make_state(requirement=request.requirement, project_path=str(tmp_path))))

    assert test_command_runs == 2
    assert fake_model.tester_calls == 3
    assert fake_model.developer_runs == 2
    assert "Implementation plan from analyst" in fake_model.developer_prompts[0]
    assert "Workflow step complete." in fake_model.developer_prompts[0]
    retry_prompt = fake_model.developer_prompts[-1]
    assert "ROUTED HANDOFF" in retry_prompt
    assert "Test failed - fix and rerun" not in retry_prompt
    assert "Failed PySide6 test - fix missing dependency and rerun full suite" in retry_prompt
    assert "ImportError: No module named PySide6" in retry_prompt
    assert [message["agent_id"] for message in result["messages"]][-1] == "summary"


def test_tester_cannot_claim_pass_without_executing_test_command(tmp_path: Path, monkeypatch) -> None:
    class FakeModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="All tests passed. [TEST_RESULT: PASS]")

    store = make_settings_store(tmp_path)
    configure_gpt4o(store)
    monkeypatch.setattr(workflow_module, "model_settings", store)
    tester = AgentConfig(id="tester", kind="tester", name="Tester", instruction="Run tests", model="azure-gpt-4o", tools=[])
    summary = make_agent("summary", "summary")
    request = RunRequest(project_path=str(tmp_path), requirement="Verify feature", agents=[tester, summary], edges=[EdgeConfig(source="tester", target="summary", label="Tests passed")])
    graph = build_workflow(request, tmp_path, asyncio.Queue(), model_factory=lambda _config, _key: FakeModel(), api_key_resolver=lambda _model_id: "test-key")

    with pytest.raises(RuntimeError, match="did not execute a test command"):
        asyncio.run(graph.ainvoke(make_state(requirement=request.requirement, project_path=str(tmp_path))))


def test_failed_workflow_status_is_saved_before_error_is_streamed(tmp_path: Path, monkeypatch) -> None:
    class FailingGraph:
        async def astream(self, _state, **_kwargs):
            raise RuntimeError("PySide6 import failure")
            yield {}

    store = RunStore(tmp_path / "failed-runs.sqlite3")
    monkeypatch.setattr(api, "run_store", store)
    monkeypatch.setattr(api, "missing_model_settings", lambda _agents: [])
    monkeypatch.setattr(api, "build_workflow", lambda *_args, **_kwargs: FailingGraph())
    client = TestClient(api.app)

    response = client.post("/api/workflows/run", json={
        "project_path": str(tmp_path),
        "requirement": "Run tests",
        "agents": [{"id": "tester", "kind": "tester", "name": "Tester", "instruction": "Run tests", "model": "azure-gpt-4o", "tools": []}],
        "edges": [{"source": "tester", "target": "tester", "label": "Retry"}],
    })
    events = [json.loads(frame[6:]) for frame in response.text.split("\n\n") if frame.startswith("data: ")]
    run_id = next(event["run_id"] for event in events if event["type"] == "run_started")
    history = client.get("/api/runs").json()
    details = client.get(f"/api/runs/{run_id}").json()

    assert response.status_code == 200
    assert history[0]["status"] == "failed"
    assert history[0]["completed_at"] is not None
    assert any(event["type"] == "run_error" and "PySide6 import failure" in event["message"] for event in details["events"])
def test_project_file_access_stays_inside_selected_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-threadline-test.txt"
    with pytest.raises(ValueError, match="inside the selected project"):
        safe_project_file(tmp_path, "../outside-threadline-test.txt")
    assert not outside.exists()


def test_project_path_must_be_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        resolve_project_path(str(file_path))


def test_command_tool_rejects_destructive_git_and_arbitrary_python(tmp_path: Path) -> None:
    command_tool = next(tool for tool in _project_tools(tmp_path, {"Run commands"}) if tool.name == "run_project_command")

    assert "restricted to status, diff, log, and show" in command_tool.invoke({"arguments": ["git", "reset", "--hard"]})
    assert "restricted to `python -m pytest`" in command_tool.invoke({"arguments": ["python", "-c", "print('unsafe')"]})


def test_run_any_command_permission_allows_unlisted_executable(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command, *, cwd, shell, **_kwargs):
        calls.append((command, cwd, shell))
        return subprocess.CompletedProcess(command, 0, stdout="Runner completed", stderr="")

    monkeypatch.setattr(command_tools.subprocess, "run", fake_run)
    unrestricted_tool = next(tool for tool in _project_tools(tmp_path, {"Run any command"}) if tool.name == "run_any_project_command")
    restricted_tools = _project_tools(tmp_path, {"Run commands"})

    result = unrestricted_tool.invoke({"arguments": ["run", "--project", "supaHelper"]})

    assert "Exit code: 0" in result
    assert calls == [(["run", "--project", "supaHelper"], tmp_path, False)]
    assert all(tool.name != "run_any_project_command" for tool in restricted_tools)


def test_dependency_installer_uses_project_venv_and_allowed_manifest(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "requirements.txt").write_text("PySide6>=6.7,<7\n", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, *, cwd, **_kwargs):
        calls.append((command, cwd))
        if command[1:3] == ["-m", "venv"]:
            python_path = Path(command[-1]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python_path.parent.mkdir(parents=True)
            python_path.write_text("test interpreter", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="Dependencies installed", stderr="")

    monkeypatch.setattr(dependency_tools.subprocess, "run", fake_run)
    installer = next(tool for tool in _project_tools(tmp_path, {"Install dependencies"}) if tool.name == "install_project_dependencies")

    result = installer.invoke({"manifest_name": "requirements.txt"})

    project_python = tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert "installed into project .venv" in result
    assert calls[0][0] == [sys.executable, "-m", "venv", str(tmp_path / ".venv")]
    assert calls[1][0] == [str(project_python.resolve()), "-m", "pip", "install", "-r", "requirements.txt"]
    assert all(cwd == tmp_path for _, cwd in calls)


def test_dependency_installer_rejects_unlisted_manifest_and_is_opt_in(tmp_path: Path) -> None:
    tools_without_permission = _project_tools(tmp_path, {"Run commands"})
    assert all(tool.name != "install_project_dependencies" for tool in tools_without_permission)

    installer = next(tool for tool in _project_tools(tmp_path, {"Install dependencies"}) if tool.name == "install_project_dependencies")
    result = installer.invoke({"manifest_name": "../../outside.txt"})

    assert "Install rejected" in result


def test_node_dependency_installer_uses_locked_npm_ci(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "package.json").write_text('{"name":"sample-ui"}', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependency_tools.shutil, "which", lambda name: f"C:/tools/{name}.cmd")
    monkeypatch.setattr(dependency_tools.subprocess, "run", lambda command, **kwargs: (calls.append((command, kwargs["cwd"])) or subprocess.CompletedProcess(command, 0, stdout="added packages", stderr="")))
    installer = next(tool for tool in _project_tools(tmp_path, {"Install dependencies"}) if tool.name == "install_project_dependencies")

    result = installer.invoke({"ecosystem": "node"})

    assert "project node_modules" in result
    assert calls[0][1] == tmp_path
    if os.name == "nt":
        assert calls[0][0][0].endswith("cmd.exe")
        assert "npm.cmd ci" in calls[0][0][-1]
    else:
        assert calls[0][0] == ["C:/tools/npm.cmd", "ci"]


@pytest.mark.parametrize(("lockfile", "manager", "arguments"), [
    ("pnpm-lock.yaml", "pnpm", ["install", "--frozen-lockfile"]),
    ("yarn.lock", "yarn", ["install", "--frozen-lockfile"]),
])
def test_node_installer_selects_lockfile_package_manager(tmp_path: Path, monkeypatch, lockfile: str, manager: str, arguments: list[str]) -> None:
    (tmp_path / "package.json").write_text('{"name":"frontend"}', encoding="utf-8")
    (tmp_path / lockfile).write_text("lock", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependency_tools.shutil, "which", lambda name: f"C:/tools/{name}.cmd")
    monkeypatch.setattr(dependency_tools.subprocess, "run", lambda command, **kwargs: (calls.append(command) or subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")))
    installer = next(tool for tool in _project_tools(tmp_path, {"Install dependencies"}) if tool.name == "install_project_dependencies")

    result = installer.invoke({"ecosystem": "node"})

    assert "project node_modules" in result
    if os.name == "nt":
        assert manager in calls[0][-1]
        assert " ".join(arguments) in calls[0][-1]
    else:
        assert calls[0] == [f"C:/tools/{manager}.cmd", *arguments]


def test_dotnet_dependency_installer_restores_selected_solution(tmp_path: Path, monkeypatch) -> None:
    solution = tmp_path / "App.sln"
    solution.write_text("solution", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependency_tools.shutil, "which", lambda name: "C:/dotnet/dotnet.exe" if name == "dotnet" else None)
    monkeypatch.setattr(dependency_tools.subprocess, "run", lambda command, **kwargs: (calls.append((command, kwargs["cwd"])) or subprocess.CompletedProcess(command, 0, stdout="restore complete", stderr="")))
    installer = next(tool for tool in _project_tools(tmp_path, {"Install dependencies"}) if tool.name == "install_project_dependencies")

    result = installer.invoke({"ecosystem": "dotnet", "manifest_name": "App.sln"})

    assert "project NuGet restore" in result
    assert calls == [(["C:/dotnet/dotnet.exe", "restore", "App.sln"], tmp_path)]


def test_dotnet_installer_auto_discovers_single_project(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "Service.csproj").write_text("<Project />", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependency_tools.shutil, "which", lambda name: "C:/dotnet/dotnet.exe" if name == "dotnet" else None)
    monkeypatch.setattr(dependency_tools.subprocess, "run", lambda command, **kwargs: (calls.append(command) or subprocess.CompletedProcess(command, 0, stdout="restored", stderr="")))
    installer = next(tool for tool in _project_tools(tmp_path, {"Install dependencies"}) if tool.name == "install_project_dependencies")

    result = installer.invoke({"ecosystem": "dotnet"})

    assert "project NuGet restore" in result
    assert calls == [["C:/dotnet/dotnet.exe", "restore", "Service.csproj"]]


def test_pytest_command_uses_project_virtualenv(tmp_path: Path, monkeypatch) -> None:
    python_path = tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python_path.parent.mkdir(parents=True)
    python_path.write_text("project interpreter", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, **_kwargs):
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, stdout="2 passed", stderr="")

    monkeypatch.setattr(command_tools.subprocess, "run", fake_run)
    command_tool = next(tool for tool in _project_tools(tmp_path, {"Run commands"}) if tool.name == "run_project_command")

    result = command_tool.invoke({"arguments": ["pytest", "tests/test_gui.py", "-q"]})

    assert "Exit code: 0" in result
    assert calls[0][0] == [str(python_path.resolve()), "-m", "pytest", "tests/test_gui.py", "-q"]
    assert calls[0][1] == tmp_path


def test_model_catalog_requires_only_selected_azure_settings(monkeypatch, tmp_path: Path) -> None:
    store = make_settings_store(tmp_path)
    configure_gpt4o(store, api_key="azure-test-key")
    monkeypatch.setattr(workflow_module, "model_settings", store)
    azure_agent = make_agent("developer")

    assert {model["id"] for model in store.list_models()} == {"azure-gpt-4o", "azure-gpt-4.1", "azure-gpt-4.1-mini"}
    assert required_model_settings([azure_agent]) == {"azure-gpt-4o"}
    assert get_model_api_key(azure_agent.model) == "azure-test-key"


def test_missing_settings_report_only_selected_azure_deployment(monkeypatch, tmp_path: Path) -> None:
    store = make_settings_store(tmp_path)
    monkeypatch.setattr(workflow_module, "model_settings", store)

    assert missing_model_settings([make_agent("developer")]) == [
        "Azure OpenAI - GPT-4o: API key",
        "Azure OpenAI - GPT-4o: API version",
        "Azure OpenAI - GPT-4o: deployment name",
        "Azure OpenAI - GPT-4o: endpoint",
    ]


def test_model_settings_keep_secrets_out_of_user_json(tmp_path: Path, monkeypatch) -> None:
    store = make_settings_store(tmp_path)
    monkeypatch.setattr(workflow_module, "model_settings", store)
    saved = configure_gpt4o(store, api_key="do-not-write-this-to-json")
    store.save(ModelSettingsInput(
        id="azure-gpt-4o", name="Renamed GPT-4o", model="gpt-4o", deployment="project-gpt4o",
        endpoint="https://unit-test.openai.azure.com", api_version="2024-10-21", api_key="",
    ))
    persisted = (tmp_path / "user-settings.json").read_text(encoding="utf-8")

    assert saved["api_key_configured"] is True
    assert "api_key" not in saved
    assert "do-not-write-this-to-json" not in persisted
    assert get_model_api_key("azure-gpt-4o") == "do-not-write-this-to-json"
    assert store.get_model("azure-gpt-4o")["name"] == "Renamed GPT-4o"


def test_model_settings_can_remove_model_without_a_saved_key(tmp_path: Path) -> None:
    store = make_settings_store(tmp_path)
    store.delete("azure-gpt-4o")

    assert {model["id"] for model in store.list_models()} == {"azure-gpt-4.1", "azure-gpt-4.1-mini"}


def test_run_store_persists_full_agent_and_tool_events(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    run_id = store.create_run("sample-project", str(tmp_path), "Run tests", {
        "agents": [{"id": "tester", "name": "Test engineer", "kind": "tester", "instruction": "Run tests"}],
        "edges": [],
    })
    store.append_event(run_id, {"type": "agent_progress", "name": "Test engineer", "message": "Running focused test"})
    store.append_event(run_id, {"type": "tool_completed", "tool": "run_project_command", "detail": "2 passed, 1 failed"})
    store.append_event(run_id, {"type": "agent_completed", "agent_name": "Test engineer", "content": "ImportError: No module named PySide6"})
    store.finish_run(run_id, "failed")

    history = store.list_runs()
    details = store.get_run(run_id)

    assert history[0]["run_id"] == run_id
    assert history[0]["status"] == "failed"
    assert history[0]["agent_count"] == 1
    assert details is not None
    assert [event["type"] for event in details["events"]] == ["agent_progress", "tool_completed", "agent_completed"]
    assert details["events"][-1]["content"] == "ImportError: No module named PySide6"


def test_run_history_api_exposes_list_and_full_detail(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "history.sqlite3")
    run_id = store.create_run("project", str(tmp_path), "Investigate missing dependency", {"agents": [], "edges": []})
    store.append_event(run_id, {"type": "run_error", "message": "No module named PySide6"})
    store.finish_run(run_id, "failed")
    monkeypatch.setattr(api, "run_store", store)
    client = TestClient(api.app)

    listing = client.get("/api/runs")
    detail = client.get(f"/api/runs/{run_id}")
    missing = client.get("/api/runs/unknown-run")

    assert listing.status_code == 200
    assert listing.json()[0]["run_id"] == run_id
    assert detail.status_code == 200
    assert detail.json()["events"][0]["message"] == "No module named PySide6"
    assert missing.status_code == 404


def test_completed_workflow_stream_is_added_to_run_history(tmp_path: Path, monkeypatch) -> None:
    class FakeGraph:
        async def astream(self, _state, **_kwargs):
            yield {"tester": {"messages": [{"agent_id": "tester", "agent_name": "Test engineer", "content": "PySide6 tests passed in project .venv."}]}}

    store = RunStore(tmp_path / "history.sqlite3")
    monkeypatch.setattr(api, "run_store", store)
    monkeypatch.setattr(api, "missing_model_settings", lambda _agents: [])
    monkeypatch.setattr(api, "build_workflow", lambda *_args, **_kwargs: FakeGraph())
    client = TestClient(api.app)

    response = client.post("/api/workflows/run", json={
        "project_path": str(tmp_path),
        "requirement": "Run project-local PySide6 tests",
        "agents": [{"id": "tester", "kind": "tester", "name": "Test engineer", "instruction": "Run test", "model": "azure-gpt-4o", "tools": ["Run commands"]}],
        "edges": [{"source": "tester", "target": "tester", "label": "rerun"}],
    })
    events = [json.loads(frame[6:]) for frame in response.text.split("\n\n") if frame.startswith("data: ")]
    run_id = next(event["run_id"] for event in events if event["type"] == "run_started")
    history = client.get("/api/runs")
    detail = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    assert any(event["type"] == "run_completed" for event in events)
    assert history.json()[0]["run_id"] == run_id
    assert history.json()[0]["status"] == "completed"
    assert any(event.get("content") == "PySide6 tests passed in project .venv." for event in detail.json()["events"])


def test_azure_chat_model_uses_selected_endpoint_and_deployment(monkeypatch, tmp_path: Path) -> None:
    store = make_settings_store(tmp_path)
    configure_gpt4o(store)
    monkeypatch.setattr(workflow_module, "model_settings", store)
    model = create_chat_model(store.get_model("azure-gpt-4o"), "test-key")

    assert model.deployment_name == "project-gpt4o"
    assert str(model.azure_endpoint) == "https://unit-test.openai.azure.com"
    assert model.openai_api_version == "2024-10-21"
    assert model.model_name == "gpt-4o"


def test_foundry_endpoint_uses_openai_v1_deployment_route(monkeypatch, tmp_path: Path) -> None:
    store = make_settings_store(tmp_path)
    store.save(ModelSettingsInput(id="azure-gpt-4.1-mini", name="Mini", model="gpt-4.1-mini", deployment="mini-prod-deployment", endpoint="https://unit-test.services.ai.azure.com", api_key="test-key"))
    monkeypatch.setattr(workflow_module, "model_settings", store)

    model = create_chat_model(store.get_model("azure-gpt-4.1-mini"), "test-key")

    assert model.model_name == "mini-prod-deployment"
    assert str(model.openai_api_base) == "https://unit-test.services.ai.azure.com/openai/v1/"
    assert "AZURE_OPENAI_API_VERSION" not in required_model_settings([make_agent("developer", model="azure-gpt-4.1-mini")])


def test_api_catalog_and_key_preflight_only_require_selected_provider(tmp_path: Path, monkeypatch) -> None:
    store = make_settings_store(tmp_path)
    configure_gpt4o(store)
    monkeypatch.setattr(api, "model_settings", store)
    monkeypatch.setattr(api, "missing_model_settings", lambda _agents: [])
    client = TestClient(api.app)
    models_response = client.get("/api/models")
    settings_response = client.get("/api/settings/models")
    saved_model_response = client.post("/api/settings/models", json={
        "name": "New mini deployment",
        "model": "gpt-4.1-mini",
        "deployment": "team-mini",
        "endpoint": "https://team.services.ai.azure.com",
        "api_key": "only-in-keyring",
    })
    run_request = {
        "project_path": str(tmp_path),
        "requirement": "Test selected provider",
        "agents": [
            {"id": "analyst", "kind": "understand", "name": "Analyst", "instruction": "Analyze", "model": "azure-gpt-4o", "tools": []},
            {"id": "developer", "kind": "developer", "name": "Developer", "instruction": "Build", "model": "azure-gpt-4o", "tools": []},
        ],
        "edges": [{"source": "analyst", "target": "developer"}],
    }
    run_response = client.post("/api/workflows/run", json=run_request)
    monkeypatch.setattr(api, "missing_model_settings", lambda _agents: ["Azure OpenAI - GPT-4o: deployment name"])
    missing_key_response = client.post("/api/workflows/run", json=run_request)

    assert {model["id"] for model in models_response.json()} == {"azure-gpt-4o", "azure-gpt-4.1", "azure-gpt-4.1-mini"}
    assert {model["id"] for model in models_response.json() if model["configured"]} == {"azure-gpt-4o"}
    assert all("api_key" not in model for model in settings_response.json())
    assert saved_model_response.status_code == 200
    assert saved_model_response.json()["api_key_configured"] is True
    assert "api_key" not in saved_model_response.json()
    assert "only-in-keyring" not in saved_model_response.text
    assert run_response.status_code == 200
    assert missing_key_response.status_code == 400
    assert missing_key_response.json()["detail"] == "Configure these Azure OpenAI settings in app Settings: Azure OpenAI - GPT-4o: deployment name"


def test_workflow_with_removed_model_returns_client_error(tmp_path: Path, monkeypatch) -> None:
    store = make_settings_store(tmp_path)
    monkeypatch.setattr(api, "model_settings", store)
    client = TestClient(api.app)

    response = client.post("/api/workflows/run", json={
        "project_path": str(tmp_path),
        "requirement": "Invalid model should be a setup error",
        "agents": [{"id": "dev", "kind": "developer", "name": "Developer", "instruction": "Implement", "model": "deleted-model", "tools": []}],
        "edges": [{"source": "dev", "target": "dev"}],
    })

    assert response.status_code == 400
    assert "Unknown model 'deleted-model'" in response.json()["detail"]