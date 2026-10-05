import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
import main as api
import workflow as workflow_module

from workflow import AgentConfig, EdgeConfig, RunRequest, MODEL_CATALOG, _project_tools, _review_targets, build_workflow, create_chat_model, get_model_api_key, missing_model_settings, required_model_settings, resolve_project_path, safe_project_file


def make_agent(agent_id: str, kind: str = "developer", model: str = "azure-gpt-4o") -> AgentConfig:
    return AgentConfig(id=agent_id, kind=kind, name=agent_id, instruction="Do assigned work", model=model)


def test_reviewer_routes_revision_and_approval() -> None:
    reviewer = make_agent("critic", "critic")
    links = [
        EdgeConfig(source="critic", target="developer", label="changes requested"),
        EdgeConfig(source="critic", target="tester", label="approved"),
    ]
    revise_state = {"decisions": {"critic": "revise"}, "review_counts": {"critic": 1}}
    approved_state = {"decisions": {"critic": "approved"}, "review_counts": {"critic": 1}}

    assert _review_targets(reviewer, links, revise_state) == ["developer"]
    assert _review_targets(reviewer, links, approved_state) == ["tester"]


def test_review_loop_stops_after_five_revisions() -> None:
    reviewer = make_agent("critic", "critic")
    links = [EdgeConfig(source="critic", target="developer", label="changes requested")]

    with pytest.raises(RuntimeError, match="five-review safety limit"):
        _review_targets(reviewer, links, {"decisions": {"critic": "revise"}, "review_counts": {"critic": 5}})


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


def test_graph_executes_revision_loop_then_continues(tmp_path: Path) -> None:
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

    result = asyncio.run(graph.ainvoke({"requirement": request.requirement, "project_path": str(tmp_path), "messages": [], "decisions": {}, "review_counts": {}}))

    assert fake_model.reviews == 2
    assert [message["agent_id"] for message in result["messages"]].count("developer") == 2
    assert [message["agent_id"] for message in result["messages"]][-1] == "tester"
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    assert sum(event["type"] == "agent_started" for event in events) == 6
    assert all(event["phase"] == "model" for event in events if event["type"] == "agent_progress")


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


def test_model_catalog_requires_only_selected_azure_settings(monkeypatch) -> None:
    settings = {
        "AZURE_OPENAI_API_KEY": "azure-test-key",
        "AZURE_OPENAI_ENDPOINT": "https://unit-test.openai.azure.com/",
    }
    monkeypatch.setattr(workflow_module, "get_setting", lambda name: settings.get(name))
    azure_agent = make_agent("developer")

    assert {model["id"] for model in MODEL_CATALOG} == {"azure-gpt-4o", "azure-gpt-4.1", "azure-gpt-4.1-mini"}
    assert required_model_settings([azure_agent]) == {
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_GPT_4O_DEPLOYMENT",
    }
    assert get_model_api_key(azure_agent.model) == "azure-test-key"


def test_missing_settings_report_only_selected_azure_deployment(monkeypatch) -> None:
    monkeypatch.setattr(workflow_module, "get_setting", lambda _name: None)

    assert missing_model_settings([make_agent("developer")]) == [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_GPT_4O_DEPLOYMENT",
    ]


def test_dotenv_lookup_reads_the_selected_provider_key_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(workflow_module, "BACKEND_DIR", tmp_path)
    (tmp_path / ".env").write_text('AZURE_OPENAI_API_KEY="selected-key"\nUNRELATED_SECRET="unused-key"\n', encoding="utf-8")

    assert get_model_api_key("azure-gpt-4o") == "selected-key"


def test_azure_chat_model_uses_selected_endpoint_and_deployment(monkeypatch) -> None:
    settings = {
        "AZURE_OPENAI_ENDPOINT": "https://unit-test.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION": "2024-10-21",
        "AZURE_OPENAI_GPT_4O_DEPLOYMENT": "project-gpt4o",
    }
    monkeypatch.setattr(workflow_module, "get_setting", lambda name: settings.get(name))

    model = create_chat_model(MODEL_CATALOG[0], "test-key")

    assert model.deployment_name == "project-gpt4o"
    assert str(model.azure_endpoint) == "https://unit-test.openai.azure.com/"
    assert model.openai_api_version == "2024-10-21"
    assert model.model_name == "gpt-4o"


def test_foundry_endpoint_uses_openai_v1_deployment_route(monkeypatch) -> None:
    settings = {
        "AZURE_OPENAI_ENDPOINT": "https://unit-test.services.ai.azure.com/",
        "AZURE_OPENAI_GPT_4_1_MINI_DEPLOYMENT": "mini-prod-deployment",
        "AZURE_OPENAI_API_KEY": "test-key",
    }
    monkeypatch.setattr(workflow_module, "get_setting", lambda name: settings.get(name))

    model = create_chat_model(workflow_module.get_model_config("azure-gpt-4.1-mini"), "test-key")

    assert model.model_name == "mini-prod-deployment"
    assert str(model.openai_api_base) == "https://unit-test.services.ai.azure.com/openai/v1/"
    assert "AZURE_OPENAI_API_VERSION" not in required_model_settings([make_agent("developer", model="azure-gpt-4.1-mini")])


def test_api_catalog_and_key_preflight_only_require_selected_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api, "missing_model_settings", lambda _agents: [])
    monkeypatch.setattr(api, "is_model_deployment_configured", lambda model_id: model_id == "azure-gpt-4o")
    client = TestClient(api.app)
    models_response = client.get("/api/models")
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
    monkeypatch.setattr(api, "missing_model_settings", lambda _agents: ["AZURE_OPENAI_GPT_4O_DEPLOYMENT"])
    missing_key_response = client.post("/api/workflows/run", json=run_request)

    assert {model["id"] for model in models_response.json()} == {"azure-gpt-4o", "azure-gpt-4.1", "azure-gpt-4.1-mini"}
    assert {model["id"] for model in models_response.json() if model["configured"]} == {"azure-gpt-4o"}
    assert run_response.status_code == 200
    assert missing_key_response.status_code == 400
    assert missing_key_response.json()["detail"] == "Configure these Azure OpenAI settings in backend/.env: AZURE_OPENAI_GPT_4O_DEPLOYMENT"