import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from unittest.mock import Mock
import main as api
import workflow as workflow_module
from settings_store import ModelSettingsInput, ModelSettingsStore

from workflow import AgentConfig, EdgeConfig, RunRequest, _project_tools, _review_targets, build_workflow, create_chat_model, get_model_api_key, missing_model_settings, required_model_settings, resolve_project_path, safe_project_file


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