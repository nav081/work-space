# Threadline

Local-first visual workflow builder for configurable LangGraph agents.

## Development

Requirements: Node.js 20.19+ (or 22.12+) and Python 3.11+.

From the `desktop` directory:

```powershell
npm install
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

Open **Settings** from the gear in the top bar to add Azure model deployments. Set each model's display name, model ID, deployment name, endpoint, optional API version, and API key. Keys are stored in Windows Credential Manager; endpoints and model metadata are stored in `%LOCALAPPDATA%\Threadline\models.json`. Do not share that per-user settings file as a way to share credentials; each person configures their own Azure access.

Existing `backend/.env` settings are imported into Windows Credential Manager and the per-user model settings on first launch. The `.env` file is left untouched so you can remove old credentials after verifying the import.

Start the API and UI in separate terminals:

```powershell
npm run dev:backend
npm run dev
```

Open http://127.0.0.1:5173. Workflow configuration is saved in local browser storage. Select a project directory, enter a build request, and run the graph. The local API streams LangGraph agent activity to the UI.

## Agent Tools

Agents only receive tools enabled in their configuration. Project file reads, searches, and writes resolve paths inside the selected project directory. **Run commands** uses an executable and subcommand allowlist. The separate **Run any command** permission accepts any executable and arguments, without a shell, and runs it from the selected project directory. It is not an operating-system sandbox; enabled processes can still access files and services available to Threadline. Only enable it for projects you trust.

The Test engineer can use **Install dependencies** for Python, Node.js, and .NET projects. Python manifests install into the selected project's `.venv`; `pytest` then uses that interpreter. Node projects use npm/pnpm/yarn based on their lockfile (including React and Angular projects). .NET projects use `dotnet restore` against the selected solution or project. In monorepos, the agent can specify a manifest path relative to the project root. This permission is separate from **Run commands** and can be disabled per agent. Package install/build hooks may execute, so only enable it for projects you trust.

## Backend Structure

Backend code is organized under `backend/threadline_backend`: `domain` owns validated workflow contracts and routing policies; `application` coordinates agent execution, graph compilation, and run streaming; `infrastructure` owns filesystem/command tools, model providers, and external adapters; `presentation` serializes SSE and run-history responses; `core/constants.py` owns shared limits and defaults. `main.py` is the FastAPI composition root. The top-level `workflow.py`, `run_store.py`, and `settings_store.py` remain stable entry points for the desktop service and existing integrations.

Tester success is evidence-gated: every executed test command must exit successfully. After a Developer handoff, if Tester responds without issuing a new test command, Threadline automatically replays that Tester agent's previous command(s) in the same project environment before routing.

Connections are directed from source agent to target agent and display an arrowhead. Every routed agent receives the connection message, source agent, decision, and full previous result as handoff context. Select an agent and configure **Outcome routing** to choose its outgoing wire for success and for failure; the selected wire's message is passed to its target. Set **Maximum retries** per agent from 0 to 20 (default 5). Configured non-Tester agents must finish with `[AGENT_RESULT: SUCCESS]` or `[AGENT_RESULT: FAILURE]`; a failed command forces the failure route. Testers still require verified passing test commands, and Reviewers still use `[DECISION: REVISE]` or `[DECISION: APPROVED]`. Unconfigured Testers route using their labeled wires and role fallback.

Run backend checks with:

```powershell
backend/.venv/Scripts/python.exe -m pytest backend/test_workflow.py -q
```