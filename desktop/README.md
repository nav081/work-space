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

Tester success is evidence-gated: every executed test command must exit successfully. After a Developer handoff, if Tester responds without issuing a new test command, Threadline automatically replays that Tester agent's previous command(s) in the same project environment before routing.

The graph supports multiple roots, connected agent steps, and labeled reviewer branches. Every routed agent receives a handoff containing the source agent, connection label, decision, and full previous result; the same handoff is shown in the live console. A reviewer routes to an edge labeled `changes requested`, `revise`, `reject`, or `feedback` when it responds with `[DECISION: REVISE]`; otherwise it follows an `approved`, `continue`, or `pass` edge. A review loop stops after five requested revisions. Test engineer edges labeled `failed` or `error` return to Developer with the failure report; edges labeled `passed`, `success`, or `continue` proceed. For custom labels, a PASS follows an unambiguous normal outgoing wire, and a FAIL prefers a wire directed to Developer. Test retries stop after five failures.

Run backend checks with:

```powershell
backend/.venv/Scripts/python.exe -m pytest backend/test_workflow.py -q
```