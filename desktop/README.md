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

Agents only receive tools enabled in their configuration. Project file reads, searches, and writes resolve paths inside the selected project directory. The optional command tool has an executable and subcommand allowlist, but builds and tests can run project-defined scripts. Only enable it for projects you trust; it is not an operating-system sandbox.

The graph supports multiple roots, connected agent steps, and labeled reviewer branches. A reviewer routes to an edge labeled `changes requested`, `revise`, `reject`, or `feedback` when it responds with `[DECISION: REVISE]`; otherwise it follows an `approved`, `continue`, or `pass` edge. A review loop stops after five requested revisions.

Run backend checks with:

```powershell
backend/.venv/Scripts/python.exe -m pytest backend/test_workflow.py -q
```