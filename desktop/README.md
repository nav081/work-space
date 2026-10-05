# Threadline

Local-first visual workflow builder for configurable LangGraph agents.

## Development

Requirements: Node.js 20.19+ (or 22.12+) and Python 3.11+.

From the `desktop` directory:

```powershell
npm install
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
```

Threadline is configured for Azure OpenAI. In `backend/.env`, set your Azure resource API key, endpoint, and API version. Set a deployment name for each Azure model you have deployed. The model catalog in `backend/models.json` maps selectable models to those deployment variables; only deployment settings selected by agents are required for a workflow run.

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