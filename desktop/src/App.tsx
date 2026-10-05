import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import {
  addEdge, Background, Controls, Handle, MiniMap, Position, ReactFlow, ReactFlowProvider,
  useEdgesState, useNodesState, useReactFlow,
  type Connection, type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import {
  Activity, ArrowRight, Check, ChevronDown, CircleHelp, ClipboardList,
  Code2, FileCode2, FileText, FlaskConical, GitBranch, Layers2, MessageSquareText,
  MoreHorizontal, Plus, Search, Settings2, ShieldCheck, Sparkles, Terminal,
  X, Zap,
} from 'lucide-react'
import './App.css'

type AgentKind = 'understand' | 'developer' | 'critic' | 'tester' | 'summary'
type AgentData = {
  kind: AgentKind
  name: string
  detail: string
  color: string
  instruction: string
  model: string
  modelLabel: string
  tools: string[]
  status?: string
}
type ModelOption = { id: string; name: string; provider: string; configured: boolean }
type ConsoleEntry = { id: number; time: string; agent: string; level: string; message: string }

const presets: Record<AgentKind, Omit<AgentData, 'status'>> = {
  understand: { kind: 'understand', name: 'Requirement analyst', detail: 'Clarifies scope and context', color: 'mint', instruction: 'Read the request and relevant project files. Produce an implementation brief with assumptions, acceptance criteria, and likely affected files or systems.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Search codebase'] },
  developer: { kind: 'developer', name: 'Developer', detail: 'Implements the agreed changes', color: 'coral', instruction: 'Implement the assigned functionality in the selected project. Follow existing conventions, keep changes focused, and report changed files and validation performed.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Edit files', 'Run commands'] },
  critic: { kind: 'critic', name: 'Code reviewer', detail: 'Reviews changes and routes fixes', color: 'blue', instruction: 'Review the implementation against requirements. Report actionable issues with file references and severity. Return to the developer when changes are needed; approve when clear.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Search codebase'] },
  tester: { kind: 'tester', name: 'Test engineer', detail: 'Verifies the completed behavior', color: 'yellow', instruction: 'Determine and run the most relevant tests for the changed functionality. Report exact commands, results, and missing coverage.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Run commands'] },
  summary: { kind: 'summary', name: 'Release notes', detail: 'Summarizes the delivered work', color: 'lilac', instruction: 'Summarize what changed, why it changed, how it was validated, and useful next improvements. Ground the summary in run results.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files'] },
}

const workflowStorageKey = 'threadline.workflow.v1'
const legacyModelIds: Record<string, string> = {
  'Claude 3.7 Sonnet': 'azure-gpt-4o',
  'claude-3-7-sonnet-latest': 'azure-gpt-4o',
  'claude-sonnet-4-5': 'azure-gpt-4o',
  'GPT-4.1': 'azure-gpt-4o',
  'gpt-4.1': 'azure-gpt-4o',
  'Gemini 2.5 Pro': 'azure-gpt-4o',
  'gemini-2.5-pro': 'azure-gpt-4o',
}

function loadWorkflow() {
  try {
    const saved = JSON.parse(localStorage.getItem(workflowStorageKey) ?? 'null')
    const savedNodes = Array.isArray(saved?.nodes) ? saved.nodes as Node<AgentData>[] : null
    const oldStarterX = [60, 340, 620, 900, 1180]
    const isOriginalStarter = savedNodes?.length === starterNodes.length && savedNodes.every((node, index) => node.id === `a${index + 1}` && node.position.x === oldStarterX[index] && node.position.y === 120)
    const hasLegacyLabels = savedNodes?.every((node) => node.data.modelLabel?.toUpperCase().startsWith('AZURE GPT-'))
    const isLegacySingleRow = hasLegacyLabels && savedNodes?.every((node) => Math.abs(node.position.y - savedNodes[0].position.y) < 24)
    const shouldMigrateLayout = isOriginalStarter || isLegacySingleRow
    return {
      nodes: savedNodes ? savedNodes.map((node, index) => { const migratedModel = legacyModelIds[node.data.model] ?? (node.data.modelLabel?.toUpperCase().startsWith('AZURE GPT-') ? 'azure-gpt-4o' : node.data.model); return { ...node, position: shouldMigrateLayout ? starterNodes[index].position : node.position, data: { ...node.data, model: migratedModel, modelLabel: migratedModel === 'azure-gpt-4o' ? 'Azure GPT-4o' : node.data.modelLabel, status: 'Ready' } } }) : starterNodes,
      edges: Array.isArray(saved?.edges) ? saved.edges as Edge[] : starterEdges,
      projectPath: typeof saved?.projectPath === 'string' ? saved.projectPath : '',
      requirement: typeof saved?.requirement === 'string' ? saved.requirement : '',
    }
  } catch {
    return { nodes: starterNodes, edges: starterEdges, projectPath: '', requirement: '' }
  }
}

const icons = { understand: MessageSquareText, developer: Code2, critic: ShieldCheck, tester: FlaskConical, summary: FileText }
const starterNodes: Node<AgentData>[] = [
  { id: 'a1', type: 'agent', position: { x: 60, y: 90 }, data: { ...presets.understand, status: 'Ready' } },
  { id: 'a2', type: 'agent', position: { x: 350, y: 90 }, data: { ...presets.developer, status: 'Ready' } },
  { id: 'a3', type: 'agent', position: { x: 640, y: 90 }, data: { ...presets.critic, status: 'Ready' } },
  { id: 'a4', type: 'agent', position: { x: 640, y: 340 }, data: { ...presets.tester, status: 'Ready' } },
  { id: 'a5', type: 'agent', position: { x: 350, y: 340 }, data: { ...presets.summary, status: 'Ready' } },
]
const starterEdges: Edge[] = [
  { id: 'e1-2', source: 'a1', target: 'a2', targetHandle: 'input', animated: true },
  { id: 'e2-3', source: 'a2', target: 'a3', sourceHandle: 'output', targetHandle: 'input', animated: true, label: 'implementation' },
  { id: 'e3-2', source: 'a3', target: 'a2', animated: true, label: 'changes requested', type: 'smoothstep', sourceHandle: 'loop-output', targetHandle: 'loop-input' },
  { id: 'e3-4', source: 'a3', target: 'a4', sourceHandle: 'output', targetHandle: 'input', animated: true, label: 'approved' },
  { id: 'e4-5', source: 'a4', target: 'a5', sourceHandle: 'output', targetHandle: 'input', animated: true },
]

function AgentNode({ data, selected }: NodeProps<Node<AgentData>>) {
  const Icon = icons[data.kind]
  return <div className={`agent-node ${selected ? 'is-selected' : ''}`}>
    <Handle id="input" type="target" position={Position.Left} />
    <Handle id="output" type="source" position={Position.Right} />
    <Handle id="loop-input" type="target" position={Position.Left} className="loop-handle" />
    <Handle id="loop-output" type="source" position={Position.Left} className="loop-handle" />
    <div className="node-topline"><span className={`agent-icon ${data.color}`}><Icon size={15} /></span><span className="node-kind">{data.kind}</span><button className="node-menu" title="Agent options"><MoreHorizontal size={16} /></button></div>
    <strong>{data.name}</strong><span className="node-detail">{data.detail}</span>
    <div className="node-bottom"><span className="model-label"><Sparkles size={12} /> {data.modelLabel}</span><span className={`node-status ${data.status === 'Running' ? 'running' : ''}`}><i />{data.status ?? 'Ready'}</span></div>
  </div>
}

const nodeTypes = { agent: AgentNode }
function WorkflowEditor() {
  const [savedWorkflow] = useState(loadWorkflow)
  const [nodes, setNodes, onNodesChange] = useNodesState(savedWorkflow.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(savedWorkflow.edges)
  const [selectedId, setSelectedId] = useState<string | null>('a2')
  const [projectPath, setProjectPath] = useState(savedWorkflow.projectPath)
  const [projectName, setProjectName] = useState(savedWorkflow.projectPath ? savedWorkflow.projectPath.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || savedWorkflow.projectPath : 'No project selected')
  const [requirement, setRequirement] = useState(savedWorkflow.requirement)
  const [runState, setRunState] = useState<'idle' | 'running' | 'complete' | 'error'>('idle')
  const [consoleEntries, setConsoleEntries] = useState<ConsoleEntry[]>([])
  const [activeTab, setActiveTab] = useState('Workflow')
  const [query, setQuery] = useState('')
  const [toast, setToast] = useState('')
  const [backendReady, setBackendReady] = useState(false)
  const [models, setModels] = useState<ModelOption[]>([])
  const { screenToFlowPosition } = useReactFlow()
  const consoleEndRef = useRef<HTMLDivElement>(null)
  const selectedNode = nodes.find((node) => node.id === selectedId)

  useEffect(() => {
    localStorage.setItem(workflowStorageKey, JSON.stringify({ nodes, edges, projectPath, requirement }))
  }, [nodes, edges, projectPath, requirement])

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' })
  }, [consoleEntries])

  useEffect(() => {
    let active = true
    Promise.all([fetch('/api/health'), fetch('/api/models')]).then(async ([healthResponse, modelsResponse]) => {
      if (!healthResponse.ok || !modelsResponse.ok) throw new Error('Local service unavailable')
      const health = await healthResponse.json()
      const availableModels = await modelsResponse.json() as ModelOption[]
      if (active) { setBackendReady(health.status === 'ok'); setModels(availableModels) }
    }).catch(() => { if (active) setBackendReady(false) })
    return () => { active = false }
  }, [])

  const onConnect = useCallback((connection: Connection) => setEdges((current) => addEdge({ ...connection, animated: true }, current)), [setEdges])
  const addAgent = useCallback((kind: AgentKind, position?: { x: number; y: number }) => {
    const id = `agent-${Date.now()}`
    setNodes((current) => [...current, { id, type: 'agent', position: position ?? { x: 170 + current.length * 35, y: 310 + (current.length * 36) % 180 }, data: { ...presets[kind], status: 'Ready' } }])
    setSelectedId(id)
  }, [setNodes])
  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    const kind = event.dataTransfer.getData('application/agent-kind') as AgentKind
    if (kind in presets) addAgent(kind, screenToFlowPosition({ x: event.clientX, y: event.clientY }))
  }, [addAgent, screenToFlowPosition])
  const updateSelected = (key: keyof AgentData, value: string | string[]) => {
    setNodes((current) => current.map((node) => node.id === selectedId ? { ...node, data: { ...node.data, [key]: value } } : node))
  }
  const appendConsole = (message: string, agent = 'SYSTEM', level = 'info') => {
    setConsoleEntries((current) => [...current, { id: Date.now() + Math.random(), time: new Date().toLocaleTimeString([], { hour12: false }), agent, level, message }].slice(-250))
  }
  const setProject = async () => {
    if (!projectPath.trim()) { setToast('Enter a local project path first'); window.setTimeout(() => setToast(''), 2600); return }
    try {
      const response = await fetch('/api/projects/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: projectPath.trim() }) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? 'Could not access that project directory.')
      setProjectPath(result.path)
      setProjectName(result.name)
      setToast('Project directory connected')
    } catch (error) {
      setToast(error instanceof Error ? error.message : 'Could not reach the local agent service.')
    }
    window.setTimeout(() => setToast(''), 3200)
  }
  const runWorkflow = async () => {
    if (runState === 'running') return
    if (!projectPath.trim()) { setToast('Connect a local project directory first'); window.setTimeout(() => setToast(''), 2600); return }
    if (!requirement.trim()) { setToast('Describe the functionality you want to build first'); window.setTimeout(() => setToast(''), 2600); return }
    setRunState('running')
    setConsoleEntries([])
    appendConsole(`Preparing ${nodes.length} agents for ${projectName}.`)
    setNodes((current) => current.map((node) => ({ ...node, data: { ...node.data, status: 'Pending' } })))
    try {
      const response = await fetch('/api/workflows/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_path: projectPath,
          requirement,
          agents: nodes.map(({ id, data }) => ({ id, kind: data.kind, name: data.name, detail: data.detail, instruction: data.instruction, model: data.model, tools: data.tools })),
          edges: edges.map(({ id, source, target, label }) => ({ id, source, target, label: typeof label === 'string' ? label : '' })),
        }),
      })
      if (!response.ok) {
        const result = await response.json()
        throw new Error(result.detail ?? 'The workflow could not be started.')
      }
      if (!response.body) throw new Error('The local service did not provide a run stream.')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let failed = false
      const handleEvent = (event: { type: string; agent_id?: string; name?: string; content?: string; message?: string; phase?: string; tool?: string }) => {
        if (event.type === 'agent_started' && event.agent_id) {
          setNodes((current) => current.map((node) => node.id === event.agent_id ? { ...node, data: { ...node.data, status: 'Running' } } : node))
          appendConsole(`${event.name ?? 'Agent'} started.`, event.name ?? 'AGENT', 'start')
        } else if (event.type === 'agent_progress') {
          appendConsole(event.message ?? 'Agent is working.', event.name ?? 'AGENT', event.phase ?? 'progress')
        } else if (event.type === 'tool_started') {
          appendConsole(event.message ?? `Calling ${event.tool ?? 'tool'}.`, event.name ?? 'AGENT', 'tool')
        } else if (event.type === 'tool_completed') {
          appendConsole(event.message ?? `${event.tool ?? 'Tool'} completed.`, event.name ?? 'AGENT', 'result')
        } else if (event.type === 'agent_completed' && event.agent_id) {
          setNodes((current) => current.map((node) => node.id === event.agent_id ? { ...node, data: { ...node.data, status: 'Complete' } } : node))
          const output = event.content ?? 'No text output returned.'
          appendConsole(`Completed. Output: ${output.slice(0, 360)}${output.length > 360 ? '...' : ''}`, event.name ?? 'AGENT', 'complete')
        } else if (event.type === 'run_error') {
          failed = true
          setRunState('error')
          setNodes((current) => current.map((node) => ({ ...node, data: { ...node.data, status: node.data.status === 'Running' ? 'Failed' : node.data.status === 'Pending' ? 'Skipped' : node.data.status } })))
          appendConsole(event.message ?? 'The agent run failed.', 'ERROR', 'error')
        } else if (event.type === 'run_completed') {
          setRunState('complete')
          setNodes((current) => current.map((node) => node.data.status === 'Pending' ? { ...node, data: { ...node.data, status: 'Skipped' } } : node))
          appendConsole('Workflow execution completed.')
        }
      }
      while (true) {
        const { done, value } = await reader.read()
        buffer += decoder.decode(value, { stream: !done })
        let boundary = buffer.indexOf('\n\n')
        while (boundary >= 0) {
          const frame = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)
          const dataLine = frame.split(/\r?\n/).find((line) => line.startsWith('data: '))
          if (dataLine) handleEvent(JSON.parse(dataLine.slice(6)))
          boundary = buffer.indexOf('\n\n')
        }
        if (done) break
      }
      if (!failed) setRunState('complete')
    } catch (error) {
      setRunState('error')
      const message = error instanceof Error ? error.message : 'Could not reach the local agent service.'
      setNodes((current) => current.map((node) => ({ ...node, data: { ...node.data, status: node.data.status === 'Running' ? 'Failed' : node.data.status === 'Pending' ? 'Skipped' : node.data.status } })))
      appendConsole(message, 'ERROR', 'error')
    }
  }
  const filteredKinds = (Object.keys(presets) as AgentKind[]).filter((kind) => `${presets[kind].name} ${presets[kind].detail}`.toLowerCase().includes(query.toLowerCase()))

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Layers2 size={18} /></span><span>THREADLINE</span><span className="brand-divider" /><span className="workspace-label">WORKSPACE</span></div>
      <div className="topbar-center"><span className="live-dot" /> LOCAL WORKSPACE <ChevronDown size={13} /></div>
      <div className="topbar-actions"><button className="icon-button" title="Help"><CircleHelp size={17} /></button><button className="icon-button" title="Settings"><Settings2 size={17} /></button><span className="avatar">R</span></div>
    </header>
    <div className="project-strip">
      <div className="project-title"><span className="project-folder"><FileCode2 size={16} /></span><div><span className="eyebrow">PROJECT</span><strong>{projectName}</strong></div></div>
      <div className="path-entry"><Terminal size={15} /><input aria-label="Local project path" placeholder="Enter local project path, e.g. D:\projects\my-app" value={projectPath} onChange={(event) => setProjectPath(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') setProject() }} /><button className="path-save" onClick={setProject}>Set path <ArrowRight size={13} /></button></div>
      <div className="project-state"><span className={`state-indicator ${projectName !== 'No project selected' ? 'connected' : ''}`} />{projectName === 'No project selected' ? 'PATH NOT SET' : 'PATH SAVED'}</div>
    </div>
    <div className="tabbar"><div className="tabs"><button className={activeTab === 'Workflow' ? 'active' : ''} onClick={() => setActiveTab('Workflow')}><GitBranch size={15} /> Workflow</button><button className={activeTab === 'Runs' ? 'active' : ''} onClick={() => setActiveTab('Runs')}><Activity size={15} /> Runs <span className="tab-count">0</span></button></div><div className="workflow-meta"><span><span className="meta-green" /> Draft</span><span className="meta-divider" /><span>{nodes.length} agents</span><button className="icon-button small" title="More workflow options"><MoreHorizontal size={17} /></button></div></div>
    <section className="requirement-bar"><span className="requirement-mark"><FileText size={16} /></span><div className="requirement-input"><span className="eyebrow">BUILD REQUEST</span><textarea aria-label="Describe the functionality to build" placeholder="Describe the functionality you want the agents to implement..." rows={2} value={requirement} onChange={(event) => setRequirement(event.target.value)} /></div><div className={`backend-badge ${backendReady ? 'online' : 'offline'}`}><span className="backend-dot" /><span>{backendReady ? 'AZURE OPENAI' : 'SERVICE OFFLINE'}</span><small>{backendReady ? `${models.filter((model) => model.configured).length} DEPLOYMENT(S) CONFIGURED` : 'START THE LANGGRAPH API'}</small></div></section>

    {activeTab === 'Workflow' ? <main className="workspace">
      <aside className="agent-library">
        <div className="panel-heading"><div><span className="eyebrow">BUILD</span><h1>Agent library</h1></div><button className="icon-button small" title="Add custom agent" onClick={() => addAgent('developer')}><Plus size={17} /></button></div>
        <label className="search-field"><Search size={15} /><input placeholder="Find an agent" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
        <div className="library-section-title">STARTER ROLES <span>{filteredKinds.length}</span></div>
        <div className="agent-list">{filteredKinds.map((kind) => { const preset = presets[kind]; const Icon = icons[kind]; return <button className="library-agent" key={kind} draggable onDragStart={(event) => { event.dataTransfer.setData('application/agent-kind', kind); event.dataTransfer.effectAllowed = 'move' }} onClick={() => addAgent(kind)} title="Click or drag onto the canvas to add"><span className={`agent-icon ${preset.color}`}><Icon size={16} /></span><span className="library-agent-copy"><strong>{preset.name}</strong><small>{preset.detail}</small></span><Plus className="library-add" size={15} /></button> })}</div>
        <button className="custom-agent-button" onClick={() => addAgent('developer')}><Plus size={15} /> Create custom agent</button>
        <div className="library-tip"><Sparkles size={15} /><span><strong>Shape the workflow</strong><small>Drag a role onto the canvas, then connect it to define what happens next.</small></span></div>
        <div className="library-footer"><span className="footer-status" /><span>LOCAL WORKSPACE</span><span className="footer-version">v0.1.0</span></div>
      </aside>

      <section className="canvas-column">
        <div className="canvas-toolbar"><div className="canvas-title"><h2>Application workflow</h2><span className="draft-badge"><span /> DRAFT</span></div><div className="canvas-actions"><button className="secondary-button" onClick={() => { setNodes(starterNodes); setEdges(starterEdges); setSelectedId('a2') }}>Reset canvas</button><button className="run-button" onClick={runWorkflow} disabled={runState === 'running'}><Zap size={15} /> {runState === 'running' ? 'Running demo' : runState === 'complete' ? 'Run again' : 'Run workflow'} <span className="run-shortcut">Ctrl ↵</span></button></div></div>
        <div className="canvas-wrap" onDrop={onDrop} onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }}>
          <div className="canvas-caption"><span>MAIN FLOW</span><span className="caption-rule" /><span>{nodes.length} STEPS · REVIEW LOOP</span></div>
          <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={(_, node) => setSelectedId(node.id)} onPaneClick={() => setSelectedId(null)} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.25} maxZoom={1.4}>
            <Background color="#d8ddd8" gap={22} size={1} /><Controls position="bottom-left" showInteractive={false} /><MiniMap position="bottom-right" pannable zoomable nodeColor={(node) => ({ mint: '#77b9a0', coral: '#d9856e', blue: '#7197b9', yellow: '#c8a84d', lilac: '#a796c5' }[node.data.color as string] ?? '#aab2ac')} />
          </ReactFlow>
          {nodes.length === 0 && <div className="canvas-empty"><Layers2 size={23} /><strong>Start with an agent</strong><span>Drag one from the library to shape your workflow.</span></div>}
          <div className="canvas-bottom-note"><span className="keyboard-hint">SPACE</span> pan <span className="note-dot">·</span> scroll to zoom</div>
        </div>
        <div className="canvas-statusbar"><span><span className="status-green" /> All changes saved</span><span className="statusbar-right"><GitBranch size={13} /> main <span className="meta-divider" /> Updated just now</span></div>
      </section>

      <aside className="inspector">
        <div className="inspector-tabs"><button className="selected"><Settings2 size={15} /> Configure</button><button><ClipboardList size={15} /> Context</button></div>
        {selectedNode ? <>
          <div className="inspector-title"><span className={`agent-icon ${selectedNode.data.color}`}><Settings2 size={16} /></span><div><span className="eyebrow">AGENT CONFIGURATION</span><strong>{selectedNode.data.name}</strong></div><button className="icon-button small" title="Close selection" onClick={() => setSelectedId(null)}><X size={16} /></button></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-name">Agent name</label><input id="agent-name" className="text-input" value={selectedNode.data.name} onChange={(event) => updateSelected('name', event.target.value)} /></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-kind">Role</label><div className="select-wrap"><select id="agent-kind" value={selectedNode.data.kind} onChange={(event) => { const preset = presets[event.target.value as AgentKind]; setNodes((current) => current.map((node) => node.id === selectedId ? { ...node, data: { ...preset, status: node.data.status } } : node)) }}>{Object.keys(presets).map((kind) => <option key={kind} value={kind}>{presets[kind as AgentKind].name}</option>)}</select><ChevronDown size={15} /></div></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-instructions">Instructions <span className="label-hint">PROMPT</span></label><textarea id="agent-instructions" className="instructions-input" value={selectedNode.data.instruction} onChange={(event) => updateSelected('instruction', event.target.value)} rows={7} /><span className="field-footnote">This agent receives upstream results and project context.</span></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-model">Azure deployment</label><div className="select-wrap"><select id="agent-model" value={selectedNode.data.model} onChange={(event) => { const model = models.find((item) => item.id === event.target.value); updateSelected('model', event.target.value); updateSelected('modelLabel', model?.name ?? event.target.value) }} disabled={!models.some((model) => model.configured)}><option value={selectedNode.data.model}>{models.find((model) => model.id === selectedNode.data.model)?.name ?? selectedNode.data.modelLabel}</option>{models.filter((model) => model.configured && model.id !== selectedNode.data.model).map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select><ChevronDown size={15} /></div><span className="field-footnote">Set Azure key, endpoint, API version, and deployment in backend/.env.</span></div>
          <div className="form-section tools-section"><div className="field-label">Tools <span className="label-hint">{selectedNode.data.tools.length} ENABLED</span></div><div className="tool-chips">{['Read files', 'Search codebase', 'Edit files', 'Run commands'].map((tool) => <button className={`tool-chip ${selectedNode.data.tools.includes(tool) ? 'enabled' : ''}`} key={tool} onClick={() => updateSelected('tools', selectedNode.data.tools.includes(tool) ? selectedNode.data.tools.filter((item) => item !== tool) : [...selectedNode.data.tools, tool])}><span className="tool-check">{selectedNode.data.tools.includes(tool) && <Check size={11} />}</span>{tool}</button>)}</div>{selectedNode.data.tools.includes('Run commands') && <span className="tool-warning">Commands run on this computer. Enable only for projects you trust.</span>}</div>
          <div className="inspector-bottom"><span><span className="status-green" /> Config saved</span><button className="delete-agent" onClick={() => { setNodes((current) => current.filter((node) => node.id !== selectedId)); setEdges((current) => current.filter((edge) => edge.source !== selectedId && edge.target !== selectedId)); setSelectedId(null) }}><X size={13} /> Remove agent</button></div>
        </> : <div className="inspector-empty"><Settings2 size={22} /><strong>Select an agent</strong><span>Choose a canvas node to edit its role, instructions, model, and tools.</span></div>}
      </aside>
    </main> : <main className="runs-view"><div className="runs-heading"><div><span className="eyebrow">EXECUTION HISTORY</span><h1>Runs</h1><p>Workflow runs and agent activity will appear here.</p></div><span className="runs-zero">0 RUNS</span></div><div className="runs-empty"><Activity size={24} /><strong>No workflow runs yet</strong><span>Start a run from the Workflow tab to see activity here.</span><button className="secondary-button" onClick={() => setActiveTab('Workflow')}>Go to workflow <ArrowRight size={14} /></button></div></main>}

    <section className="activity-drawer">
      <div className="activity-heading">
        <div className="activity-title"><span className={`activity-pulse ${runState === 'running' ? 'pulsing' : ''}`}><Activity size={15} /></span><strong>Execution monitor</strong><span className="activity-count">{nodes.length} agents</span></div>
        <div className="activity-tools"><span className={`run-indicator ${runState}`}>{runState === 'running' ? 'RUNNING' : runState === 'complete' ? 'COMPLETED' : runState === 'error' ? 'ERROR' : 'IDLE'}</span></div>
      </div>
      <div className="execution-console-grid">
        <section className="agent-status-panel" aria-label="Agent execution status">
          <div className="monitor-section-heading"><span>AGENT</span><span>MODEL</span><span>STATUS</span></div>
          <div className="agent-status-rows">{nodes.map((node) => <div className="agent-status-row" key={node.id}><span className="agent-row-name"><i className={`agent-state-dot ${(node.data.status ?? 'Ready').toLowerCase()}`} />{node.data.name}</span><span className="agent-row-model">{node.data.modelLabel}</span><span className={`agent-state-label ${(node.data.status ?? 'Ready').toLowerCase()}`}>{node.data.status ?? 'Ready'}</span></div>)}</div>
        </section>
        <section className="agent-console-panel" aria-label="Workflow console">
          <div className="console-toolbar"><span><Terminal size={12} /> LIVE CONSOLE</span><span>{consoleEntries.length} EVENTS</span></div>
          <div className="console-lines" role="log" aria-live="polite">{consoleEntries.length === 0 ? <div className="console-empty">Agent and tool activity will appear here when a workflow runs.</div> : consoleEntries.map((entry) => <div className={`console-line ${entry.level}`} key={entry.id}><time>{entry.time}</time><span className="console-agent">{entry.agent}</span><span className="console-message">{entry.message}</span></div>)}<div ref={consoleEndRef} /></div>
        </section>
      </div>
    </section>
    {toast && <div className="toast"><Check size={15} />{toast}</div>}
    <footer className="app-footer"><span>THREADLINE <span className="footer-separator">/</span> WORKFLOW BUILDER</span><span>LOCAL-FIRST AGENT ORCHESTRATION <span className="footer-separator">·</span> PROTOTYPE</span></footer>
  </div>
}

function App() {
  return <ReactFlowProvider><WorkflowEditor /></ReactFlowProvider>
}

export default App