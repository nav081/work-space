import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from 'react'
import {
  addEdge, Background, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow, ReactFlowProvider,
  useEdgesState, useNodesState, useReactFlow,
  type Connection, type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import {
  Activity, ArrowRight, Check, ChevronDown, CircleHelp, ClipboardList, Download,
  Code2, FileCode2, FileText, FlaskConical, GitBranch, Layers2, MessageSquareText,
  FileUp, MoreHorizontal, Plus, Save, Search, Settings2, ShieldCheck, Sparkles, Terminal,
  Trash2, X, Zap,
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
  successEdgeId?: string
  failureEdgeId?: string
  maxRetries?: number
  status?: string
}
type ModelOption = { id: string; name: string; provider: string; model: string; deployment: string; endpoint: string; api_version: string; api_key_configured: boolean; configured: boolean }
type ConsoleEntry = { id: number; time: string; agent: string; level: string; message: string; detail?: string }
type ModelForm = { id?: string; name: string; model: string; deployment: string; endpoint: string; api_version: string; api_key: string }
type WorkflowSnapshot = { nodes: Node<AgentData>[]; edges: Edge[]; projectPath: string; requirement: string }
type RunSummary = { run_id: string; started_at: string; completed_at?: string | null; status: string; project_name: string; project_path: string; requirement: string; agent_count: number; agents: Array<{ id: string; name: string; kind: string }> }
type PersistedRunEvent = { created_at: string; type: string; agent_id?: string; agent_name?: string; name?: string; message?: string; content?: string; detail?: string; tool?: string; target_name?: string; connection_label?: string; decision?: string }
type RunDetail = RunSummary & { workflow: { agents: Array<{ id: string; name: string; kind: string; model: string; instruction: string; tools: string[] }>; edges: Array<{ source: string; target: string; label: string }> }; events: PersistedRunEvent[] }

const emptyModelForm: ModelForm = { name: '', model: '', deployment: '', endpoint: '', api_version: '', api_key: '' }

const presets: Record<AgentKind, Omit<AgentData, 'status'>> = {
  understand: { kind: 'understand', name: 'Requirement analyst', detail: 'Clarifies scope and context', color: 'mint', instruction: 'Read the request and relevant project files. Produce an implementation brief with assumptions, acceptance criteria, and likely affected files or systems.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Search codebase'] },
  developer: { kind: 'developer', name: 'Developer', detail: 'Implements the agreed changes', color: 'coral', instruction: 'Implement the assigned functionality in the selected project. Follow existing conventions, keep changes focused, and report changed files and validation performed.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Edit files', 'Run commands'] },
  critic: { kind: 'critic', name: 'Code reviewer', detail: 'Reviews changes and routes fixes', color: 'blue', instruction: 'Review the implementation against requirements. Report actionable issues with file references and severity. Return to the developer when changes are needed; approve when clear.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Search codebase'] },
  tester: { kind: 'tester', name: 'Test engineer', detail: 'Verifies the completed behavior', color: 'yellow', instruction: 'Run the requested focused tests and report exact commands and results. If a test fails, identify the failure and route it to Developer. When Developer returns with a fix, rerun the failing test, then run the full relevant suite before reporting success.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files', 'Run commands', 'Install dependencies'] },
  summary: { kind: 'summary', name: 'Release notes', detail: 'Summarizes the delivered work', color: 'lilac', instruction: 'Summarize what changed, why it changed, how it was validated, and useful next improvements. Ground the summary in run results.', model: 'azure-gpt-4o', modelLabel: 'Azure GPT-4o', tools: ['Read files'] },
}

const workflowStorageKey = 'threadline.workflow.v1'
const withArrowMarker = (edge: Edge): Edge => ({ ...edge, markerEnd: edge.markerEnd ?? { type: MarkerType.ArrowClosed } })
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
    const migrateTesterDependencies = saved?.settingsVersion !== 2
    const oldStarterX = [60, 340, 620, 900, 1180]
    const isOriginalStarter = savedNodes?.length === starterNodes.length && savedNodes.every((node, index) => node.id === `a${index + 1}` && node.position.x === oldStarterX[index] && node.position.y === 120)
    const hasLegacyLabels = savedNodes?.every((node) => node.data.modelLabel?.toUpperCase().startsWith('AZURE GPT-'))
    const isLegacySingleRow = hasLegacyLabels && savedNodes?.every((node) => Math.abs(node.position.y - savedNodes[0].position.y) < 24)
    const shouldMigrateLayout = isOriginalStarter || isLegacySingleRow
    return {
      nodes: savedNodes ? savedNodes.map((node, index) => { const migratedModel = legacyModelIds[node.data.model] ?? (node.data.modelLabel?.toUpperCase().startsWith('AZURE GPT-') ? 'azure-gpt-4o' : node.data.model); const tools = node.data.kind === 'tester' && migrateTesterDependencies ? [...new Set([...node.data.tools, 'Install dependencies'])] : node.data.tools; return { ...node, position: shouldMigrateLayout ? starterNodes[index].position : node.position, data: { ...node.data, tools, model: migratedModel, modelLabel: migratedModel === 'azure-gpt-4o' ? 'Azure GPT-4o' : node.data.modelLabel, status: 'Ready' } } }) : starterNodes,
      edges: Array.isArray(saved?.edges) ? (saved.edges as Edge[]).map(withArrowMarker) : starterEdges.map(withArrowMarker),
      projectPath: typeof saved?.projectPath === 'string' ? saved.projectPath : '',
      requirement: typeof saved?.requirement === 'string' ? saved.requirement : '',
    }
  } catch {
    return { nodes: starterNodes, edges: starterEdges.map(withArrowMarker), projectPath: '', requirement: '' }
  }
}

function workflowSnapshot(snapshot: WorkflowSnapshot): WorkflowSnapshot {
  return {
    ...snapshot,
    nodes: snapshot.nodes.map((node) => {
      const savedNode = { ...node }
      delete savedNode.selected
      delete savedNode.dragging
      delete savedNode.measured
      delete savedNode.width
      delete savedNode.height
      delete savedNode.resizing
      return { ...savedNode, data: { ...node.data, status: 'Ready' } }
    }),
    edges: snapshot.edges.map((edge) => {
      const savedEdge = { ...edge }
      delete savedEdge.selected
      return { ...savedEdge, label: typeof edge.label === 'string' ? edge.label : '' }
    }),
  }
}

function workflowSignature(snapshot: WorkflowSnapshot): string {
  return JSON.stringify(workflowSnapshot(snapshot))
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
  const [savedWorkflow, setSavedWorkflow] = useState(loadWorkflow)
  const [nodes, setNodes, onNodesChange] = useNodesState(savedWorkflow.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(savedWorkflow.edges)
  const [selectedId, setSelectedId] = useState<string | null>('a2')
  const [projectPath, setProjectPath] = useState(savedWorkflow.projectPath)
  const [projectName, setProjectName] = useState(savedWorkflow.projectPath ? savedWorkflow.projectPath.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || savedWorkflow.projectPath : 'No project selected')
  const [requirement, setRequirement] = useState(savedWorkflow.requirement)
  const [runState, setRunState] = useState<'idle' | 'running' | 'complete' | 'error'>('idle')
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [consoleEntries, setConsoleEntries] = useState<ConsoleEntry[]>([])
  const [activeTab, setActiveTab] = useState('Workflow')
  const [runHistory, setRunHistory] = useState<RunSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [runViewerOpen, setRunViewerOpen] = useState(false)
  const [runViewerDetail, setRunViewerDetail] = useState<RunDetail | null>(null)
  const [runViewerLoading, setRunViewerLoading] = useState(false)
  const [query, setQuery] = useState('')
  const [toast, setToast] = useState('')
  const [backendReady, setBackendReady] = useState(false)
  const [models, setModels] = useState<ModelOption[]>([])
  const [modelForm, setModelForm] = useState<ModelForm>(emptyModelForm)
  const [editingModelId, setEditingModelId] = useState<string | null>(null)
  const [savingModel, setSavingModel] = useState(false)
  const { screenToFlowPosition } = useReactFlow()
  const consoleEndRef = useRef<HTMLDivElement>(null)
  const importInputRef = useRef<HTMLInputElement>(null)
  const consoleSequenceRef = useRef(0)
  const agentSequenceRef = useRef(0)
  const selectedNode = nodes.find((node) => node.id === selectedId)
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null)
  const selectedEdge = edges.find((edge) => edge.id === selectedEdgeId)
  const selectedNodeOutgoing = selectedNode ? edges.filter((edge) => edge.source === selectedNode.id) : []
  const hasUnsavedChanges = workflowSignature({ nodes, edges, projectPath, requirement }) !== workflowSignature(savedWorkflow)

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' })
  }, [consoleEntries])

  useEffect(() => {
    let active = true
    Promise.all([fetch('/api/health'), fetch('/api/settings/models')]).then(async ([healthResponse, modelsResponse]) => {
      if (!healthResponse.ok || !modelsResponse.ok) throw new Error('Local service unavailable')
      const health = await healthResponse.json()
      const availableModels = await modelsResponse.json() as ModelOption[]
      if (active) { setBackendReady(health.status === 'ok'); setModels(availableModels) }
    }).catch(() => { if (active) setBackendReady(false) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (activeTab !== 'Runs') return
    let active = true
    fetch('/api/runs?limit=100').then(async (response) => {
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? 'Could not load run history.')
      if (active) { setRunHistory(result); setHistoryError('') }
    }).catch((error) => { if (active) setHistoryError(error instanceof Error ? error.message : 'Could not load run history.') })
      .finally(() => { if (active) setHistoryLoading(false) })
    return () => { active = false }
  }, [activeTab])

  const onConnect = useCallback((connection: Connection) => {
    setEdges((current) => addEdge({ ...connection, label: '', animated: true, markerEnd: { type: MarkerType.ArrowClosed } }, current))
    setSelectedId(null)
  }, [setEdges])
  const addAgent = useCallback((kind: AgentKind, position?: { x: number; y: number }) => {
    const id = `agent-${Date.now()}-${++agentSequenceRef.current}`
    setNodes((current) => [...current, { id, type: 'agent', position: position ?? { x: 170 + current.length * 35, y: 310 + (current.length * 36) % 180 }, data: { ...presets[kind], status: 'Ready' } }])
    setSelectedId(id)
    setSelectedEdgeId(null)
  }, [setNodes])
  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    const kind = event.dataTransfer.getData('application/agent-kind') as AgentKind
    if (kind in presets) addAgent(kind, screenToFlowPosition({ x: event.clientX, y: event.clientY }))
  }, [addAgent, screenToFlowPosition])
  const updateSelected = (key: keyof AgentData, value: string | string[]) => {
    setNodes((current) => current.map((node) => node.id === selectedId ? { ...node, data: { ...node.data, [key]: value } } : node))
  }
  const updateSelectedMaxRetries = (value: number) => {
    setNodes((current) => current.map((node) => node.id === selectedId ? { ...node, data: { ...node.data, maxRetries: value } } : node))
  }
  const saveWorkflow = () => {
    const snapshot = workflowSnapshot({ nodes, edges, projectPath, requirement })
    localStorage.setItem(workflowStorageKey, JSON.stringify({ format: 'threadline.workflow', version: 1, settingsVersion: 2, ...snapshot }))
    setSavedWorkflow(snapshot)
    setNodes(snapshot.nodes)
    setEdges(snapshot.edges)
    setToast('Workflow saved. Agents will use this version.')
    window.setTimeout(() => setToast(''), 2800)
  }
  const discardWorkflowDraft = () => {
    if (!hasUnsavedChanges || !window.confirm('Discard all unsaved workflow changes?')) return
    setNodes(savedWorkflow.nodes)
    setEdges(savedWorkflow.edges)
    setProjectPath(savedWorkflow.projectPath)
    setProjectName(savedWorkflow.projectPath ? savedWorkflow.projectPath.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || savedWorkflow.projectPath : 'No project selected')
    setRequirement(savedWorkflow.requirement)
    setSelectedId(null)
    setSelectedEdgeId(null)
    setToast('Unsaved changes discarded')
    window.setTimeout(() => setToast(''), 2800)
  }
  const exportWorkflow = () => {
    const snapshot = workflowSnapshot({ nodes, edges, projectPath, requirement })
    const document = { format: 'threadline.workflow', version: 1, exportedAt: new Date().toISOString(), ...snapshot }
    const blob = new Blob([JSON.stringify(document, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = window.document.createElement('a')
    link.href = url
    link.download = 'threadline-workflow.json'
    link.click()
    URL.revokeObjectURL(url)
  }
  const importWorkflow = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget
    const file = input.files?.[0]
    if (!file) return
    try {
      if (hasUnsavedChanges && !window.confirm('Discard the current unsaved workflow draft and import this file?')) return
      const document = JSON.parse(await file.text())
      if (document.format !== 'threadline.workflow' || document.version !== 1 || !Array.isArray(document.nodes) || !Array.isArray(document.edges)) {
        throw new Error('Choose a supported Threadline workflow JSON file.')
      }
      if (document.nodes.length > 100 || document.edges.length > 300) throw new Error('This workflow exceeds the supported size.')
      const importedNodes: Node<AgentData>[] = document.nodes.map((node: Node<AgentData>) => {
        const preset = node?.data && presets[node.data.kind]
        if (!node || typeof node.id !== 'string' || !preset || !Number.isFinite(node.position?.x) || !Number.isFinite(node.position?.y)) {
          throw new Error('Workflow contains an invalid agent.')
        }
        if (typeof node.data.name !== 'string' || typeof node.data.instruction !== 'string' || typeof node.data.model !== 'string' || !Array.isArray(node.data.tools)) {
          throw new Error(`Agent ${node.id} is missing required configuration.`)
        }
        return {
          id: node.id,
          type: 'agent',
          position: { x: node.position.x, y: node.position.y },
          data: {
            kind: node.data.kind,
            name: node.data.name,
            detail: typeof node.data.detail === 'string' ? node.data.detail : preset.detail,
            color: preset.color,
            instruction: node.data.instruction,
            model: node.data.model,
            modelLabel: typeof node.data.modelLabel === 'string' ? node.data.modelLabel : node.data.model,
            tools: node.data.tools.filter((tool): tool is string => typeof tool === 'string'),
            successEdgeId: typeof node.data.successEdgeId === 'string' ? node.data.successEdgeId : undefined,
            failureEdgeId: typeof node.data.failureEdgeId === 'string' ? node.data.failureEdgeId : undefined,
            maxRetries: typeof node.data.maxRetries === 'number' && Number.isInteger(node.data.maxRetries) && node.data.maxRetries >= 0 && node.data.maxRetries <= 20 ? node.data.maxRetries : 5,
            status: 'Ready',
          },
        }
      })
      const nodeIds = new Set(importedNodes.map((node) => node.id))
      const importedEdges: Edge[] = document.edges.map((edge: Edge) => {
        if (!edge || typeof edge.id !== 'string' || !nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
          throw new Error('Workflow contains a connection to a missing agent.')
        }
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
          type: edge.type,
          animated: edge.animated,
          markerEnd: edge.markerEnd ?? { type: MarkerType.ArrowClosed },
          label: typeof edge.label === 'string' ? edge.label : '',
        }
      })
      const snapshot = workflowSnapshot({
        nodes: importedNodes,
        edges: importedEdges,
        projectPath: typeof document.projectPath === 'string' ? document.projectPath : '',
        requirement: typeof document.requirement === 'string' ? document.requirement : '',
      })
      setNodes(snapshot.nodes)
      setEdges(snapshot.edges)
      setProjectPath(snapshot.projectPath)
      setProjectName(snapshot.projectPath ? snapshot.projectPath.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || snapshot.projectPath : 'No project selected')
      setRequirement(snapshot.requirement)
      setSelectedId(snapshot.nodes[0]?.id ?? null)
      setSelectedEdgeId(null)
      setToast('Workflow imported as a draft. Save it before running agents.')
    } catch (error) {
      setToast(error instanceof Error ? error.message : 'Could not import that workflow file.')
    } finally {
      input.value = ''
      window.setTimeout(() => setToast(''), 4000)
    }
  }
  const updateSelectedEdgeLabel = (label: string) => {
    setEdges((current) => current.map((edge) => edge.id === selectedEdgeId ? { ...edge, label } : edge))
  }
  const deleteSelectedEdge = () => {
    if (!selectedEdgeId) return
    setEdges((current) => current.filter((edge) => edge.id !== selectedEdgeId))
    setSelectedEdgeId(null)
  }
  const appendConsole = (message: string, agent = 'SYSTEM', level = 'info', detail?: string) => {
    const id = ++consoleSequenceRef.current
    setConsoleEntries((current) => [...current, { id, time: new Date().toLocaleTimeString([], { hour12: false }), agent, level, message, detail }])
  }
  const openCurrentRun = () => {
    setRunViewerDetail(null)
    setRunViewerOpen(true)
  }
  const openHistoricalRun = async (runId: string) => {
    setRunViewerOpen(true)
    setRunViewerLoading(true)
    setRunViewerDetail(null)
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`)
      const detail = await response.json()
      if (!response.ok) throw new Error(detail.detail ?? 'Could not load this run.')
      setRunViewerDetail(detail)
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Could not load this run.')
    } finally {
      setRunViewerLoading(false)
    }
  }
  const runViewerEvents: ConsoleEntry[] = runViewerDetail
    ? runViewerDetail.events.map((event, index) => ({
      id: index + 1,
      time: new Date(event.created_at).toLocaleTimeString([], { hour12: false }),
      agent: event.name ?? event.agent_name ?? 'SYSTEM',
      level: event.type,
      message: event.message ?? (event.type === 'agent_completed' ? 'Agent completed.' : event.type.replaceAll('_', ' ')),
      detail: event.detail ?? event.content,
    }))
    : consoleEntries
  const runViewerTitle = runViewerDetail
    ? `${runViewerDetail.project_name} · ${new Date(runViewerDetail.started_at).toLocaleString()}`
    : currentRunId ? `${projectName} · Live run` : 'No active run'
  const editModel = (model: ModelOption) => {
    setEditingModelId(model.id)
    setModelForm({ id: model.id, name: model.name, model: model.model, deployment: model.deployment, endpoint: model.endpoint, api_version: model.api_version, api_key: '' })
  }
  const resetModelForm = () => {
    setEditingModelId(null)
    setModelForm(emptyModelForm)
  }
  const saveModel = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSavingModel(true)
    try {
      const response = await fetch('/api/settings/models', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(modelForm) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? 'Could not save model settings.')
      setModels((current) => current.some((model) => model.id === result.id) ? current.map((model) => model.id === result.id ? result : model) : [...current, result])
      resetModelForm()
      setToast('Model settings saved securely')
    } catch (error) {
      setToast(error instanceof Error ? error.message : 'Could not save model settings.')
    } finally {
      setSavingModel(false)
      window.setTimeout(() => setToast(''), 3200)
    }
  }
  const removeModel = async (model: ModelOption) => {
    if (!window.confirm(`Remove ${model.name} and its saved API key from this computer?`)) return
    try {
      const response = await fetch(`/api/settings/models/${encodeURIComponent(model.id)}`, { method: 'DELETE' })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? 'Could not remove model settings.')
      setModels((current) => current.filter((item) => item.id !== model.id))
      if (editingModelId === model.id) resetModelForm()
      setToast('Model and saved credential removed')
    } catch (error) {
      setToast(error instanceof Error ? error.message : 'Could not remove model settings.')
    }
    window.setTimeout(() => setToast(''), 3200)
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
    if (hasUnsavedChanges) { setToast('Save workflow changes before running agents.'); window.setTimeout(() => setToast(''), 2800); return }
    if (!projectPath.trim()) { setToast('Connect a local project directory first'); window.setTimeout(() => setToast(''), 2600); return }
    if (!requirement.trim()) { setToast('Describe the functionality you want to build first'); window.setTimeout(() => setToast(''), 2600); return }
    setRunState('running')
    setCurrentRunId(null)
    setConsoleEntries([])
    appendConsole(`Preparing ${nodes.length} agents for ${projectName}.`)
    setNodes((current) => current.map((node) => ({ ...node, data: { ...node.data, status: 'Pending' } })))
    try {
      const response = await fetch('/api/workflows/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_path: savedWorkflow.projectPath,
          requirement: savedWorkflow.requirement,
          agents: savedWorkflow.nodes.map(({ id, data }) => ({ id, kind: data.kind, name: data.name, detail: data.detail, instruction: data.instruction, model: data.model, tools: data.tools, success_edge_id: data.successEdgeId, failure_edge_id: data.failureEdgeId, max_retries: data.maxRetries ?? 5 })),
          edges: savedWorkflow.edges.map(({ id, source, target, label }) => ({ id, source, target, label: typeof label === 'string' ? label : '' })),
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
      const handleEvent = (event: { type: string; run_id?: string; agent_id?: string; name?: string; content?: string; message?: string; detail?: string; phase?: string; tool?: string; target_name?: string; connection_label?: string; decision?: string }) => {
        if (event.type === 'run_started') {
          if (event.run_id) setCurrentRunId(event.run_id)
          appendConsole(`Run ${event.run_id ?? ''} started for ${event.name ?? projectName}.`)
        } else if (event.type === 'agent_started' && event.agent_id) {
          setNodes((current) => current.map((node) => node.id === event.agent_id ? { ...node, data: { ...node.data, status: 'Running' } } : node))
          appendConsole(`${event.name ?? 'Agent'} started.`, event.name ?? 'AGENT', 'start')
        } else if (event.type === 'agent_progress') {
          appendConsole(event.message ?? 'Agent is working.', event.name ?? 'AGENT', event.phase ?? 'progress', event.detail)
        } else if (event.type === 'tool_started') {
          appendConsole(event.message ?? `Calling ${event.tool ?? 'tool'}.`, event.name ?? 'AGENT', 'tool', event.detail)
        } else if (event.type === 'tool_completed') {
          appendConsole(event.message ?? `${event.tool ?? 'Tool'} completed.`, event.name ?? 'AGENT', 'result', event.detail)
        } else if (event.type === 'handoff_routed') {
          appendConsole(`${event.decision?.toUpperCase() ?? 'CONTINUE'} handoff to ${event.target_name ?? 'next agent'} via "${event.connection_label || 'default connection'}". ${event.message ?? ''}`, event.name ?? 'AGENT', 'handoff', event.detail)
        } else if (event.type === 'agent_completed' && event.agent_id) {
          setNodes((current) => current.map((node) => node.id === event.agent_id ? { ...node, data: { ...node.data, status: 'Complete' } } : node))
          const output = event.content ?? 'No text output returned.'
          appendConsole(`Completed with ${output.length} output characters.`, event.name ?? 'AGENT', 'complete', output)
        } else if (event.type === 'run_error') {
          failed = true
          setRunState('error')
          setNodes((current) => current.map((node) => ({ ...node, data: { ...node.data, status: node.data.status === 'Running' ? 'Failed' : node.data.status === 'Pending' ? 'Skipped' : node.data.status } })))
          appendConsole(event.message ?? 'The agent run failed.', 'ERROR', 'error')
          fetch('/api/runs?limit=100').then((response) => response.json()).then((runs) => setRunHistory(runs)).catch(() => undefined)
        } else if (event.type === 'run_completed') {
          setRunState('complete')
          setNodes((current) => current.map((node) => node.data.status === 'Pending' ? { ...node, data: { ...node.data, status: 'Skipped' } } : node))
          appendConsole('Workflow execution completed.')
          fetch('/api/runs?limit=100').then((response) => response.json()).then((runs) => setRunHistory(runs)).catch(() => undefined)
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
      <div className="topbar-actions"><button className="icon-button" title="Help"><CircleHelp size={17} /></button><button className={`icon-button ${activeTab === 'Settings' ? 'active' : ''}`} title="Settings" aria-label="Settings" onClick={() => setActiveTab(activeTab === 'Settings' ? 'Workflow' : 'Settings')}><Settings2 size={17} /></button><span className="avatar">R</span></div>
    </header>
    <div className="project-strip">
      <div className="project-title"><span className="project-folder"><FileCode2 size={16} /></span><div><span className="eyebrow">PROJECT</span><strong>{projectName}</strong></div></div>
      <div className="path-entry"><Terminal size={15} /><input aria-label="Local project path" placeholder="Enter local project path, e.g. D:\projects\my-app" value={projectPath} onChange={(event) => setProjectPath(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') setProject() }} /><button className="path-save" onClick={setProject}>Set path <ArrowRight size={13} /></button></div>
      <div className="project-state"><span className={`state-indicator ${projectName !== 'No project selected' ? 'connected' : ''}`} />{projectName === 'No project selected' ? 'PATH NOT SET' : 'PATH SAVED'}</div>
    </div>
    <div className="tabbar"><div className="tabs"><button className={activeTab === 'Workflow' ? 'active' : ''} onClick={() => setActiveTab('Workflow')}><GitBranch size={15} /> Workflow</button><button className={activeTab === 'Runs' ? 'active' : ''} onClick={() => { setHistoryLoading(true); setActiveTab('Runs') }}><Activity size={15} /> Runs <span className="tab-count">{runHistory.length}</span></button></div><div className="workflow-meta"><span><span className={hasUnsavedChanges ? 'meta-amber' : 'meta-green'} /> {hasUnsavedChanges ? 'Unsaved draft' : 'Saved'}</span><span className="meta-divider" /><span>{nodes.length} agents</span></div></div>
    <section className="requirement-bar"><span className="requirement-mark"><FileText size={16} /></span><div className="requirement-input"><span className="eyebrow">BUILD REQUEST</span><textarea aria-label="Describe the functionality to build" placeholder="Describe the functionality you want the agents to implement..." rows={2} value={requirement} onChange={(event) => setRequirement(event.target.value)} /></div><div className={`backend-badge ${backendReady ? 'online' : 'offline'}`}><span className="backend-dot" /><span>{backendReady ? 'AZURE OPENAI' : 'SERVICE OFFLINE'}</span><small>{backendReady ? `${models.filter((model) => model.configured).length} DEPLOYMENT(S) CONFIGURED` : 'START THE LANGGRAPH API'}</small></div></section>

    {activeTab === 'Settings' ? <main className="settings-view">
      <div className="settings-heading"><div><span className="eyebrow">LOCAL CONFIGURATION</span><h1>Models & credentials</h1><p>Model endpoints are stored in your user profile. API keys are saved in Windows Credential Manager and never returned to the app.</p></div><span className="settings-secure-badge"><ShieldCheck size={14} /> WINDOWS CREDENTIAL MANAGER</span></div>
      <div className="settings-layout">
        <section className="saved-models-panel"><div className="settings-section-heading"><div><span className="eyebrow">MODEL CATALOG</span><h2>Azure deployments</h2></div><span className="settings-count">{models.length}</span></div>
          {models.length === 0 ? <div className="settings-empty"><Layers2 size={20} /><strong>No models configured</strong><span>Add an Azure deployment to make it available to agents.</span></div> : <div className="saved-model-list">{models.map((model) => <article className="saved-model-row" key={model.id}><div className="saved-model-icon"><Sparkles size={16} /></div><div className="saved-model-main"><strong>{model.name}</strong><span>{model.model} <i /> {model.deployment || 'Deployment not set'}</span><small title={model.endpoint}>{model.endpoint || 'Endpoint not set'}{model.api_version ? ` · API ${model.api_version}` : ' · v1 API'}</small></div><div className="saved-model-status"><span className={`model-config-dot ${model.configured ? 'configured' : ''}`} /><span>{model.configured ? 'READY' : model.api_key_configured ? 'INCOMPLETE' : 'KEY NEEDED'}</span></div><div className="saved-model-actions"><button className="icon-button small" title={`Edit ${model.name}`} aria-label={`Edit ${model.name}`} onClick={() => editModel(model)}><Settings2 size={15} /></button><button className="icon-button small remove-model-button" title={`Remove ${model.name}`} aria-label={`Remove ${model.name}`} onClick={() => removeModel(model)}><X size={15} /></button></div></article>)}</div>}
          <div className="settings-security-note"><ShieldCheck size={15} /><span><strong>Credentials stay on this Windows account</strong><small>Only model metadata is written to the Threadline user settings file. Keys are stored in Windows Credential Manager.</small></span></div>
        </section>
        <form className="model-form-panel" onSubmit={saveModel}>
          <div className="settings-section-heading"><div><span className="eyebrow">{editingModelId ? 'EDIT DEPLOYMENT' : 'NEW DEPLOYMENT'}</span><h2>{editingModelId ? 'Model settings' : 'Add Azure model'}</h2></div>{editingModelId && <button type="button" className="icon-button small" title="Cancel editing" aria-label="Cancel editing" onClick={resetModelForm}><X size={16} /></button>}</div>
          <label className="settings-field"><span>Display name</span><input className="text-input" required maxLength={120} placeholder="Azure GPT-4.1 mini" value={modelForm.name} onChange={(event) => setModelForm((current) => ({ ...current, name: event.target.value }))} /></label>
          <label className="settings-field"><span>Model ID</span><input className="text-input" required maxLength={120} placeholder="gpt-4.1-mini" value={modelForm.model} onChange={(event) => setModelForm((current) => ({ ...current, model: event.target.value }))} /><small>Azure model family ID, not the deployment name.</small></label>
          <label className="settings-field"><span>Deployment name</span><input className="text-input" required maxLength={160} placeholder="The exact name from Azure AI Foundry" value={modelForm.deployment} onChange={(event) => setModelForm((current) => ({ ...current, deployment: event.target.value }))} /></label>
          <label className="settings-field"><span>Azure endpoint</span><input className="text-input" type="url" required placeholder="https://resource.services.ai.azure.com" value={modelForm.endpoint} onChange={(event) => setModelForm((current) => ({ ...current, endpoint: event.target.value }))} /></label>
          <label className="settings-field"><span>API version <small>Optional for Azure AI Foundry v1</small></span><input className="text-input" placeholder="2024-10-21" value={modelForm.api_version} onChange={(event) => setModelForm((current) => ({ ...current, api_version: event.target.value }))} /></label>
          <label className="settings-field"><span>Azure API key {editingModelId && <small>Leave blank to keep the saved key</small>}</span><input className="text-input" type="password" autoComplete="new-password" required={!editingModelId} placeholder={editingModelId ? 'Stored securely; enter a new key to rotate it' : 'Paste Azure OpenAI API key'} value={modelForm.api_key} onChange={(event) => setModelForm((current) => ({ ...current, api_key: event.target.value }))} /></label>
          <div className="model-form-actions"><button className="secondary-button" type="button" onClick={resetModelForm} disabled={!editingModelId && !modelForm.name && !modelForm.model}>Clear</button><button className="run-button" type="submit" disabled={savingModel || !backendReady}><Check size={14} /> {savingModel ? 'Saving securely' : editingModelId ? 'Save changes' : 'Add model'}</button></div>
        </form>
      </div>
    </main> : activeTab === 'Workflow' ? <main className="workspace">
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
        <div className="canvas-toolbar"><div className="canvas-title"><h2>Application workflow</h2><span className={`draft-badge ${hasUnsavedChanges ? 'unsaved' : ''}`}><span /> {hasUnsavedChanges ? 'UNSAVED' : 'SAVED'}</span></div><div className="canvas-actions"><button className="secondary-button" title="Import workflow JSON" onClick={() => importInputRef.current?.click()}><FileUp size={14} /> Import</button><input ref={importInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={importWorkflow} /><button className="secondary-button export-workflow-button" title="Export workflow JSON" onClick={exportWorkflow}><Download size={14} /> Export</button><button className="secondary-button" title="Delete selected connection" aria-label="Delete selected connection" onClick={deleteSelectedEdge} disabled={!selectedEdge}><Trash2 size={14} /></button>{hasUnsavedChanges && <button className="secondary-button discard-workflow-button" title="Discard unsaved changes" onClick={discardWorkflowDraft}><X size={14} /> Discard</button>}<button className="run-button save-workflow-button" onClick={saveWorkflow} disabled={!hasUnsavedChanges}><Save size={14} /> Save</button><button className="run-button" onClick={runWorkflow} disabled={runState === 'running' || hasUnsavedChanges} title={hasUnsavedChanges ? 'Save the draft before running agents' : 'Run the saved workflow'}><Zap size={15} /> {runState === 'running' ? 'Running' : runState === 'complete' ? 'Run again' : 'Run workflow'}</button></div></div>
        <div className="canvas-wrap" onDrop={onDrop} onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move' }}>
          <div className="canvas-caption"><span>MAIN FLOW</span><span className="caption-rule" /><span>{nodes.length} STEPS · REVIEW LOOP</span></div>
          <ReactFlow nodes={nodes} edges={edges} defaultEdgeOptions={{ markerEnd: { type: MarkerType.ArrowClosed } }} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={(_, node) => { setSelectedId(node.id); setSelectedEdgeId(null) }} onEdgeClick={(_, edge) => { setSelectedEdgeId(edge.id); setSelectedId(null) }} onPaneClick={() => { setSelectedId(null); setSelectedEdgeId(null) }} nodeTypes={nodeTypes} deleteKeyCode={['Backspace', 'Delete']} fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.25} maxZoom={1.4}>
            <Background color="#d8ddd8" gap={22} size={1} /><Controls position="bottom-left" showInteractive={false} /><MiniMap position="bottom-right" pannable zoomable nodeColor={(node) => ({ mint: '#77b9a0', coral: '#d9856e', blue: '#7197b9', yellow: '#c8a84d', lilac: '#a796c5' }[node.data.color as string] ?? '#aab2ac')} />
          </ReactFlow>
          {nodes.length === 0 && <div className="canvas-empty"><Layers2 size={23} /><strong>Start with an agent</strong><span>Drag one from the library to shape your workflow.</span></div>}
          <div className="canvas-bottom-note"><span className="keyboard-hint">SPACE</span> pan <span className="note-dot">·</span> scroll to zoom</div>
        </div>
        <div className="canvas-statusbar"><span><span className={hasUnsavedChanges ? 'status-amber' : 'status-green'} /> {hasUnsavedChanges ? 'Draft only · agents still use the saved workflow' : 'All changes saved'}</span><span className="statusbar-right"><GitBranch size={13} /> {edges.length} connections</span></div>
      </section>

      <aside className="inspector">
        <div className="inspector-tabs"><button className="selected"><Settings2 size={15} /> Configure</button><button><ClipboardList size={15} /> Context</button></div>
        {selectedEdge ? <>
          <div className="inspector-title"><span className="connection-icon"><GitBranch size={16} /></span><div><span className="eyebrow">CONNECTION</span><strong>{nodes.find((node) => node.id === selectedEdge.source)?.data.name ?? selectedEdge.source} to {nodes.find((node) => node.id === selectedEdge.target)?.data.name ?? selectedEdge.target}</strong></div><button className="icon-button small" title="Close connection" onClick={() => setSelectedEdgeId(null)}><X size={16} /></button></div>
          <div className="form-section"><label className="field-label" htmlFor="connection-label">Message on connection</label><input id="connection-label" className="text-input" value={typeof selectedEdge.label === 'string' ? selectedEdge.label : ''} onChange={(event) => updateSelectedEdgeLabel(event.target.value)} placeholder="Describe what this agent passes along" maxLength={120} /><span className="field-footnote">This message is included in the saved workflow and passed to the target agent. Configure this wire as an agent's success or failure route in its settings.</span></div>
          <div className="inspector-bottom"><span><span className={hasUnsavedChanges ? 'status-amber' : 'status-green'} /> {hasUnsavedChanges ? 'Unsaved connection' : 'Saved connection'}</span><button className="delete-agent" onClick={deleteSelectedEdge}><Trash2 size={13} /> Remove connection</button></div>
        </> : selectedNode ? <>
          <div className="inspector-title"><span className={`agent-icon ${selectedNode.data.color}`}><Settings2 size={16} /></span><div><span className="eyebrow">AGENT CONFIGURATION</span><strong>{selectedNode.data.name}</strong></div><button className="icon-button small" title="Close selection" onClick={() => setSelectedId(null)}><X size={16} /></button></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-name">Agent name</label><input id="agent-name" className="text-input" value={selectedNode.data.name} onChange={(event) => updateSelected('name', event.target.value)} /></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-kind">Role</label><div className="select-wrap"><select id="agent-kind" value={selectedNode.data.kind} onChange={(event) => { const preset = presets[event.target.value as AgentKind]; setNodes((current) => current.map((node) => node.id === selectedId ? { ...node, data: { ...preset, status: node.data.status } } : node)) }}>{Object.keys(presets).map((kind) => <option key={kind} value={kind}>{presets[kind as AgentKind].name}</option>)}</select><ChevronDown size={15} /></div></div>
          <div className="form-section"><div className="field-label">Outcome routing</div><label className="field-label" htmlFor="agent-success-wire">On success</label><div className="select-wrap"><select id="agent-success-wire" value={selectedNodeOutgoing.some((edge) => edge.id === selectedNode.data.successEdgeId) ? selectedNode.data.successEdgeId : ''} onChange={(event) => updateSelected('successEdgeId', event.target.value)}><option value="">Not configured</option>{selectedNodeOutgoing.map((edge) => <option key={edge.id} value={edge.id}>{nodes.find((node) => node.id === edge.target)?.data.name ?? edge.target}{typeof edge.label === 'string' && edge.label ? ` · ${edge.label}` : ''}</option>)}</select><ChevronDown size={15} /></div><label className="field-label" htmlFor="agent-failure-wire">On failure</label><div className="select-wrap"><select id="agent-failure-wire" value={selectedNodeOutgoing.some((edge) => edge.id === selectedNode.data.failureEdgeId) ? selectedNode.data.failureEdgeId : ''} onChange={(event) => updateSelected('failureEdgeId', event.target.value)}><option value="">Not configured</option>{selectedNodeOutgoing.map((edge) => <option key={edge.id} value={edge.id}>{nodes.find((node) => node.id === edge.target)?.data.name ?? edge.target}{typeof edge.label === 'string' && edge.label ? ` · ${edge.label}` : ''}</option>)}</select><ChevronDown size={15} /></div><label className="field-label" htmlFor="agent-max-retries">Maximum retries <span className="label-hint">0–20</span></label><input id="agent-max-retries" className="text-input" type="number" min={0} max={20} step={1} value={selectedNode.data.maxRetries ?? 5} onChange={(event) => { const parsed = Number.parseInt(event.target.value, 10); updateSelectedMaxRetries(Number.isFinite(parsed) ? Math.min(20, Math.max(0, parsed)) : 0) }} /><span className="field-footnote">Failure handoffs allowed before stopping. Set to 0 to stop on the first failure. The selected wire’s message is passed to its destination.</span></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-instructions">Instructions <span className="label-hint">PROMPT</span></label><textarea id="agent-instructions" className="instructions-input" value={selectedNode.data.instruction} onChange={(event) => updateSelected('instruction', event.target.value)} rows={7} /><span className="field-footnote">This agent receives upstream results and project context.</span></div>
          <div className="form-section"><label className="field-label" htmlFor="agent-model">Azure deployment</label><div className="select-wrap"><select id="agent-model" value={selectedNode.data.model} onChange={(event) => { const model = models.find((item) => item.id === event.target.value); updateSelected('model', event.target.value); updateSelected('modelLabel', model?.name ?? event.target.value) }} disabled={!models.some((model) => model.configured)}><option value={selectedNode.data.model}>{models.find((model) => model.id === selectedNode.data.model)?.name ?? selectedNode.data.modelLabel}</option>{models.filter((model) => model.configured && model.id !== selectedNode.data.model).map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select><ChevronDown size={15} /></div><span className="field-footnote">Set Azure key, endpoint, API version, and deployment in backend/.env.</span></div>
          <div className="form-section tools-section"><div className="field-label">Tools <span className="label-hint">{selectedNode.data.tools.length} ENABLED</span></div><div className="tool-chips">{['Read files', 'Search codebase', 'Edit files', 'Run commands', 'Run any command', 'Install dependencies'].map((tool) => <button className={`tool-chip ${selectedNode.data.tools.includes(tool) ? 'enabled' : ''}`} key={tool} onClick={() => updateSelected('tools', selectedNode.data.tools.includes(tool) ? selectedNode.data.tools.filter((item) => item !== tool) : [...selectedNode.data.tools, tool])}><span className="tool-check">{selectedNode.data.tools.includes(tool) && <Check size={11} />}</span>{tool}</button>)}</div>{selectedNode.data.tools.includes('Run commands') && <span className="tool-warning">Commands run on this computer. Enable only for projects you trust.</span>}{selectedNode.data.tools.includes('Run any command') && <span className="tool-warning">Allows any executable and arguments to run locally from this project directory. Not an operating-system sandbox; enable only for trusted projects.</span>}{selectedNode.data.tools.includes('Install dependencies') && <span className="tool-warning">Installs Python, Node, or .NET dependencies from project manifests. Package build hooks may execute.</span>}</div>
          <div className="inspector-bottom"><span><span className="status-green" /> Config saved</span><button className="delete-agent" onClick={() => { setNodes((current) => current.filter((node) => node.id !== selectedId)); setEdges((current) => current.filter((edge) => edge.source !== selectedId && edge.target !== selectedId)); setSelectedId(null) }}><X size={13} /> Remove agent</button></div>
        </> : <div className="inspector-empty"><Settings2 size={22} /><strong>Select an agent</strong><span>Choose a canvas node to edit its role, instructions, model, and tools.</span></div>}
      </aside>
    </main> : activeTab === 'Runs' ? <main className="runs-view">
      <div className="runs-heading"><div><span className="eyebrow">EXECUTION HISTORY</span><h1>Runs</h1><p>Open a run to inspect every agent, tool call, handoff, and result.</p></div><span className="runs-zero">{runHistory.length} RUNS</span></div>
      {historyLoading ? <div className="runs-empty"><Activity size={22} /><strong>Loading run history</strong></div> : historyError ? <div className="runs-empty"><strong>Could not load runs</strong><span>{historyError}</span></div> : runHistory.length === 0 ? <div className="runs-empty"><Activity size={24} /><strong>No workflow runs yet</strong><span>Start a saved workflow to build run history.</span><button className="secondary-button" onClick={() => setActiveTab('Workflow')}>Go to workflow <ArrowRight size={14} /></button></div> : <div className="run-history-list">{runHistory.map((run) => <button className="run-history-row" key={run.run_id} onClick={() => openHistoricalRun(run.run_id)}><span className={`run-history-status ${run.status}`} /><span className="run-history-main"><strong>{run.project_name} <i /> {run.requirement || 'Workflow run'}</strong><small>{new Date(run.started_at).toLocaleString()} · {run.agent_count} agents · {run.status}</small></span><span className="run-history-agents">{run.agents.map((agent) => agent.name).join(' · ')}</span><ArrowRight size={15} /></button>)}</div>}
    </main> : null}

    <section className="activity-drawer">
      <div className="activity-heading">
        <div className="activity-title"><span className={`activity-pulse ${runState === 'running' ? 'pulsing' : ''}`}><Activity size={15} /></span><strong>Execution monitor</strong><span className="activity-count">{nodes.length} agents</span></div>
        <div className="activity-tools"><span className={`run-indicator ${runState}`}>{runState === 'running' ? 'RUNNING' : runState === 'complete' ? 'COMPLETED' : runState === 'error' ? 'ERROR' : 'IDLE'}</span><button className="secondary-button full-run-button" onClick={openCurrentRun} disabled={!currentRunId}><Activity size={13} /> View full run</button></div>
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
    {runViewerOpen && <div className="run-viewer-backdrop" role="presentation" onClick={(event) => { if (event.target === event.currentTarget) setRunViewerOpen(false) }}><section className="run-viewer" role="dialog" aria-modal="true" aria-label="Full run details"><header className="run-viewer-header"><div><span className="eyebrow">FULL INTERACTION LOG</span><h1>{runViewerTitle}</h1><p>{runViewerDetail ? `${runViewerDetail.agent_count} agents · ${runViewerDetail.events.length} events` : `${nodes.length} agents · ${runViewerEvents.length} live events`}</p></div><div className="run-viewer-actions"><span className={`run-status-pill ${runViewerDetail?.status ?? runState}`}>{runViewerDetail?.status ?? runState}</span><button className="icon-button" title="Close run details" aria-label="Close run details" onClick={() => setRunViewerOpen(false)}><X size={19} /></button></div></header><div className="run-viewer-content">{runViewerLoading ? <div className="runs-empty"><Activity size={22} /><strong>Loading full run</strong></div> : runViewerEvents.length === 0 ? <div className="runs-empty"><Activity size={22} /><strong>No interaction events yet</strong><span>Start a workflow to view prompts, tool output, agent reports, and routes here.</span></div> : <div className="run-detail-timeline">{runViewerEvents.map((entry) => <article className={`run-detail-event ${entry.level}`} key={entry.id}><header><time>{entry.time}</time><span className="run-detail-agent">{entry.agent}</span><span className="run-detail-type">{entry.level.replaceAll('_', ' ')}</span></header><p>{entry.message}</p>{entry.detail && <details><summary>View full interaction</summary><pre>{entry.detail}</pre></details>}</article>)}<div ref={consoleEndRef} /></div>}</div></section></div>}
    {toast && <div className="toast"><Check size={15} />{toast}</div>}
    <footer className="app-footer"><span>THREADLINE <span className="footer-separator">/</span> WORKFLOW BUILDER</span><span>LOCAL-FIRST AGENT ORCHESTRATION <span className="footer-separator">·</span> PROTOTYPE</span></footer>
  </div>
}

function App() {
  return <ReactFlowProvider><WorkflowEditor /></ReactFlowProvider>
}

export default App