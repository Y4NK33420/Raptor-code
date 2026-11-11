import { Background, BackgroundVariant, Controls, type Edge, type Node, ReactFlow } from '@xyflow/react'
import { ChevronDown, ChevronRight, Code2, CornerDownLeft, FileCode2, Folder } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Navigate, useNavigate, useParams, useSearchParams } from 'react-router'
import { urls } from '@/api/client'
import { useCodeFiles, useCodeGraph, useCodeRepos, useCodeSymbol } from '@/api/hooks'
import type { CodeFile, CodeHit } from '@/api/types'
import { CodeViewer, type LineRange } from '@/components/CodeViewer'
import { StreamingText } from '@/components/Retrieval'
import { Button } from '@/components/ui/button'
import { Badge, Empty, ErrorNote, Skeleton, Textarea } from '@/components/ui/primitives'
import { Select, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/radix'
import { useCodeAsk } from '@/hooks/useStreams'
import { cn } from '@/lib/utils'

// ----------------------------------------------------------------- file tree
interface Dir {
  name: string
  path: string
  dirs: Map<string, Dir>
  files: CodeFile[]
}

function buildTree(files: CodeFile[]): Dir {
  const root: Dir = { name: '', path: '', dirs: new Map(), files: [] }
  for (const f of files) {
    const parts = f.path.split('/')
    let cur = root
    parts.slice(0, -1).forEach((p, i) => {
      const path = parts.slice(0, i + 1).join('/')
      if (!cur.dirs.has(p)) cur.dirs.set(p, { name: p, path, dirs: new Map(), files: [] })
      cur = cur.dirs.get(p)!
    })
    cur.files.push(f)
  }
  return root
}

function DirNode({
  dir,
  depth,
  current,
  onOpen,
  defaultOpen,
}: {
  dir: Dir
  depth: number
  current: string | null
  onOpen: (path: string, range?: LineRange) => void
  defaultOpen: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const pad = { paddingLeft: 8 + depth * 12 }
  return (
    <li>
      {dir.name && (
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          style={pad}
          className="flex w-full items-center gap-1.5 rounded-md py-1 pr-2 text-left text-[13px] text-ink-2 hover:bg-surface-2"
        >
          {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          <Folder className="size-3.5 text-muted" />
          {dir.name}
        </button>
      )}
      {(open || !dir.name) && (
        <ul>
          {[...dir.dirs.values()].map((d) => (
            <DirNode key={d.path} dir={d} depth={dir.name ? depth + 1 : depth} current={current} onOpen={onOpen} defaultOpen={depth < 1} />
          ))}
          {dir.files.map((f) => (
            <li key={f.path}>
              <button
                type="button"
                onClick={() => onOpen(f.path)}
                aria-current={current === f.path ? 'true' : undefined}
                style={{ paddingLeft: 8 + (dir.name ? depth + 1 : depth) * 12 + 18 }}
                className={cn(
                  'flex w-full items-center gap-1.5 rounded-md py-1 pr-2 text-left text-[13px] hover:bg-surface-2',
                  current === f.path ? 'bg-accent-soft text-ink' : 'text-ink-2',
                )}
              >
                <FileCode2 className="size-3.5 flex-none text-muted" />
                <span className="truncate">{f.path.split('/').pop()}</span>
                <span className="num ml-auto text-[10px] text-muted">{f.symbols.length}</span>
              </button>
              {current === f.path && f.symbols.length > 0 && (
                <ul className="mb-1">
                  {f.symbols.map((s) => (
                    <li key={s.id}>
                      <button
                        type="button"
                        onClick={() => onOpen(f.path, { start: s.start_line, end: s.end_line })}
                        style={{ paddingLeft: 8 + (dir.name ? depth + 1 : depth) * 12 + 34 }}
                        className="flex w-full items-center gap-1.5 rounded-md py-0.5 pr-2 text-left text-xs text-muted hover:bg-surface-2 hover:text-ink"
                      >
                        <span className="num w-8 flex-none text-[10px] uppercase">{s.kind.slice(0, 4)}</span>
                        <span className="truncate">{s.symbol}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  )
}

// ------------------------------------------------------------------ citations
const CITE = /([\w./-]+\.[A-Za-z0-9]+):(\d+)(?:[-–](\d+))?/g

/** Text with every `path:line` or `path:start-end` turned into a button that opens the code viewer. */
export function CitedText({ text, onOpen }: { text: string; onOpen: (path: string, range: LineRange) => void }) {
  const parts: React.ReactNode[] = []
  let last = 0
  for (const m of text.matchAll(CITE)) {
    const i = m.index ?? 0
    parts.push(text.slice(last, i))
    const start = Number(m[2])
    const end = m[3] ? Number(m[3]) : start
    parts.push(
      <button
        key={i}
        type="button"
        onClick={() => onOpen(m[1], { start, end })}
        className="num mx-0.5 rounded-md bg-accent-soft px-1.5 py-px text-[0.85em] text-accent underline-offset-2 hover:underline"
      >
        {m[0]}
      </button>,
    )
    last = i + m[0].length
  }
  parts.push(text.slice(last))
  return <p className="text-[15px] leading-7 whitespace-pre-wrap text-ink">{parts}</p>
}

function AskCode({ repoId, onOpen }: { repoId: string; onOpen: (path: string, range: LineRange) => void }) {
  const [q, setQ] = useState('')
  const ask = useCodeAsk(urls.codeAsk(repoId))
  const s = ask.state
  const busy = s.status === 'streaming'
  return (
    <div className="flex h-full min-h-0 flex-col">
      <form
        className="space-y-2 border-b border-line p-4"
        onSubmit={(e) => {
          e.preventDefault()
          if (q.trim() && !busy) void ask.run(q.trim())
        }}
      >
        <Textarea rows={3} aria-label="Question about the code" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Where is retrieval implemented?" />
        <div className="flex justify-end">
          <Button type="submit" variant="primary" size="sm" disabled={!q.trim() || busy}>
            <CornerDownLeft /> Ask
          </Button>
        </div>
      </form>
      <div className="scrollbar-thin min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {s.status === 'idle' && <p className="text-sm text-muted">Answers cite <code>path:line</code>; click a citation to open the code.</p>}
        {(s.text || busy) && (
          <section aria-live="polite" data-testid="code-answer">
            {busy ? <StreamingText text={s.text} active /> : <CitedText text={s.text} onOpen={onOpen} />}
          </section>
        )}
        {s.status === 'error' && <ErrorNote error={s.error} />}
        {s.citations.length > 0 && (
          <section>
            <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Sources</h3>
            <ul className="space-y-1">
              {s.citations.map((c, i) => (
                <li key={i}>
                  <HitButton hit={c} onOpen={onOpen} />
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}

function HitButton({ hit, onOpen }: { hit: CodeHit; onOpen: (path: string, range: LineRange) => void }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(hit.path, { start: hit.start_line, end: hit.end_line })}
      className="block w-full rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 text-left hover:border-line-strong"
    >
      <span className="flex items-center justify-between gap-2 text-[13px]">
        <span className="truncate font-medium">{hit.symbol}</span>
        <Badge>{hit.kind}</Badge>
      </span>
      <span className="num text-[11px] text-muted">
        {hit.path}:{hit.start_line}-{hit.end_line}
      </span>
    </button>
  )
}

// --------------------------------------------------------------------- symbol
function SymbolPanel({ repoId, nid, onOpen }: { repoId: string; nid: number | null; onOpen: (path: string, range: LineRange) => void }) {
  const sym = useCodeSymbol(repoId, nid)
  if (nid === null) return <Empty title="No symbol selected">Pick a symbol under a file, or a node in the graph.</Empty>
  if (sym.isLoading) return <Skeleton className="m-4 h-40" />
  if (sym.isError) return <div className="p-4"><ErrorNote error={sym.error} /></div>
  const d = sym.data!
  const groups: [string, CodeHit[]][] = [
    ['Callers', d.callers],
    ['Calls', d.callees],
    ['Contained in', d.parents],
    ['Contains', d.children],
  ]
  return (
    <div className="scrollbar-thin h-full space-y-4 overflow-y-auto p-4">
      <div>
        <h2 className="font-display text-lg font-semibold">{d.hit.symbol}</h2>
        <button type="button" onClick={() => onOpen(d.hit.path, { start: d.hit.start_line, end: d.hit.end_line })} className="num text-xs text-accent hover:underline">
          {d.hit.path}:{d.hit.start_line}-{d.hit.end_line}
        </button>
      </div>
      <pre className="scrollbar-thin max-h-48 overflow-auto rounded-lg bg-surface-2 p-2.5 text-[12px] leading-[18px] text-ink-2">{d.hit.text}</pre>
      {groups.map(([title, hits]) =>
        hits.length ? (
          <section key={title}>
            <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">{title}</h3>
            <ul className="space-y-1">
              {hits.slice(0, 12).map((h, i) => (
                <li key={i}>
                  <HitButton hit={h} onOpen={onOpen} />
                </li>
              ))}
            </ul>
          </section>
        ) : null,
      )}
    </div>
  )
}

// ---------------------------------------------------------------------- graph
const EDGE_COLOR: Record<string, string> = { calls: 'var(--layer-1)', imports: 'var(--layer-0)', inherits: 'var(--layer-2)' }

function GraphPanel({ repoId, onSelect }: { repoId: string; onSelect: (id: number) => void }) {
  const [kinds, setKinds] = useState('calls,imports')
  const g = useCodeGraph(repoId, kinds)
  const { nodes, edges } = useMemo(() => {
    const data = g.data
    if (!data) return { nodes: [] as Node[], edges: [] as Edge[] }
    const cols = new Map<string, number>()
    const rows = new Map<string, number>()
    const ns: Node[] = data.nodes.map((n) => {
      if (!cols.has(n.path)) cols.set(n.path, cols.size)
      const r = rows.get(n.path) ?? 0
      rows.set(n.path, r + 1)
      return {
        id: String(n.id),
        position: { x: cols.get(n.path)! * 230, y: r * 64 },
        data: { label: n.label, nid: n.id },
        style: {
          width: 190,
          fontSize: 12,
          padding: 6,
          background: 'var(--surface)',
          color: 'var(--ink)',
          border: `1.5px solid var(--layer-${Math.min(n.layer, 5)})`,
          borderRadius: 10,
        },
      }
    })
    const es: Edge[] = data.edges.map((e, i) => ({
      id: `${i}`,
      source: String(e.src),
      target: String(e.dst),
      style: { stroke: EDGE_COLOR[e.kind] ?? 'var(--line-strong)', strokeWidth: 1.3 },
      markerEnd: { type: 'arrowclosed' as never, color: EDGE_COLOR[e.kind] ?? 'var(--line-strong)' },
    }))
    return { nodes: ns, edges: es }
  }, [g.data])
  return (
    <div className="flex h-full min-h-[280px] flex-col">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <Select
          label="Edge kinds"
          className="h-8 w-44"
          value={kinds}
          onValueChange={setKinds}
          options={[
            { value: 'calls,imports', label: 'Calls + imports' },
            { value: 'calls', label: 'Calls' },
            { value: 'imports', label: 'Imports' },
            { value: 'calls,imports,inherits', label: 'All edges' },
          ]}
        />
        <span className="num text-xs text-muted">
          {nodes.length} nodes · {edges.length} edges
        </span>
      </div>
      <div className="min-h-0 flex-1">
        {g.isError ? (
          <ErrorNote error={g.error} />
        ) : g.isLoading ? (
          <Skeleton className="m-3 h-[80%]" />
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            minZoom={0.1}
            nodesConnectable={false}
            proOptions={{ hideAttribution: true }}
            onNodeClick={(_, n) => onSelect(Number((n.data as { nid: number }).nid))}
            aria-label="Call and import graph"
          >
            <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="var(--dot)" />
            <Controls showInteractive={false} />
          </ReactFlow>
        )}
      </div>
    </div>
  )
}

// ----------------------------------------------------------------------- page
export default function CodePage() {
  const { repoId } = useParams()
  const navigate = useNavigate()
  const repos = useCodeRepos()
  const [sp, setSp] = useSearchParams()
  const files = useCodeFiles(repoId)
  const [panel, setPanel] = useState('ask')
  const [symbolId, setSymbolId] = useState<number | null>(null)
  const path = sp.get('path')
  const range: LineRange | null = sp.get('l')
    ? { start: Number(sp.get('l')!.split('-')[0]), end: Number(sp.get('l')!.split('-')[1] ?? sp.get('l')!.split('-')[0]) }
    : null
  const tree = useMemo(() => buildTree(files.data ?? []), [files.data])

  const open = (p: string, r?: LineRange) => {
    const next = new URLSearchParams()
    next.set('path', p)
    if (r) next.set('l', `${r.start}-${r.end}`)
    setSp(next)
    if (r && files.data) {
      const f = files.data.find((x) => x.path === p)
      const s = f?.symbols.find((x) => x.start_line === r.start)
      if (s) {
        setSymbolId(s.id)
        setPanel('symbol')
      }
    }
  }

  if (repos.isLoading) return <Skeleton className="m-6 h-[60vh]" />
  if (repos.isError) return <div className="p-6"><ErrorNote error={repos.error} /></div>
  const list = repos.data ?? []
  if (!list.length)
    return (
      <Empty icon={<Code2 />} title="No code indexes yet" className="min-h-[60vh]">
        Index a repository with <code className="rounded bg-surface-2 px-1">raptor index &lt;path&gt;</code> and it will show up here.
      </Empty>
    )
  const repo = list.find((r) => r.id === repoId) ?? list[0]
  if (!repoId) return <Navigate to={`/code/${encodeURIComponent(repo.id)}`} replace />

  return (
    <div className="flex h-[calc(100dvh-3.5rem)] min-h-0 flex-col lg:h-dvh">
      <header className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3 sm:px-6">
        <div className="min-w-0 flex-1">
          <h1 className="font-display text-xl font-semibold">Code Explorer</h1>
          <p className="num truncate text-xs text-muted">
            {repo.n_files} files · {repo.n_symbols} symbols · {repo.num_layers} layers
            {repo.stub && ' · stub index'}
          </p>
        </div>
        {list.length > 1 && (
          <Select
            label="Repository"
            className="w-56"
            value={repo.id}
            onValueChange={(v) => navigate(`/code/${encodeURIComponent(v)}`)}
            options={list.map((r) => ({ value: r.id, label: r.name }))}
          />
        )}
      </header>
      <div className="grid min-h-0 flex-1 grid-rows-[160px_minmax(260px,1fr)_minmax(320px,1fr)] lg:grid-cols-[250px_minmax(0,1fr)_400px] lg:grid-rows-1">
        <nav aria-label="Files" className="scrollbar-thin overflow-y-auto border-b border-line p-2 lg:border-r lg:border-b-0">
          {files.isLoading ? (
            <Skeleton className="h-40" />
          ) : files.isError ? (
            <ErrorNote error={files.error} />
          ) : (
            <ul>
              <DirNode dir={tree} depth={0} current={path} onOpen={open} defaultOpen />
            </ul>
          )}
        </nav>
        <section aria-label="Source" className="min-h-0 border-b border-line lg:border-r lg:border-b-0">
          {path ? (
            <div className="flex h-full flex-col">
              <div className="num flex-none truncate border-b border-line px-4 py-2 text-xs text-ink-2">{path}</div>
              <div className="min-h-0 flex-1">
                <CodeViewer repoId={repo.id} path={path} range={range} />
              </div>
            </div>
          ) : (
            <Empty icon={<FileCode2 />} title="Choose a file">Pick a file on the left, or ask a question and follow its citations.</Empty>
          )}
        </section>
        <Tabs value={panel} onValueChange={setPanel} className="flex min-h-0 flex-col">
          <TabsList aria-label="Code panels" className="m-2 self-start">
            <TabsTrigger value="ask">Ask</TabsTrigger>
            <TabsTrigger value="symbol">Symbol</TabsTrigger>
            <TabsTrigger value="graph">Graph</TabsTrigger>
          </TabsList>
          <TabsContent value="ask" className="min-h-0 flex-1">
            <AskCode repoId={repo.id} onOpen={open} />
          </TabsContent>
          <TabsContent value="symbol" className="min-h-0 flex-1">
            <SymbolPanel repoId={repo.id} nid={symbolId} onOpen={open} />
          </TabsContent>
          <TabsContent value="graph" className="min-h-0 flex-1">
            <GraphPanel
              repoId={repo.id}
              onSelect={(id) => {
                setSymbolId(id)
                setPanel('symbol')
              }}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
