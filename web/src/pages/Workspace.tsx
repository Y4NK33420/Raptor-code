import { ArrowLeft, Map as MapIcon, MessageSquare, Network, Swords } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { useIndex, useProjection, useTree } from '@/api/hooks'
import type { RetrievalEvent } from '@/api/types'
import { ArenaPanel } from '@/components/ArenaPanel'
import { AskPanel } from '@/components/AskPanel'
import { EmbeddingMap } from '@/components/EmbeddingMap'
import { NodeInspector } from '@/components/NodeInspector'
import { Strata } from '@/components/Strata'
import { type Highlight, TreeView } from '@/components/TreeView'
import { Badge, Empty, ErrorNote, Skeleton } from '@/components/ui/primitives'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/radix'
import { cn } from '@/lib/utils'

const TABS = ['ask', 'tree', 'map', 'arena'] as const
type Tab = (typeof TABS)[number]

export default function WorkspacePage() {
  const { id = '', tab } = useParams()
  const navigate = useNavigate()
  const active: Tab = (TABS as readonly string[]).includes(tab ?? '') ? (tab as Tab) : 'ask'
  const index = useIndex(id)
  const tree = useTree(id)
  const projection = useProjection(id, active === 'map')
  const [selected, setSelected] = useState<number | null>(null)
  const [retrieval, setRetrieval] = useState<RetrievalEvent | undefined>()

  const highlights = useMemo(() => {
    const m = new Map<number, Highlight>()
    retrieval?.nodes.forEach((n, rank) => m.set(n.id, { rank, score: n.score }))
    return m
  }, [retrieval])
  const hlSet = useMemo(() => new Set(highlights.keys()), [highlights])

  if (index.isError)
    return (
      <div className="p-6">
        <ErrorNote error={index.error} />
      </div>
    )
  const info = index.data
  const goto = (t: string) => navigate(`/indexes/${encodeURIComponent(id)}/${t}`)

  let canvas: React.ReactNode
  if (tree.isError)
    canvas = (
      <div className="p-6">
        <ErrorNote error={tree.error} />
      </div>
    )
  else if (!tree.data) canvas = <Skeleton className="m-4 h-[70%]" />
  else if (active === 'map') {
    canvas = projection.data ? (
      <EmbeddingMap projection={projection.data} tree={tree.data} selected={selected} onSelect={setSelected} highlights={hlSet} />
    ) : projection.isError ? (
      <Empty title="No map available">
        <ErrorNote error={projection.error} />
      </Empty>
    ) : (
      <Skeleton className="m-4 h-[70%]" />
    )
  } else {
    canvas = (
      <TreeView tree={tree.data} selected={selected} onSelect={setSelected} highlights={highlights} focusHighlights={active === 'ask'} />
    )
  }
  const inspecting = selected !== null && !!tree.data

  return (
    <div className="flex h-[calc(100dvh-3.5rem)] min-h-0 flex-col lg:h-dvh">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 py-3 sm:px-6">
        <Link to="/" aria-label="Back to library" className="rounded-lg p-1.5 text-muted hover:bg-surface-2 hover:text-ink">
          <ArrowLeft className="size-4" />
        </Link>
        <div className="min-w-0 flex-1">
          <h1 className="truncate font-display text-xl font-semibold">{info?.name ?? <Skeleton className="h-6 w-48" />}</h1>
          {info && (
            <p className="flex flex-wrap items-center gap-x-2 text-xs text-muted">
              <span className="num">{info.num_nodes} nodes</span>·<span>{info.num_layers} layers</span>
              {info.origin === 'demo' && <Badge>demo</Badge>}
            </p>
          )}
        </div>
        {info && <Strata counts={info.layer_counts} className="hidden h-12 w-24 sm:flex" />}
        <Tabs value={active} onValueChange={goto}>
          <TabsList aria-label="Workspace views">
            <TabsTrigger value="ask">
              <MessageSquare className="size-3.5" /> Ask
            </TabsTrigger>
            <TabsTrigger value="tree">
              <Network className="size-3.5" /> Tree
            </TabsTrigger>
            <TabsTrigger value="map">
              <MapIcon className="size-3.5" /> Map
            </TabsTrigger>
            <TabsTrigger value="arena">
              <Swords className="size-3.5" /> Arena
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      <div className="min-h-0 flex-1">
        {active === 'arena' ? (
          <ArenaPanel
            indexId={id}
            setId={info?.source.set_id}
            onPick={(n) => {
              setSelected(n)
              goto('tree')
            }}
          />
        ) : (
          <div className="grid h-full min-h-0 grid-rows-[minmax(220px,42%)_minmax(0,1fr)] lg:grid-cols-[minmax(0,1fr)_380px] lg:grid-rows-1">
            <div className="relative min-h-0 border-b border-line lg:border-r lg:border-b-0">{canvas}</div>
            <div className="min-h-0 overflow-hidden bg-surface">
              {/* Ask stays mounted (it owns the stream) and is hidden while a node is inspected. */}
              {active === 'ask' && (
                <div className={cn('h-full', inspecting && 'hidden')}>
                  <AskPanel indexId={id} onRetrieval={setRetrieval} onPick={setSelected} selected={selected} />
                </div>
              )}
              {inspecting && tree.data ? (
                <NodeInspector indexId={id} nodeId={selected!} tree={tree.data} onSelect={setSelected} onClose={() => setSelected(null)} />
              ) : (
                active !== 'ask' && <Empty title="Pick a node">Click any node to read its text, parents and children.</Empty>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
