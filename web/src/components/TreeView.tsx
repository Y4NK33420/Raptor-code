import {
  Background,
  BackgroundVariant,
  Controls,
  type Edge,
  Handle,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react'
import { memo, useEffect, useMemo } from 'react'
import type { TreeData } from '@/api/types'
import { layoutTree } from '@/lib/treeLayout'
import { cn, layerColor } from '@/lib/utils'

export interface Highlight {
  rank: number // 0 = best
  score: number
  side?: 'flat' | 'raptor'
}

type RNodeData = {
  nid: number
  layer: number
  preview: string
  tokens: number
  nChildren: number
  leaf: boolean
  selected: boolean
  highlight?: Highlight
  dimmed: boolean
}

const RaptorNode = memo(function RaptorNode({ data }: NodeProps<Node<RNodeData>>) {
  const color = layerColor(data.layer)
  return (
    <div
      className={cn(
        'relative rounded-xl border bg-surface text-left shadow-card transition-[opacity,box-shadow,transform] duration-300',
        data.leaf ? 'w-[150px] p-2' : 'w-[210px] p-2.5',
        data.selected ? 'border-accent ring-2 ring-accent/50' : 'border-line',
        data.dimmed && 'opacity-30',
        data.highlight && 'scale-[1.04]',
      )}
      style={
        data.highlight
          ? { boxShadow: `0 0 0 2px ${color}, 0 0 22px -2px ${color}` }
          : undefined
      }
    >
      <Handle type="target" position={Position.Bottom} className="!size-1.5 !border-0 !bg-[var(--line-strong)]" />
      <Handle type="source" position={Position.Top} className="!size-1.5 !border-0 !bg-[var(--line-strong)]" />
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-3 rounded-full" style={{ background: color }} />
          <span className="num text-[10px] text-muted">#{data.nid}</span>
        </span>
        {data.highlight ? (
          <span
            className="num rounded-full px-1.5 text-[10px] font-semibold"
            style={{ background: color, color: 'white' }}
            title={`rank ${data.highlight.rank + 1}, similarity ${data.highlight.score.toFixed(3)}`}
          >
            {data.highlight.score.toFixed(2)}
          </span>
        ) : (
          <span className="num text-[10px] text-muted">{data.leaf ? `${data.tokens}t` : `${data.nChildren}↓`}</span>
        )}
      </div>
      <p className={cn('text-[11px] leading-[15px] text-ink-2', data.leaf ? 'line-clamp-3' : 'line-clamp-4')}>
        {data.preview}
      </p>
    </div>
  )
})

const nodeTypes = { raptor: RaptorNode }

function Inner({
  tree,
  selected,
  onSelect,
  highlights,
  focusHighlights,
  compact,
}: TreeViewProps) {
  const rf = useReactFlow()
  const layout = useMemo(
    () => layoutTree(tree.nodes, { nodeWidth: compact ? 160 : 210, gapX: compact ? 14 : 24, rowHeight: compact ? 170 : 200 }),
    [tree, compact],
  )
  const anyHL = !!highlights && highlights.size > 0

  const nodes: Node<RNodeData>[] = useMemo(
    () =>
      tree.nodes.map((n) => {
        const p = layout.get(n.id)!
        const leaf = n.layer === 0
        return {
          id: String(n.id),
          type: 'raptor',
          position: { x: p.x + (leaf ? 30 : 0), y: p.y },
          data: {
            nid: n.id,
            layer: n.layer,
            preview: n.preview,
            tokens: n.tokens,
            nChildren: n.children.length,
            leaf,
            selected: selected === n.id,
            highlight: highlights?.get(n.id),
            dimmed: anyHL && !highlights?.has(n.id) && selected !== n.id,
          },
          draggable: false,
        }
      }),
    [tree, layout, selected, highlights, anyHL],
  )

  const edges: Edge[] = useMemo(() => {
    const sel = selected
    const layerOf = new Map(tree.nodes.map((n) => [n.id, n.layer]))
    return tree.edges.map((e) => {
      const lit =
        (sel !== null && sel !== undefined && (e.source === sel || e.target === sel)) ||
        (anyHL && highlights!.has(e.source) && highlights!.has(e.target))
      const parent = { layer: layerOf.get(e.source) }
      return {
        id: `${e.source}-${e.target}`,
        source: String(e.source),
        target: String(e.target),
        type: 'default',
        style: {
          stroke: lit ? layerColor(parent?.layer ?? 1) : 'var(--line-strong)',
          strokeWidth: lit ? 2 : 1,
          opacity: anyHL && !lit ? 0.25 : 0.8,
        },
        animated: !!lit && anyHL,
      }
    })
  }, [tree, selected, highlights, anyHL])

  // Zoom to retrieved nodes when a new retrieval lands.
  useEffect(() => {
    if (!focusHighlights || !highlights || highlights.size === 0) return
    const ids = [...highlights.keys()].map(String)
    const t = setTimeout(() => rf.fitView({ nodes: ids.map((id) => ({ id })), padding: 0.25, duration: 600, maxZoom: 1.1 }), 50)
    return () => clearTimeout(t)
  }, [highlights, focusHighlights, rf])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.12 }}
      minZoom={0.05}
      maxZoom={2}
      nodesConnectable={false}
      elementsSelectable
      onNodeClick={(_, n) => onSelect?.(Number(n.id))}
      onPaneClick={() => onSelect?.(null)}
      proOptions={{ hideAttribution: true }}
      aria-label="Tree of summary and leaf nodes"
    >
      <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="var(--dot)" />
      <Controls showInteractive={false} position="bottom-right" />
    </ReactFlow>
  )
}

export interface TreeViewProps {
  tree: TreeData
  selected?: number | null
  onSelect?: (id: number | null) => void
  highlights?: Map<number, Highlight>
  focusHighlights?: boolean
  compact?: boolean
  className?: string
}

export function TreeView(props: TreeViewProps) {
  return (
    <div className={cn('h-full w-full', props.className)} data-testid="tree-view">
      <ReactFlowProvider>
        <Inner {...props} />
      </ReactFlowProvider>
    </div>
  )
}
