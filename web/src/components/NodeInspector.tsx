import { ArrowDown, ArrowUp, X } from 'lucide-react'
import { useNode } from '@/api/hooks'
import type { TreeData } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Badge, Skeleton } from '@/components/ui/primitives'
import { layerColor, layerName } from '@/lib/utils'

function Chips({
  ids,
  tree,
  onSelect,
  max = 30,
}: {
  ids: number[]
  tree: TreeData
  onSelect: (id: number) => void
  max?: number
}) {
  const layerOf = new Map(tree.nodes.map((n) => [n.id, n.layer]))
  return (
    <div className="flex flex-wrap gap-1">
      {ids.slice(0, max).map((id) => (
        <button
          key={id}
          type="button"
          onClick={() => onSelect(id)}
          className="num inline-flex items-center gap-1 rounded-md border border-line bg-surface-2 px-1.5 py-0.5 text-[11px] text-ink-2 hover:border-line-strong hover:text-ink"
        >
          <span className="size-1.5 rounded-full" style={{ background: layerColor(layerOf.get(id) ?? 0) }} />#{id}
        </button>
      ))}
      {ids.length > max && <span className="text-[11px] text-muted">+{ids.length - max} more</span>}
    </div>
  )
}

export function NodeInspector({
  indexId,
  nodeId,
  tree,
  onSelect,
  onClose,
}: {
  indexId: string
  nodeId: number
  tree: TreeData
  onSelect: (id: number) => void
  onClose: () => void
}) {
  const node = useNode(indexId, nodeId)
  const d = node.data
  const start = d?.metadata?.start as number | undefined
  const end = d?.metadata?.end as number | undefined
  return (
    <aside className="flex h-full flex-col" aria-label={`Node ${nodeId}`}>
      <div className="flex items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-4 rounded-full" style={{ background: layerColor(d?.layer ?? 0) }} />
          <h2 className="font-display text-lg font-semibold">Node {nodeId}</h2>
          {d && <Badge>{layerName(d.layer, tree.num_layers)}</Badge>}
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close inspector">
          <X />
        </Button>
      </div>
      <div className="scrollbar-thin flex-1 space-y-5 overflow-y-auto p-4">
        {!d ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-24" />
          </div>
        ) : (
          <>
            <dl className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <dt className="text-muted">tokens</dt>
                <dd className="num text-sm">{d.tokens}</dd>
              </div>
              <div>
                <dt className="text-muted">children</dt>
                <dd className="num text-sm">{d.children.length}</dd>
              </div>
              <div>
                <dt className="text-muted">parents</dt>
                <dd className="num text-sm">{d.parents.length}</dd>
              </div>
            </dl>
            <section>
              <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">
                {d.layer === 0 ? 'Chunk text' : 'Summary'}
              </h3>
              <p className="text-sm leading-6 whitespace-pre-wrap text-ink">{d.text}</p>
              {start !== undefined && end !== undefined && (
                <p className="num mt-2 text-[11px] text-muted">
                  source chars {start.toLocaleString()}–{end.toLocaleString()}
                </p>
              )}
            </section>
            {d.parents.length > 0 && (
              <section>
                <h3 className="mb-1.5 flex items-center gap-1 text-[11px] font-semibold tracking-wider text-muted uppercase">
                  <ArrowUp className="size-3" /> Summarised by
                </h3>
                <Chips ids={d.parents} tree={tree} onSelect={onSelect} />
                {d.parents.length > 1 && (
                  <p className="mt-1 text-[11px] text-muted">Soft clustering placed this node in several clusters.</p>
                )}
              </section>
            )}
            {d.children.length > 0 && (
              <section>
                <h3 className="mb-1.5 flex items-center gap-1 text-[11px] font-semibold tracking-wider text-muted uppercase">
                  <ArrowDown className="size-3" /> Summarises
                </h3>
                <Chips ids={d.children} tree={tree} onSelect={onSelect} />
              </section>
            )}
            {d.siblings.length > 0 && (
              <section>
                <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Same cluster</h3>
                <Chips ids={d.siblings} tree={tree} onSelect={onSelect} max={16} />
              </section>
            )}
          </>
        )}
      </div>
    </aside>
  )
}
