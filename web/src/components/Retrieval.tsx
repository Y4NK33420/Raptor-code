import { motion } from 'motion/react'
import type { RetrievalEvent } from '@/api/types'
import { Tip } from '@/components/ui/radix'
import { cn, layerColor, layerName } from '@/lib/utils'

/**
 * Where the context came from: one stacked bar, a segment per layer sized by its share of
 * retrieved context tokens. A 2px surface gap separates segments; a legend names them.
 */
export function LayerMix({ retrieval, className }: { retrieval: RetrievalEvent; className?: string }) {
  const tokens: Record<number, number> = {}
  const counts: Record<number, number> = {}
  for (const n of retrieval.nodes) {
    tokens[n.layer] = (tokens[n.layer] ?? 0) + n.tokens
    counts[n.layer] = (counts[n.layer] ?? 0) + 1
  }
  const total = Object.values(tokens).reduce((a, b) => a + b, 0) || 1
  const layers = Object.keys(tokens)
    .map(Number)
    .sort((a, b) => a - b)
  const summaryShare = layers.filter((l) => l > 0).reduce((a, l) => a + tokens[l], 0) / total
  return (
    <div className={className}>
      <div className="mb-1.5 flex items-baseline justify-between text-xs">
        <span className="font-medium text-ink-2">Context by layer</span>
        <span className="text-muted">
          <span className="num text-ink">{Math.round(summaryShare * 100)}%</span> from summaries ·{' '}
          <span className="num">{retrieval.context_tokens}</span> tokens
        </span>
      </div>
      <div className="flex h-3 gap-[2px] overflow-hidden rounded-[4px]" role="img" aria-label={layers.map((l) => `${layerName(l)} ${Math.round((100 * tokens[l]) / total)}%`).join(', ')}>
        {layers.map((l) => (
          <Tip key={l} content={`${layerName(l)}: ${counts[l]} nodes, ${tokens[l]} tokens (${Math.round((100 * tokens[l]) / total)}%)`}>
            <motion.div
              className="h-full first:rounded-l-[4px] last:rounded-r-[4px]"
              style={{ background: layerColor(l) }}
              initial={{ width: 0 }}
              animate={{ width: `${(100 * tokens[l]) / total}%` }}
              transition={{ type: 'spring', stiffness: 120, damping: 20 }}
            />
          </Tip>
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
        {layers.map((l) => (
          <span key={l} className="inline-flex items-center gap-1.5 text-[11px] text-ink-2">
            <span className="size-2 rounded-[2px]" style={{ background: layerColor(l) }} />
            {layerName(l)} <span className="num text-muted">×{counts[l]}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

export function RetrievedList({
  retrieval,
  onPick,
  selected,
  className,
}: {
  retrieval: RetrievalEvent
  onPick?: (id: number) => void
  selected?: number | null
  className?: string
}) {
  const max = Math.max(...retrieval.nodes.map((n) => n.score), 0.0001)
  return (
    <ol className={cn('space-y-1', className)} aria-label="Retrieved nodes">
      {retrieval.nodes.map((n, i) => (
        <li key={n.id}>
          <button
            type="button"
            onClick={() => onPick?.(n.id)}
            className={cn(
              'group grid w-full grid-cols-[18px_1fr_auto] items-start gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2',
              selected === n.id && 'bg-surface-2',
            )}
          >
            <span className="num pt-px text-[11px] text-muted">{i + 1}</span>
            <span className="min-w-0">
              <span className="flex items-center gap-1.5 text-[11px] text-muted">
                <span className="h-1.5 w-2.5 rounded-full" style={{ background: layerColor(n.layer) }} />
                {layerName(n.layer)} · #{n.id} · <span className="num">{n.tokens}t</span>
              </span>
              <span className="line-clamp-2 text-xs leading-[17px] text-ink-2">{n.preview}</span>
            </span>
            <span className="flex w-14 flex-col items-end gap-1 pt-0.5">
              <span className="num text-[11px] text-ink">{n.score.toFixed(3)}</span>
              <span className="h-1 w-full rounded-full bg-surface-3">
                <span className="block h-1 rounded-full bg-ink-2/60" style={{ width: `${(100 * n.score) / max}%` }} />
              </span>
            </span>
          </button>
        </li>
      ))}
    </ol>
  )
}

export function StreamingText({ text, active }: { text: string; active: boolean }) {
  return (
    <p className="text-[15px] leading-7 whitespace-pre-wrap text-ink">
      {text}
      {active && <span className="ml-0.5 inline-block h-4 w-[7px] translate-y-0.5 animate-pulse rounded-[2px] bg-accent" aria-hidden />}
    </p>
  )
}
