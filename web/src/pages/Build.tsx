import { useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Check, CircleAlert, Layers, Loader2 } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useReducer, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router'
import { urls } from '@/api/client'
import { type SseMessage, streamSse } from '@/api/sse'
import { PageHeader } from '@/components/layout/AppShell'
import { Button } from '@/components/ui/button'
import { Badge, Card, CardBody, CardHeader, CardTitle, ErrorNote, Swatch } from '@/components/ui/primitives'
import { type BuildState, buildPositions, initialBuild, reduceBuild, type Stage } from '@/lib/buildState'
import { cn, compact, layerColor, layerName } from '@/lib/utils'

const STEPS: { stage: Stage; label: string; detail: (s: BuildState) => string }[] = [
  { stage: 'chunking', label: 'Chunk', detail: (s) => (s.chunks.length ? `${s.chunks.length} chunks` : 'splitting sentences') },
  { stage: 'embedding', label: 'Embed', detail: (s) => `${s.cost.embedded_texts} texts embedded` },
  { stage: 'clustering', label: 'Cluster', detail: (s) => (s.layers.length ? `${s.expected} clusters over ${s.layers.length} layer${s.layers.length > 1 ? 's' : ''}` : 'UMAP + GMM') },
  { stage: 'summarizing', label: 'Summarise', detail: (s) => `${s.summarized} / ${s.expected} summaries` },
  { stage: 'complete', label: 'Saved', detail: (s) => (s.result ? `${s.result.num_nodes} nodes` : 'writing index') },
]
const ORDER: Stage[] = ['queued', 'chunking', 'embedding', 'clustering', 'summarizing', 'saving', 'complete']

function Stepper({ state }: { state: BuildState }) {
  const cur = ORDER.indexOf(state.stage === 'error' ? 'queued' : state.stage)
  return (
    <ol className="space-y-1" aria-label="Pipeline stages">
      {STEPS.map((step) => {
        const idx = ORDER.indexOf(step.stage)
        const done = cur > idx || state.stage === 'complete'
        const active = !done && (cur === idx || (step.stage === 'complete' && state.stage === 'saving'))
        return (
          <li key={step.stage} className="flex items-center gap-3 rounded-lg px-2 py-1.5" aria-current={active ? 'step' : undefined}>
            <span
              className={cn(
                'grid size-6 shrink-0 place-items-center rounded-full border text-[11px]',
                done && 'border-transparent bg-accent-fill text-accent-ink',
                active && 'border-accent text-accent',
                !done && !active && 'border-line text-muted',
              )}
            >
              {done ? <Check className="size-3.5" /> : active ? <Loader2 className="size-3.5 animate-spin" /> : null}
            </span>
            <div className="min-w-0">
              <div className={cn('text-sm', done || active ? 'text-ink' : 'text-muted')}>{step.label}</div>
              {(done || active) && <div className="num truncate text-[11px] text-muted">{step.detail(state)}</div>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

/** The tree growing: leaves along the bottom, clusters fanning up into summary nodes. */
function GrowingTree({ state }: { state: BuildState }) {
  const [hover, setHover] = useState<number | null>(null)
  const pos = useMemo(() => buildPositions(state), [state])
  const topLayer = Math.max(0, ...state.layers)
  const W = 1000
  const rowH = 150
  const H = 60 + Math.max(1, topLayer) * rowH + 40
  const y = (layer: number) => H - 40 - layer * rowH
  const parents = [...state.parents.values()]
  const leafW = Math.max(2, Math.min(18, (W * 0.9) / Math.max(state.chunks.length, 1) - 2))
  const hovered = hover !== null ? state.parents.get(hover) : undefined

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Tree being built">
        {/* edges */}
        {parents.map((p) =>
          p.children.map((c) => {
            const x1 = (pos.get(p.id) ?? 0.5) * W
            const x2 = (pos.get(c) ?? 0.5) * W
            const y1 = y(p.layer) + 10
            const y2 = y(p.layer - 1) - (p.layer - 1 === 0 ? 8 : 10)
            const my = (y1 + y2) / 2
            const lit = hover === p.id
            return (
              <motion.path
                key={`${p.id}-${c}`}
                d={`M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`}
                fill="none"
                stroke={lit ? layerColor(p.layer) : 'var(--line-strong)'}
                strokeWidth={lit ? 2 : 1}
                strokeOpacity={p.text ? (lit ? 1 : 0.7) : 0.35}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
              />
            )
          }),
        )}
        {/* leaves */}
        {state.chunks.map((c, i) => (
          <motion.rect
            key={c.id}
            x={(pos.get(c.id) ?? 0) * W - leafW / 2}
            y={y(0) - 8}
            width={leafW}
            height={16}
            rx={Math.min(4, leafW / 2)}
            fill={layerColor(0)}
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: Math.min(i * 0.008, 1.2) }}
            style={{ transformOrigin: 'center', transformBox: 'fill-box' }}
          >
            <title>{c.preview}</title>
          </motion.rect>
        ))}
        {/* summary nodes */}
        {parents.map((p) => {
          const cx = (pos.get(p.id) ?? 0.5) * W
          return (
            <g
              key={p.id}
              onMouseEnter={() => setHover(p.id)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: 'default' }}
            >
              <motion.circle
                cx={cx}
                cy={y(p.layer)}
                r={10}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                fill={p.text ? layerColor(p.layer) : 'var(--surface)'}
                stroke={layerColor(p.layer)}
                strokeWidth={2}
                strokeDasharray={p.text ? undefined : '3 3'}
                style={{ transformOrigin: 'center', transformBox: 'fill-box' }}
              />
              {p.text && (
                <motion.circle
                  cx={cx}
                  cy={y(p.layer)}
                  r={10}
                  fill="none"
                  stroke={layerColor(p.layer)}
                  initial={{ scale: 1, opacity: 0.8 }}
                  animate={{ scale: 2.4, opacity: 0 }}
                  transition={{ duration: 0.9 }}
                  style={{ transformOrigin: 'center', transformBox: 'fill-box' }}
                />
              )}
              <circle cx={cx} cy={y(p.layer)} r={16} fill="transparent" />
            </g>
          )
        })}
        {/* row labels */}
        {[0, ...state.layers].map((l) => (
          <text key={l} x={4} y={y(l) - 18} className="fill-[var(--muted)] text-[13px]" style={{ fontFamily: 'var(--font-mono)' }}>
            {layerName(l)}
          </text>
        ))}
      </svg>
      <AnimatePresence>
        {hovered?.text && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="pointer-events-none absolute top-2 right-2 left-2 rounded-xl border border-line bg-surface/95 p-3 text-sm text-ink shadow-card backdrop-blur sm:left-auto sm:w-80"
          >
            <div className="mb-1 flex items-center gap-2 text-xs text-muted">
              <Swatch color={layerColor(hovered.layer)} label={`${layerName(hovered.layer)} · node ${hovered.id}`} />
              <span>· {hovered.children.length} children</span>
            </div>
            {hovered.text}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function BuildPage() {
  const { jobId = '' } = useParams()
  const [params] = useSearchParams()
  const qc = useQueryClient()
  const [state, dispatch] = useReducer(reduceBuild, undefined, initialBuild)
  const [streamError, setStreamError] = useState<string | null>(null)
  const indexId = state.result?.index_id ?? params.get('index') ?? undefined

  useEffect(() => {
    const ctrl = new AbortController()
    let seen = -1
    const onEvent = (m: SseMessage) => {
      if (m.id !== undefined) {
        const n = Number(m.id)
        if (n <= seen) return // de-duplicate on reconnect
        seen = n
      }
      dispatch(m)
      if (m.event === 'complete') qc.invalidateQueries({ queryKey: ['indexes'] })
    }
    streamSse(urls.jobEvents(jobId), { signal: ctrl.signal }, onEvent).catch((e: unknown) => {
      if (!ctrl.signal.aborted) setStreamError(e instanceof Error ? e.message : String(e))
    })
    return () => ctrl.abort()
  }, [jobId, qc])

  const done = state.stage === 'complete'
  return (
    <div className="mx-auto max-w-7xl pb-10">
      <PageHeader
        eyebrow={
          <span className="flex items-center gap-2">
            Building
            {done ? <Badge tone="good">complete</Badge> : state.stage === 'error' ? <Badge tone="critical">failed</Badge> : <Badge tone="accent">live</Badge>}
          </span>
        }
        title={state.name ?? 'Preparing build…'}
        actions={
          done && indexId ? (
            <Button asChild variant="primary" size="lg">
              <Link to={`/indexes/${encodeURIComponent(indexId)}`}>
                Explore the tree <ArrowRight />
              </Link>
            </Button>
          ) : undefined
        }
      >
        {state.chars ? (
          <>
            <span className="num">{compact(state.chars)}</span> characters ·{' '}
            <span className="num">{state.t.toFixed(1)}s</span> elapsed
          </>
        ) : (
          'Waiting for the build worker…'
        )}
      </PageHeader>

      <div className="grid gap-4 px-4 sm:px-8 lg:grid-cols-[1fr_300px]">
        <Card className="dot-grid overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="size-4" /> Live tree
            </CardTitle>
            <div className="flex flex-wrap gap-3">
              {[0, ...state.layers].map((l) => (
                <Swatch key={l} color={layerColor(l)} label={layerName(l)} />
              ))}
            </div>
          </CardHeader>
          <CardBody>
            {state.chunks.length === 0 ? (
              <div className="grid h-64 place-items-center text-sm text-muted">
                <span className="flex items-center gap-2">
                  <Loader2 className="size-4 animate-spin" /> chunking the document…
                </span>
              </div>
            ) : (
              <GrowingTree state={state} />
            )}
          </CardBody>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Pipeline</CardTitle>
            </CardHeader>
            <CardBody>
              <Stepper state={state} />
              {state.expected > 0 && (
                <div className="mt-3">
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-3" role="progressbar" aria-valuenow={state.summarized} aria-valuemax={state.expected} aria-label="Summaries">
                    <motion.div
                      className="h-full bg-accent"
                      animate={{ width: `${(100 * state.summarized) / state.expected}%` }}
                    />
                  </div>
                </div>
              )}
              <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-line pt-3 text-xs">
                <div>
                  <dt className="text-muted">LLM calls</dt>
                  <dd className="num text-base text-ink">{state.cost.llm_calls}</dd>
                </div>
                <div>
                  <dt className="text-muted">embed calls</dt>
                  <dd className="num text-base text-ink">{state.cost.embed_calls}</dd>
                </div>
                <div>
                  <dt className="text-muted">texts</dt>
                  <dd className="num text-base text-ink">{state.cost.embedded_texts}</dd>
                </div>
              </dl>
            </CardBody>
          </Card>
          {(state.error || streamError) && (
            <ErrorNote error={state.error ?? streamError} />
          )}
          {state.stage === 'error' && <CircleAlert className="hidden" />}
          <Card>
            <CardHeader>
              <CardTitle>Summaries as they land</CardTitle>
            </CardHeader>
            <CardBody>
              <ul className="scrollbar-thin max-h-80 space-y-2 overflow-y-auto" aria-live="polite">
                <AnimatePresence initial={false}>
                  {state.feed.map((f) => (
                    <motion.li
                      key={f.id}
                      layout
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="rounded-lg border border-line bg-surface-2/60 p-2.5 text-xs leading-5 text-ink-2"
                    >
                      <div className="mb-0.5 flex items-center justify-between">
                        <Swatch color={layerColor(f.layer)} label={`L${f.layer} · #${f.id}`} />
                        <span className="num text-muted">{f.t.toFixed(1)}s</span>
                      </div>
                      <p className="line-clamp-3">{f.text}</p>
                    </motion.li>
                  ))}
                </AnimatePresence>
                {state.feed.length === 0 && <li className="text-xs text-muted">None yet.</li>}
              </ul>
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  )
}
