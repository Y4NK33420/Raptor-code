import { FlaskConical } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useRunCompare, useRuns } from '@/api/hooks'
import type { CompareQuestion, RunSummary } from '@/api/types'
import { Badge, Card, CardBody, CardHeader, CardTitle, Empty, ErrorNote, Skeleton, Swatch } from '@/components/ui/primitives'
import { Segmented, Select } from '@/components/ui/radix'
import { defaultBaseline, factorEffects } from '@/lib/effects'
import { cn, formatP, layerColor, pct, signedPts } from '@/lib/utils'

/** Accuracy bar with a 95% CI whisker on a shared 0-100% axis. */
export function AccuracyBar({ run, highlight }: { run: RunSummary; highlight?: boolean }) {
  const [lo, hi] = run.ci95?.length === 2 ? run.ci95 : [run.accuracy, run.accuracy]
  return (
    <div
      className="relative h-5 w-full"
      role="img"
      aria-label={`Accuracy ${pct(run.accuracy)}, 95% interval ${pct(lo)} to ${pct(hi)}`}
    >
      <div className="absolute inset-y-1 left-0 w-full rounded-full bg-surface-3" />
      <div
        className={cn('absolute inset-y-1 left-0 rounded-full', run.method === 'flat' ? 'bg-flat' : 'bg-accent-fill')}
        style={{ width: `${run.accuracy * 100}%` }}
      />
      <div className="absolute top-1/2 h-px bg-ink" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
      <div className="absolute top-[5px] h-2.5 w-px bg-ink" style={{ left: `${lo * 100}%` }} />
      <div className="absolute top-[5px] h-2.5 w-px bg-ink" style={{ left: `${hi * 100}%` }} />
      {highlight && <div className="absolute -inset-x-1 inset-y-0 rounded-md ring-2 ring-accent/50" />}
    </div>
  )
}

function LayerShare({ share }: { share: Record<string, number> }) {
  const layers = Object.keys(share).map(Number).sort((a, b) => a - b)
  if (!layers.length) return <span className="text-muted">—</span>
  return (
    <div className="flex h-2 w-24 gap-px overflow-hidden rounded-full" title={layers.map((l) => `L${l} ${pct(share[l], 0)}`).join(' · ')}>
      {layers.map((l) => (
        <span key={l} style={{ width: `${share[l] * 100}%`, background: layerColor(l) }} />
      ))}
    </div>
  )
}

function RunsTable({
  runs,
  baseline,
  other,
  onBaseline,
  onOther,
}: {
  runs: RunSummary[]
  baseline: string
  other: string | null
  onBaseline: (id: string) => void
  onOther: (id: string) => void
}) {
  const sorted = useMemo(() => [...runs].sort((a, b) => b.accuracy - a.accuracy), [runs])
  return (
    <Card>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
        <span className="text-xs text-muted">sorted by accuracy · whiskers show the 95% interval</span>
      </CardHeader>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-[11px] tracking-wider text-muted uppercase">
              <th className="px-4 py-2 font-semibold">Run</th>
              <th className="px-2 py-2 font-semibold">Method</th>
              <th className="w-56 px-2 py-2 font-semibold">Accuracy</th>
              <th className="px-2 py-2 text-right font-semibold">Hard</th>
              <th className="px-2 py-2 font-semibold">Context from</th>
              <th className="px-2 py-2 text-right font-semibold">n</th>
              <th className="px-4 py-2 text-right font-semibold">Compare</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.run_id} className={cn('border-b border-line last:border-0', r.run_id === baseline && 'bg-accent-soft/40')}>
                <td className="max-w-[220px] px-4 py-2.5">
                  <div className="truncate font-medium">{r.name ?? r.run_id}</div>
                  <div className="num truncate text-[11px] text-muted">{r.run_id}</div>
                </td>
                <td className="px-2">
                  <Badge tone={r.method === 'flat' ? 'neutral' : 'accent'}>{r.method ?? 'raptor'}</Badge>
                </td>
                <td className="px-2">
                  <div className="flex items-center gap-2">
                    <div className="w-32 flex-none">
                      <AccuracyBar run={r} />
                    </div>
                    <span className="num font-medium">{pct(r.accuracy)}</span>
                  </div>
                </td>
                <td className="num px-2 text-right text-ink-2">{pct(r.accuracy_hard)}</td>
                <td className="px-2">
                  <LayerShare share={r.layer_share ?? {}} />
                </td>
                <td className="num px-2 text-right text-ink-2">{r.n_questions}</td>
                <td className="px-4 text-right">
                  <div className="inline-flex gap-1">
                    <button
                      type="button"
                      onClick={() => onBaseline(r.run_id)}
                      aria-pressed={r.run_id === baseline}
                      className={cn(
                        'rounded-md border px-2 py-0.5 text-[11px] font-medium',
                        r.run_id === baseline ? 'border-accent bg-accent-soft text-accent' : 'border-line text-ink-2 hover:bg-surface-2',
                      )}
                    >
                      baseline
                    </button>
                    <button
                      type="button"
                      onClick={() => onOther(r.run_id)}
                      disabled={r.run_id === baseline}
                      aria-pressed={r.run_id === other}
                      className={cn(
                        'rounded-md border px-2 py-0.5 text-[11px] font-medium disabled:opacity-40',
                        r.run_id === other ? 'border-accent bg-accent-soft text-accent' : 'border-line text-ink-2 hover:bg-surface-2',
                      )}
                    >
                      vs
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function Effects({ runs, baseline, onPick }: { runs: RunSummary[]; baseline: RunSummary; onPick: (id: string) => void }) {
  const fx = factorEffects(runs, baseline)
  const max = Math.max(0.05, ...fx.map((f) => Math.abs(f.delta)))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Per-factor effects</CardTitle>
        <span className="text-xs text-muted">one change from baseline</span>
      </CardHeader>
      <CardBody>
        {fx.length === 0 ? (
          <p className="text-sm text-muted">No run differs from the baseline in exactly one factor.</p>
        ) : (
          <ul className="space-y-1.5">
            {fx.map((f) => (
              <li key={f.runId}>
                <button
                  type="button"
                  onClick={() => onPick(f.runId)}
                  className="grid w-full grid-cols-[88px_minmax(0,1fr)_minmax(110px,1.4fr)_64px] items-center gap-2 rounded-lg px-1 py-1 text-left text-xs hover:bg-surface-2"
                >
                  <span className="text-muted capitalize">{f.factor}</span>
                  <span className="truncate text-ink">{f.value}</span>
                  <span className="relative block h-3" aria-hidden>
                    <span className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
                    <span
                      className={cn('absolute inset-y-0.5 rounded-sm', f.delta >= 0 ? 'bg-good' : 'bg-critical')}
                      style={
                        f.delta >= 0
                          ? { left: '50%', width: `${(f.delta / max) * 50}%` }
                          : { right: '50%', width: `${(-f.delta / max) * 50}%` }
                      }
                    />
                  </span>
                  <span className={cn('num text-right font-medium', f.delta >= 0 ? 'text-good' : 'text-critical')}>
                    {signedPts(f.delta, 0)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </CardBody>
    </Card>
  )
}

const STATUS_TONE = { fixed: 'good', broken: 'critical', both_right: 'neutral', both_wrong: 'neutral' } as const

function QuestionRow({ q }: { q: CompareQuestion }) {
  const label = (p: number | null | undefined) => (p === null || p === undefined ? '—' : `${p}`)
  return (
    <li className="space-y-1 border-b border-line px-4 py-3 last:border-0">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-ink">{q.question ?? q.question_id}</p>
        <Badge tone={STATUS_TONE[q.status]} className="flex-none">
          {q.status.replace('_', ' ')}
        </Badge>
      </div>
      <p className="num text-[11px] text-muted">
        gold {label(q.gold)} · baseline picked {label(q.a_predicted)} · other picked {label(q.b_predicted)}
        {q.difficult && ' · hard'}
      </p>
      {q.options && q.gold !== null && q.gold !== undefined && (
        <p className="text-xs text-ink-2">Answer: {q.options[q.gold - 1]}</p>
      )}
      {q.b_rationale && <p className="text-xs text-muted italic">“{q.b_rationale}”</p>}
    </li>
  )
}

function Comparison({ a, b, runs }: { a: string; b: string; runs: RunSummary[] }) {
  const cmp = useRunCompare(a, b)
  const [filter, setFilter] = useState<'changed' | 'fixed' | 'broken' | 'all'>('changed')
  const name = (id: string) => runs.find((r) => r.run_id === id)?.name ?? id
  if (cmp.isLoading) return <Skeleton className="h-64" />
  if (cmp.isError) return <ErrorNote error={cmp.error} />
  const c = cmp.data!
  const shown = (c.questions ?? []).filter((q) =>
    filter === 'all' ? true : filter === 'changed' ? q.status === 'fixed' || q.status === 'broken' : q.status === filter,
  )
  const sig = c.mcnemar_p < 0.05
  const [lo, hi] = c.delta_ci95?.length === 2 ? c.delta_ci95 : [c.delta, c.delta]
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {name(b)} <span className="font-sans text-sm font-normal text-muted">vs baseline {name(a)}</span>
        </CardTitle>
      </CardHeader>
      <CardBody className="space-y-5">
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <dt className="text-xs text-muted">Accuracy change</dt>
            <dd className={cn('num font-display text-2xl font-semibold', c.delta >= 0 ? 'text-good' : 'text-critical')}>{signedPts(c.delta)}</dd>
            <dd className="num text-[11px] text-muted">
              95% CI {signedPts(lo)} to {signedPts(hi)}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">McNemar</dt>
            <dd className="num font-display text-2xl font-semibold">{formatP(c.mcnemar_p)}</dd>
            <dd className="text-[11px] text-muted">{sig ? 'unlikely to be noise' : 'could be noise'}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Fixed</dt>
            <dd className="num font-display text-2xl font-semibold text-good">{c.fixed}</dd>
            <dd className="text-[11px] text-muted">wrong → right</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Broken</dt>
            <dd className="num font-display text-2xl font-semibold text-critical">{c.broken}</dd>
            <dd className="text-[11px] text-muted">right → wrong</dd>
          </div>
        </dl>
        <div
          className="flex h-3 overflow-hidden rounded-full"
          role="img"
          aria-label={`${c.both_right} both right, ${c.fixed} fixed, ${c.broken} broken, ${c.both_wrong} both wrong`}
        >
          {[
            [c.both_right, 'var(--surface-3)'],
            [c.fixed, 'var(--good)'],
            [c.broken, 'var(--critical)'],
            [c.both_wrong, 'var(--line-strong)'],
          ].map(([n, col], i) => (
            <span key={i} style={{ width: `${(Number(n) / Math.max(1, c.n_paired)) * 100}%`, background: String(col) }} />
          ))}
        </div>
        <p className="text-[11px] text-muted">Paired on {c.n_paired} questions. With this few questions, differences under about 10 points are hard to trust.</p>
        <div>
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">Question diff</h3>
            <Segmented
              label="Filter questions"
              value={filter}
              onChange={setFilter}
              options={[
                { value: 'changed', label: 'Changed' },
                { value: 'fixed', label: `Fixed ${c.fixed}` },
                { value: 'broken', label: `Broken ${c.broken}` },
                { value: 'all', label: 'All' },
              ]}
            />
          </div>
          <ul className="rounded-xl border border-line">
            {shown.length ? shown.map((q) => <QuestionRow key={q.question_id} q={q} />) : <li className="p-4 text-sm text-muted">Nothing here.</li>}
          </ul>
        </div>
      </CardBody>
    </Card>
  )
}

export default function ExperimentsPage() {
  const runs = useRuns()
  const [baselineId, setBaselineId] = useState<string | null>(null)
  const [otherId, setOtherId] = useState<string | null>(null)

  if (runs.isLoading)
    return (
      <div className="space-y-4 p-6">
        <Skeleton className="h-9 w-56" />
        <Skeleton className="h-64" />
      </div>
    )
  if (runs.isError) return <div className="p-6"><ErrorNote error={runs.error} /></div>
  const list = runs.data ?? []
  if (!list.length)
    return (
      <Empty icon={<FlaskConical />} title="No experiment runs yet" className="min-h-[60vh]">
        Run <code className="rounded bg-surface-2 px-1">raptor eval run configs/exp/baseline.yaml</code> and its results will appear here.
      </Empty>
    )
  const baseline = list.find((r) => r.run_id === baselineId) ?? defaultBaseline(list)!
  const other = otherId && otherId !== baseline.run_id ? otherId : null

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-4 sm:p-8">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-3xl font-semibold">Experiments</h1>
          <p className="text-sm text-muted">What changes accuracy, and by how much, on QuALITY.</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">Baseline</span>
          <Select
            label="Baseline run"
            className="w-64"
            value={baseline.run_id}
            onValueChange={setBaselineId}
            options={list.map((r) => ({ value: r.run_id, label: r.name ?? r.run_id, hint: pct(r.accuracy) }))}
          />
        </div>
      </header>
      <div className="flex gap-4">
        <Swatch color="var(--accent-fill)" label="RAPTOR" />
        <Swatch color="var(--flat)" label="Flat RAG" />
      </div>
      <RunsTable runs={list} baseline={baseline.run_id} other={other} onBaseline={setBaselineId} onOther={setOtherId} />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Effects runs={list} baseline={baseline} onPick={setOtherId} />
        {other ? (
          <Comparison a={baseline.run_id} b={other} runs={list} />
        ) : (
          <Card>
            <Empty title="Pick a run to compare">Choose “vs” on any row, or click a factor effect, to see McNemar and the questions it fixed or broke.</Empty>
          </Card>
        )}
      </div>
    </div>
  )
}
