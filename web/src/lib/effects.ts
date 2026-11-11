import type { RunSummary } from '@/api/types'

export const FACTORS = ['chunking', 'clustering', 'retrieval', 'embedding'] as const
export type Factor = (typeof FACTORS)[number]

export interface FactorEffect {
  factor: Factor
  value: string
  runId: string
  name: string
  accuracy: number
  delta: number // vs baseline, in accuracy units
}

/**
 * One-factor-at-a-time effects: for each run that differs from the baseline in exactly one
 * factor, the accuracy change it caused. Runs that differ in several factors (or use a
 * different method) are not attributable to a single factor and are left out.
 */
export function factorEffects(runs: RunSummary[], baseline: RunSummary | undefined): FactorEffect[] {
  if (!baseline) return []
  const base = baseline.factors as Record<string, unknown>
  const out: FactorEffect[] = []
  for (const r of runs) {
    if (r.run_id === baseline.run_id || r.method !== baseline.method) continue
    const f = r.factors as Record<string, unknown>
    const diff = FACTORS.filter((k) => JSON.stringify(f[k]) !== JSON.stringify(base[k]))
    if (diff.length !== 1) continue
    out.push({
      factor: diff[0],
      value: String(f[diff[0]]),
      runId: r.run_id,
      name: r.name ?? r.run_id,
      accuracy: r.accuracy,
      delta: r.accuracy - baseline.accuracy,
    })
  }
  return out.sort((a, b) => FACTORS.indexOf(a.factor) - FACTORS.indexOf(b.factor) || b.delta - a.delta)
}

/** Default baseline: the run named like "raptor-baseline", else the oldest RAPTOR run. */
export function defaultBaseline(runs: RunSummary[]): RunSummary | undefined {
  return (
    runs.find((r) => /raptor.*baseline|baseline.*raptor/i.test(r.name ?? r.run_id)) ??
    [...runs]
      .filter((r) => r.method === 'raptor')
      .sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)))[0] ??
    runs[0]
  )
}
