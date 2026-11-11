import { describe, expect, it } from 'vitest'
import { SseParser } from '@/api/sse'
import type { RunSummary } from '@/api/types'
import { factorEffects } from './effects'
import { layoutTree } from './treeLayout'
import { formatP, signedPts } from './utils'

describe('SseParser', () => {
  it('reassembles events split across chunks', () => {
    const p = new SseParser()
    expect(p.push('event: token\ndata: {"text":"he')).toEqual([])
    const out = p.push('llo"}\n\nevent: done\ndata: {}\n\n')
    expect(out.map((m) => m.event)).toEqual(['token', 'done'])
    expect(out[0].data).toEqual({ text: 'hello' })
  })
})

describe('layoutTree', () => {
  const nodes = Array.from({ length: 20 }, (_, i) => ({ id: i, layer: 0, children: [] as number[] }))
  nodes.push({ id: 100, layer: 1, children: [0, 1, 2] })
  it('puts parents above children and wraps wide rows', () => {
    const l = layoutTree(nodes, { maxPerRow: 8 })
    expect(l.get(100)!.y).toBeLessThan(l.get(0)!.y)
    const ys = new Set(nodes.filter((n) => n.layer === 0).map((n) => l.get(n.id)!.y))
    expect(ys.size).toBe(3)
  })
  it('never overlaps nodes in one row', () => {
    const l = layoutTree(nodes, { nodeWidth: 150, gapX: 10, maxPerRow: 8 })
    const byY = new Map<number, number[]>()
    for (const n of nodes.filter((n) => n.layer === 0)) {
      const p = l.get(n.id)!
      byY.set(p.y, [...(byY.get(p.y) ?? []), p.x].sort((a, b) => a - b))
    }
    for (const xs of byY.values()) for (let i = 1; i < xs.length; i++) expect(xs[i] - xs[i - 1]).toBeGreaterThanOrEqual(150)
  })
})

describe('factorEffects', () => {
  const run = (id: string, accuracy: number, factors: Record<string, string>): RunSummary =>
    ({ run_id: id, accuracy, factors, method: 'raptor', ci95: [] }) as unknown as RunSummary
  it('attributes only single-factor changes', () => {
    const base = run('b', 0.5, { chunking: 'a', retrieval: 'x' })
    const fx = factorEffects(
      [base, run('c', 0.6, { chunking: 'z', retrieval: 'x' }), run('d', 0.7, { chunking: 'z', retrieval: 'y' })],
      base,
    )
    expect(fx.map((f) => f.runId)).toEqual(['c'])
    expect(fx[0].delta).toBeCloseTo(0.1)
  })
})

describe('formatting', () => {
  it('formats points and p-values', () => {
    expect(signedPts(0.123)).toBe('+12.3 pts')
    expect(signedPts(-0.05)).toBe('−5.0 pts')
    expect(formatP(0.0001)).toBe('p < 0.001')
  })
})
