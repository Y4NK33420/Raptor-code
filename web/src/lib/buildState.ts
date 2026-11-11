import type { SseMessage } from '@/api/sse'

export type Stage = 'queued' | 'chunking' | 'embedding' | 'clustering' | 'summarizing' | 'saving' | 'complete' | 'error'

export interface BuildParent {
  id: number
  layer: number
  children: number[]
  text?: string
}

export interface BuildState {
  stage: Stage
  layer: number
  name?: string
  chars?: number
  chunks: { id: number; preview: string; tokens: number }[]
  parents: Map<number, BuildParent>
  layers: number[] // summary layers announced so far
  summarized: number
  expected: number // summaries announced by cluster events
  feed: { id: number; layer: number; text: string; t: number }[]
  cost: { llm_calls: number; embed_calls: number; embedded_texts: number }
  t: number
  result?: { index_id: string; num_nodes: number; num_layers: number; build_seconds: number }
  error?: string
}

export const initialBuild = (): BuildState => ({
  stage: 'queued',
  layer: 0,
  chunks: [],
  parents: new Map(),
  layers: [],
  summarized: 0,
  expected: 0,
  feed: [],
  cost: { llm_calls: 0, embed_calls: 0, embedded_texts: 0 },
  t: 0,
})

type Data = Record<string, unknown>

/** Fold one server-sent build event into the UI state (pure; replays give the same result). */
export function reduceBuild(state: BuildState, msg: SseMessage): BuildState {
  const d = (msg.data ?? {}) as Data
  const t = typeof d.t === 'number' ? d.t : state.t
  const s: BuildState = { ...state, t }
  switch (msg.event) {
    case 'start':
      return { ...s, name: d.name as string, chars: d.chars as number, stage: 'chunking' }
    case 'progress': {
      const stage = d.stage as string
      if (stage === 'done') return { ...s, stage: 'saving' }
      if (stage === 'chunking' || stage === 'embedding' || stage === 'clustering' || stage === 'summarizing')
        return { ...s, stage, layer: typeof d.layer === 'number' ? d.layer : s.layer }
      return s
    }
    case 'chunks':
      return { ...s, chunks: d.chunks as BuildState['chunks'] }
    case 'clusters': {
      const layer = d.layer as number
      const parents = new Map(s.parents)
      const list = d.parents as { id: number; children: number[] }[]
      for (const p of list) parents.set(p.id, { id: p.id, layer, children: p.children })
      return {
        ...s,
        parents,
        layers: s.layers.includes(layer) ? s.layers : [...s.layers, layer],
        expected: s.expected + list.length,
      }
    }
    case 'summary': {
      const id = d.id as number
      const parents = new Map(s.parents)
      const p = parents.get(id)
      if (p) parents.set(id, { ...p, text: d.preview as string })
      const cost = (d.cost as BuildState['cost']) ?? s.cost
      const feed = [{ id, layer: d.layer as number, text: d.preview as string, t }, ...s.feed].slice(0, 40)
      return { ...s, parents, summarized: s.summarized + 1, cost, feed }
    }
    case 'complete':
      return {
        ...s,
        stage: 'complete',
        result: d as unknown as BuildState['result'],
        cost: (d.cost as BuildState['cost']) ?? s.cost,
      }
    case 'error':
      return { ...s, stage: 'error', error: (d.message as string) ?? 'build failed' }
    default:
      return s
  }
}

/** x position (0..1) for every node: leaves evenly spaced, parents at their children's mean. */
export function buildPositions(state: BuildState): Map<number, number> {
  const pos = new Map<number, number>()
  const n = state.chunks.length
  state.chunks.forEach((c, i) => pos.set(c.id, (i + 0.5) / Math.max(n, 1)))
  for (const layer of [...state.layers].sort((a, b) => a - b)) {
    const row = [...state.parents.values()].filter((p) => p.layer === layer)
    const raw = row.map((p) => {
      const xs = p.children.map((c) => pos.get(c)).filter((x): x is number => x !== undefined)
      return { id: p.id, x: xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0.5 }
    })
    raw.sort((a, b) => a.x - b.x)
    // keep a minimum gap so siblings never overlap, then re-centre on the original mean
    const gap = Math.min(0.9 / Math.max(raw.length, 1), 0.07)
    for (let i = 1; i < raw.length; i++) raw[i].x = Math.max(raw[i].x, raw[i - 1].x + gap)
    if (raw.length) {
      const over = raw[raw.length - 1].x - 0.97
      if (over > 0) for (const r of raw) r.x -= over
      const under = 0.03 - raw[0].x
      if (under > 0) for (const r of raw) r.x += under
    }
    for (const r of raw) pos.set(r.id, r.x)
  }
  return pos
}
