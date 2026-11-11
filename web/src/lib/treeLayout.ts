/**
 * Layered layout for a RAPTOR tree (a DAG: soft clustering gives some nodes several parents).
 *
 * Leaves sit on the bottom row, the root layer on top. Rows are ordered top-down so each
 * parent's children are grouped under it (barycentre of parent positions), which keeps
 * edges short and makes clusters read as fans.
 */
export interface LayoutInput {
  id: number
  layer: number
  children: number[]
}

export interface Positioned {
  id: number
  layer: number
  x: number
  y: number
}

export interface LayoutOptions {
  nodeWidth?: number
  gapX?: number
  rowHeight?: number
  maxPerRow?: number
}

export function layoutTree(nodes: LayoutInput[], opts: LayoutOptions = {}): Map<number, Positioned> {
  const nodeWidth = opts.nodeWidth ?? 180
  const gapX = opts.gapX ?? 24
  const rowHeight = opts.rowHeight ?? 190
  const byLayer = new Map<number, LayoutInput[]>()
  const parents = new Map<number, number[]>()
  for (const n of nodes) {
    if (!byLayer.has(n.layer)) byLayer.set(n.layer, [])
    byLayer.get(n.layer)!.push(n)
    for (const c of n.children) {
      if (!parents.has(c)) parents.set(c, [])
      parents.get(c)!.push(n.id)
    }
  }
  const layers = [...byLayer.keys()].sort((a, b) => b - a) // top first
  const order = new Map<number, number>() // id -> position within its row
  for (const layer of layers) {
    const row = byLayer.get(layer)!
    const keyed = row.map((n, i) => {
      const ps = (parents.get(n.id) ?? []).map((p) => order.get(p)).filter((v) => v !== undefined)
      const bary = ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : Number.POSITIVE_INFINITY
      return { n, bary, i }
    })
    keyed.sort((a, b) => a.bary - b.bary || a.n.id - b.n.id || a.i - b.i)
    keyed.forEach((k, pos) => order.set(k.n.id, pos))
    byLayer.set(
      layer,
      keyed.map((k) => k.n),
    )
  }
  // Wide layers wrap into stacked sub-rows so a 100-leaf tree is not a single hairline.
  const maxPerRow = opts.maxPerRow ?? 8
  const widest = Math.min(Math.max(...[...byLayer.values()].map((r) => r.length), 1), maxPerRow)
  const span = widest * (nodeWidth + gapX)
  const out = new Map<number, Positioned>()
  let y = 0
  for (const layer of layers) {
    const row = byLayer.get(layer)!
    const subRows = Math.ceil(row.length / maxPerRow)
    const perRow = Math.ceil(row.length / subRows)
    row.forEach((n, i) => {
      const r = Math.floor(i / perRow)
      const inRow = Math.min(perRow, row.length - r * perRow)
      const step = span / inRow
      out.set(n.id, {
        id: n.id,
        layer,
        x: Math.round((i - r * perRow) * step + (step - nodeWidth) / 2),
        y: y + r * rowHeight * 0.62,
      })
    })
    y += (subRows - 1) * rowHeight * 0.62 + rowHeight
  }
  return out
}
