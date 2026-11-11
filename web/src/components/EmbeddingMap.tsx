import { ParentSize } from '@visx/responsive'
import { Zoom } from '@visx/zoom'
import { Maximize2, Minus, Plus } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { Projection, TreeData } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Swatch } from '@/components/ui/primitives'
import { Segmented } from '@/components/ui/radix'
import { hullPath, paddedHull, type Pt } from '@/lib/hulls'
import { layerColor, layerName } from '@/lib/utils'

interface Props {
  projection: Projection
  tree: TreeData
  selected: number | null
  onSelect: (id: number | null) => void
  highlights?: Set<number>
}

function Chart({ width, height, projection, tree, selected, onSelect, highlights }: Props & { width: number; height: number }) {
  const [hover, setHover] = useState<number | null>(null)
  const [hullLayer, setHullLayer] = useState<string>('1')
  const margin = 28
  const sx = (x: number) => margin + x * (width - 2 * margin)
  const sy = (y: number) => margin + (1 - y) * (height - 2 * margin)
  const byId = useMemo(() => new Map(projection.points.map((p) => [p.id, p])), [projection])
  const treeById = useMemo(() => new Map(tree.nodes.map((n) => [n.id, n])), [tree])
  const layers = [...new Set(projection.points.map((p) => p.layer))].sort((a, b) => a - b)

  // Hulls: every summary node (in the chosen layer) outlines its children's points.
  const hulls = useMemo(() => {
    const want = hullLayer === 'all' ? null : Number(hullLayer)
    return tree.nodes
      .filter((n) => n.layer > 0 && (want === null || n.layer === want))
      .map((n) => {
        const pts = n.children
          .map((c) => byId.get(c))
          .filter((p): p is NonNullable<typeof p> => !!p)
          .map((p) => [sx(p.x), sy(p.y)] as Pt)
        return { id: n.id, layer: n.layer, d: hullPath(paddedHull(pts, 9)) }
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree, byId, hullLayer, width, height])

  const focus = hover ?? selected
  const focusNode = focus !== null ? treeById.get(focus) : undefined
  const focusCluster = new Set<number>([
    ...(focusNode?.parents ?? []),
    ...(focusNode && focusNode.layer > 0 ? [focusNode.id] : []),
  ])
  const hoverPt = hover !== null ? byId.get(hover) : undefined

  return (
    <Zoom<SVGSVGElement> width={width} height={height} scaleXMin={0.5} scaleXMax={8} scaleYMin={0.5} scaleYMax={8}>
      {(zoom) => (
        <div className="relative h-full w-full">
          <svg
            width={width}
            height={height}
            ref={zoom.containerRef}
            style={{ cursor: zoom.isDragging ? 'grabbing' : 'grab', touchAction: 'none' }}
            role="img"
            aria-label="2-D map of node embeddings; outlines show which nodes each summary covers"
            onClick={() => onSelect(null)}
          >
            <g transform={zoom.toString()}>
              {hulls.map((h) => {
                const lit = focusCluster.has(h.id)
                return (
                  <path
                    key={h.id}
                    d={h.d}
                    fill={layerColor(h.layer)}
                    fillOpacity={lit ? 0.22 : 0.07}
                    stroke={layerColor(h.layer)}
                    strokeOpacity={lit ? 0.9 : 0.35}
                    strokeWidth={lit ? 2 / zoom.transformMatrix.scaleX : 1 / zoom.transformMatrix.scaleX}
                    strokeLinejoin="round"
                  />
                )
              })}
              {projection.points.map((p) => {
                const isSel = selected === p.id
                const hl = highlights?.has(p.id)
                const dim = highlights && highlights.size > 0 && !hl
                const r = (p.layer === 0 ? 4.5 : 7) / Math.sqrt(zoom.transformMatrix.scaleX)
                return (
                  <g key={p.id}>
                    <circle
                      cx={sx(p.x)}
                      cy={sy(p.y)}
                      r={r + (isSel || hl ? 2.5 : 0)}
                      fill={layerColor(p.layer)}
                      fillOpacity={dim ? 0.25 : 1}
                      stroke="var(--surface)"
                      strokeWidth={2 / zoom.transformMatrix.scaleX}
                    />
                    {(isSel || hl) && (
                      <circle cx={sx(p.x)} cy={sy(p.y)} r={r + 6} fill="none" stroke="var(--ink)" strokeWidth={1.5 / zoom.transformMatrix.scaleX} />
                    )}
                    {/* generous hit target */}
                    <circle
                      cx={sx(p.x)}
                      cy={sy(p.y)}
                      r={Math.max(r + 6, 10 / zoom.transformMatrix.scaleX)}
                      fill="transparent"
                      onMouseEnter={() => setHover(p.id)}
                      onMouseLeave={() => setHover(null)}
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelect(p.id)
                      }}
                    />
                  </g>
                )
              })}
            </g>
          </svg>

          {hoverPt && treeById.get(hoverPt.id) && (
            <div
              className="pointer-events-none absolute z-10 w-64 rounded-xl border border-line bg-surface/95 p-2.5 text-xs shadow-card backdrop-blur"
              style={{
                left: Math.min(width - 270, Math.max(4, zoom.transformMatrix.translateX + sx(hoverPt.x) * zoom.transformMatrix.scaleX + 12)),
                top: Math.min(height - 120, Math.max(4, zoom.transformMatrix.translateY + sy(hoverPt.y) * zoom.transformMatrix.scaleY + 12)),
              }}
            >
              <div className="mb-1 flex items-center justify-between text-muted">
                <Swatch color={layerColor(hoverPt.layer)} label={`${layerName(hoverPt.layer)} · #${hoverPt.id}`} />
                {hoverPt.cluster !== null && hoverPt.cluster !== undefined && <span>in #{hoverPt.cluster}</span>}
              </div>
              <p className="line-clamp-4 leading-[17px] text-ink-2">{treeById.get(hoverPt.id)!.preview}</p>
            </div>
          )}

          <div className="absolute top-3 left-3 flex flex-wrap items-center gap-2">
            <Segmented
              label="Outline clusters for layer"
              value={hullLayer}
              onChange={setHullLayer}
              options={[
                ...layers.filter((l) => l > 0).map((l) => ({ value: String(l), label: `L${l} hulls` })),
                { value: 'all', label: 'All' },
                { value: 'none', label: 'None' },
              ]}
            />
          </div>
          <div className="absolute right-3 bottom-3 flex flex-col overflow-hidden rounded-xl border border-line bg-surface shadow-card">
            <Button variant="ghost" size="icon-sm" className="rounded-none" aria-label="Zoom in" onClick={() => zoom.scale({ scaleX: 1.3, scaleY: 1.3 })}>
              <Plus />
            </Button>
            <Button variant="ghost" size="icon-sm" className="rounded-none" aria-label="Zoom out" onClick={() => zoom.scale({ scaleX: 0.77, scaleY: 0.77 })}>
              <Minus />
            </Button>
            <Button variant="ghost" size="icon-sm" className="rounded-none" aria-label="Reset view" onClick={zoom.reset}>
              <Maximize2 />
            </Button>
          </div>
          <div className="absolute bottom-3 left-3 flex flex-wrap gap-3 rounded-lg bg-surface/85 px-2 py-1 backdrop-blur">
            {layers.map((l) => (
              <Swatch key={l} color={layerColor(l)} label={layerName(l)} />
            ))}
            <span className="text-[11px] text-muted">· {projection.method.toUpperCase()}</span>
          </div>
        </div>
      )}
    </Zoom>
  )
}

export function EmbeddingMap(props: Props) {
  return (
    <div className="dot-grid h-full w-full" data-testid="embedding-map">
      <ParentSize debounceTime={50}>
        {({ width, height }) => (width > 0 && height > 0 ? <Chart {...props} width={width} height={height} /> : null)}
      </ParentSize>
    </div>
  )
}
