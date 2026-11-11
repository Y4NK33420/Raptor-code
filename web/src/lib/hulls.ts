import { polygonHull } from 'd3-polygon'

export type Pt = [number, number]

/**
 * A padded convex hull around `points`: each point is expanded to a small circle first,
 * so 1- and 2-point clusters still get a rounded outline.
 */
export function paddedHull(points: Pt[], pad: number, segments = 10): Pt[] {
  if (points.length === 0) return []
  const ring: Pt[] = []
  for (const [x, y] of points) {
    for (let i = 0; i < segments; i++) {
      const a = (i / segments) * Math.PI * 2
      ring.push([x + Math.cos(a) * pad, y + Math.sin(a) * pad])
    }
  }
  return (polygonHull(ring) as Pt[] | null) ?? ring
}

export function hullPath(poly: Pt[]): string {
  if (!poly.length) return ''
  return `M${poly.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join('L')}Z`
}
