import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Colour for a tree layer (fixed order of the validated categorical palette). */
export function layerColor(layer: number): string {
  return `var(--layer-${Math.min(Math.max(layer, 0), 5)})`
}

export function layerName(layer: number, top?: number): string {
  if (layer === 0) return 'Leaves'
  if (top !== undefined && layer === top) return `Root · L${layer}`
  return `Layer ${layer}`
}

export const pct = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined ? '—' : `${(x * 100).toFixed(digits)}%`

export const signedPts = (x: number, digits = 1) =>
  `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(digits)} pts`

export function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 10_000) return `${Math.round(n / 1000)}k`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

export function timeAgo(iso: string): string {
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.max(0, (Date.now() - t) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  const d = Math.floor(s / 86400)
  if (d < 30) return `${d} d ago`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function formatP(p: number): string {
  if (p < 0.001) return 'p < 0.001'
  return `p = ${p.toFixed(3)}`
}
