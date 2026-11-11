import { motion } from 'motion/react'
import { cn, layerColor } from '@/lib/utils'

/**
 * The index "silhouette": one bar per layer, width proportional to its node count,
 * root on top. A compact signature of the tree's shape used on cards and headers.
 */
export function Strata({
  counts,
  className,
  animate = true,
}: {
  counts: Record<string, number>
  className?: string
  animate?: boolean
}) {
  const layers = Object.keys(counts)
    .map(Number)
    .sort((a, b) => b - a)
  const max = Math.max(1, ...Object.values(counts))
  return (
    <div
      className={cn('flex flex-col items-center justify-end gap-[3px]', className)}
      role="img"
      aria-label={`Tree shape: ${layers.map((l) => `layer ${l} has ${counts[l]} nodes`).join(', ')}`}
    >
      {layers.map((l, i) => (
        <motion.div
          key={l}
          className="h-2 rounded-full"
          style={{ background: layerColor(l), width: `${Math.max(6, (counts[l] / max) * 100)}%` }}
          initial={animate ? { scaleX: 0 } : false}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.05 * (layers.length - i), type: 'spring', stiffness: 260, damping: 26 }}
        />
      ))}
    </div>
  )
}
