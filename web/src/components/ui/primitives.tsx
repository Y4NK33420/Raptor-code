import * as React from 'react'
import { cn } from '@/lib/utils'

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('rounded-2xl border border-line bg-surface shadow-card', className)}
      {...props}
    />
  )
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('flex items-start justify-between gap-3 p-4 pb-2', className)} {...props} />
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn('text-[13px] font-semibold tracking-wide text-ink-2 uppercase', className)} {...props} />
}

export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('p-4 pt-2', className)} {...props} />
}

const badgeTones = {
  neutral: 'bg-surface-2 text-ink-2 border-line',
  accent: 'bg-accent-soft text-accent border-transparent',
  good: 'bg-good/12 text-good border-transparent',
  critical: 'bg-critical/12 text-critical border-transparent',
  warning: 'bg-warning/15 text-warning border-transparent',
  outline: 'bg-transparent text-ink-2 border-line-strong',
}

export function Badge({
  tone = 'neutral',
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: keyof typeof badgeTones }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-4 whitespace-nowrap',
        badgeTones[tone],
        className,
      )}
      {...props}
    />
  )
}

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        'h-10 w-full rounded-lg border border-line bg-surface px-3 text-sm text-ink placeholder:text-muted outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/25',
        className,
      )}
      {...props}
    />
  ),
)
Input.displayName = 'Input'

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      'w-full rounded-lg border border-line bg-surface px-3 py-2.5 text-sm text-ink placeholder:text-muted outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/25',
      className,
    )}
    {...props}
  />
))
Textarea.displayName = 'Textarea'

export function Label({ className, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn('mb-1.5 block text-[13px] font-medium text-ink-2', className)} {...props} />
}

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('animate-pulse rounded-lg bg-surface-2', className)} {...props} />
}

export function Kbd({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        'rounded border border-line border-b-2 bg-surface px-1.5 py-px font-mono text-[10px] text-muted',
        className,
      )}
      {...props}
    />
  )
}

export function Empty({
  icon,
  title,
  children,
  className,
}: {
  icon?: React.ReactNode
  title: string
  children?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-2 p-10 text-center', className)}>
      {icon && <div className="mb-1 text-muted [&_svg]:size-8">{icon}</div>}
      <p className="font-display text-lg text-ink">{title}</p>
      {children && <div className="max-w-sm text-sm text-muted">{children}</div>}
    </div>
  )
}

export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error)
  return (
    <div role="alert" className="rounded-lg border border-critical/30 bg-critical/8 px-3 py-2 text-sm text-critical">
      {msg}
    </div>
  )
}

/** A small "dot + label" identity key. Text stays in ink colours; the swatch carries identity. */
export function Swatch({ color, label, className }: { color: string; label: React.ReactNode; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-xs text-ink-2', className)}>
      <span className="size-2.5 rounded-[3px]" style={{ background: color }} aria-hidden />
      {label}
    </span>
  )
}
