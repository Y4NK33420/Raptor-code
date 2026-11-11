/* shadcn/ui-style wrappers around Radix primitives, styled with the app's tokens. */
import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as SelectPrimitive from '@radix-ui/react-select'
import * as SliderPrimitive from '@radix-ui/react-slider'
import * as TabsPrimitive from '@radix-ui/react-tabs'
import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { Check, ChevronDown, X } from 'lucide-react'
import * as React from 'react'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------- tabs
export const Tabs = TabsPrimitive.Root

export const TabsList = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn('inline-flex h-10 items-center gap-1 rounded-xl border border-line bg-surface-2 p-1', className)}
    {...props}
  />
))
TabsList.displayName = 'TabsList'

export const TabsTrigger = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      'inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-[13px] font-medium text-ink-2 transition hover:text-ink data-[state=active]:bg-surface data-[state=active]:text-ink data-[state=active]:shadow-card [&_svg]:size-3.5',
      className,
    )}
    {...props}
  />
))
TabsTrigger.displayName = 'TabsTrigger'

export const TabsContent = TabsPrimitive.Content

// --------------------------------------------------------- segmented
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className,
  label,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: React.ReactNode }[]
  className?: string
  label: string
}) {
  return (
    <ToggleGroupPrimitive.Root
      type="single"
      value={value}
      aria-label={label}
      onValueChange={(v) => v && onChange(v as T)}
      className={cn('inline-flex rounded-lg border border-line bg-surface-2 p-0.5', className)}
    >
      {options.map((o) => (
        <ToggleGroupPrimitive.Item
          key={o.value}
          value={o.value}
          className="h-7 rounded-md px-2.5 text-xs font-medium text-ink-2 transition hover:text-ink data-[state=on]:bg-surface data-[state=on]:text-ink data-[state=on]:shadow-card"
        >
          {o.label}
        </ToggleGroupPrimitive.Item>
      ))}
    </ToggleGroupPrimitive.Root>
  )
}

// ------------------------------------------------------------ dialog
export const Dialog = DialogPrimitive.Root
export const DialogTrigger = DialogPrimitive.Trigger
export const DialogClose = DialogPrimitive.Close

export function DialogContent({
  className,
  children,
  title,
  description,
  ...props
}: React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
  title: string
  description?: string
}) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/45 backdrop-blur-[2px] data-[state=open]:animate-in" />
      <DialogPrimitive.Content
        className={cn(
          'fixed inset-x-0 bottom-0 z-50 max-h-[92dvh] overflow-y-auto rounded-t-2xl border border-line bg-surface p-5 shadow-card outline-none sm:inset-auto sm:top-1/2 sm:left-1/2 sm:w-[min(640px,calc(100vw-32px))] sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-2xl sm:p-6',
          className,
        )}
        {...props}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <DialogPrimitive.Title className="font-display text-2xl text-ink">{title}</DialogPrimitive.Title>
            {description && (
              <DialogPrimitive.Description className="mt-1 text-sm text-muted">
                {description}
              </DialogPrimitive.Description>
            )}
          </div>
          <DialogPrimitive.Close
            className="rounded-lg p-1.5 text-muted hover:bg-surface-2 hover:text-ink"
            aria-label="Close"
          >
            <X className="size-4" />
          </DialogPrimitive.Close>
        </div>
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

// ----------------------------------------------------------- tooltip
export const TooltipProvider = TooltipPrimitive.Provider

export function Tip({ content, children, side = 'top' }: { content: React.ReactNode; children: React.ReactNode; side?: 'top' | 'bottom' | 'left' | 'right' }) {
  return (
    <TooltipPrimitive.Root delayDuration={250}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-xs rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs text-ink shadow-card"
        >
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}

// ------------------------------------------------------------ select
export function Select({
  value,
  onValueChange,
  options,
  placeholder,
  className,
  label,
}: {
  value: string | undefined
  onValueChange: (v: string) => void
  options: { value: string; label: React.ReactNode; hint?: React.ReactNode }[]
  placeholder?: string
  className?: string
  label: string
}) {
  return (
    <SelectPrimitive.Root value={value} onValueChange={onValueChange}>
      <SelectPrimitive.Trigger
        aria-label={label}
        className={cn(
          'inline-flex h-9 min-w-0 items-center justify-between gap-2 rounded-lg border border-line bg-surface px-3 text-left text-sm text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/25',
          className,
        )}
      >
        <span className="truncate">
          <SelectPrimitive.Value placeholder={placeholder} />
        </span>
        <ChevronDown className="size-4 shrink-0 text-muted" />
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={4}
          className="z-50 max-h-[min(420px,var(--radix-select-content-available-height))] min-w-[var(--radix-select-trigger-width)] max-w-[min(560px,calc(100vw-24px))] overflow-hidden rounded-xl border border-line bg-surface shadow-card"
        >
          <SelectPrimitive.Viewport className="p-1">
            {options.map((o) => (
              <SelectPrimitive.Item
                key={o.value}
                value={o.value}
                className="relative flex cursor-default items-start gap-2 rounded-lg py-2 pr-3 pl-7 text-sm text-ink outline-none select-none data-[highlighted]:bg-surface-2"
              >
                <SelectPrimitive.ItemIndicator className="absolute top-2.5 left-2">
                  <Check className="size-3.5 text-accent" />
                </SelectPrimitive.ItemIndicator>
                <div className="min-w-0">
                  <SelectPrimitive.ItemText>{o.label}</SelectPrimitive.ItemText>
                  {o.hint && <div className="mt-0.5 text-xs text-muted">{o.hint}</div>}
                </div>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  )
}

// ------------------------------------------------------------ slider
export function Slider({
  value,
  onValueChange,
  min,
  max,
  step = 1,
  label,
}: {
  value: number
  onValueChange: (v: number) => void
  min: number
  max: number
  step?: number
  label: string
}) {
  return (
    <SliderPrimitive.Root
      value={[value]}
      min={min}
      max={max}
      step={step}
      onValueChange={(v) => onValueChange(v[0])}
      className="relative flex h-5 w-full touch-none items-center select-none"
    >
      <SliderPrimitive.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-surface-3">
        <SliderPrimitive.Range className="absolute h-full bg-accent" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        aria-label={label}
        className="block size-4 rounded-full border-2 border-accent bg-surface shadow-card"
      />
    </SliderPrimitive.Root>
  )
}
