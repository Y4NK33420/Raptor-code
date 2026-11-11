import { Code2, FlaskConical, Library, Monitor, Moon, Sun } from 'lucide-react'
import { NavLink, Outlet, useLocation } from 'react-router'
import { useInfo } from '@/api/hooks'
import { Tip } from '@/components/ui/radix'
import { type ThemePref, useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Library', icon: Library, match: (p: string) => p === '/' || p.startsWith('/indexes') || p.startsWith('/build') },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical, match: (p: string) => p.startsWith('/experiments') },
  { to: '/code', label: 'Code', icon: Code2, match: (p: string) => p.startsWith('/code') },
]

export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect x="11" y="6" width="10" height="5" rx="2.5" fill="var(--accent-fill)" />
      <rect x="6.5" y="13.5" width="19" height="5" rx="2.5" fill="var(--layer-1)" />
      <rect x="2" y="21" width="28" height="5" rx="2.5" fill="var(--layer-0)" />
    </svg>
  )
}

const THEMES: { value: ThemePref; icon: typeof Sun; label: string }[] = [
  { value: 'light', icon: Sun, label: 'Light theme' },
  { value: 'dark', icon: Moon, label: 'Dark theme' },
  { value: 'system', icon: Monitor, label: 'Match system' },
]

function ThemeSwitch({ compact = false }: { compact?: boolean }) {
  const { pref, set } = useTheme()
  return (
    <div role="radiogroup" aria-label="Theme" className={cn('flex rounded-lg border border-line bg-surface-2 p-0.5', compact && 'scale-90')}>
      {THEMES.map((t) => (
        <Tip key={t.value} content={t.label}>
          <button
            type="button"
            role="radio"
            aria-checked={pref === t.value}
            aria-label={t.label}
            onClick={() => set(t.value)}
            className={cn(
              'grid size-7 place-items-center rounded-md text-muted transition hover:text-ink',
              pref === t.value && 'bg-surface text-ink shadow-card',
            )}
          >
            <t.icon className="size-3.5" />
          </button>
        </Tip>
      ))}
    </div>
  )
}

export function AppShell() {
  const { pathname } = useLocation()
  const info = useInfo()
  return (
    <div className="flex h-full min-h-0">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-2 focus:rounded-lg focus:bg-surface focus:p-2">
        Skip to content
      </a>
      {/* desktop rail */}
      <aside className="hidden w-[220px] shrink-0 flex-col border-r border-line bg-surface/60 px-3 py-4 md:flex">
        <NavLink to="/" className="mb-6 flex items-center gap-2.5 px-2">
          <Logo className="size-7" />
          <div className="leading-none">
            <div className="font-display text-[22px] font-semibold tracking-tight text-ink">raptor</div>
            <div className="mt-1 text-[10px] tracking-[0.18em] text-muted uppercase">tree retrieval</div>
          </div>
        </NavLink>
        <nav aria-label="Main" className="flex flex-col gap-0.5">
          {NAV.map((n) => {
            const active = n.match(pathname)
            return (
              <NavLink
                key={n.to}
                to={n.to}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-ink-2 transition hover:bg-surface-2 hover:text-ink',
                  active && 'bg-surface-2 font-medium text-ink',
                )}
              >
                <n.icon className={cn('size-4', active && 'text-accent')} />
                {n.label}
              </NavLink>
            )
          })}
        </nav>
        <div className="mt-auto space-y-3 px-1">
          {info.data && (
            <div className="rounded-xl border border-line bg-surface p-3 text-[11px] leading-5 text-muted">
              <div className="mb-1 flex items-center gap-1.5 font-medium text-ink-2">
                <span
                  className={cn('size-1.5 rounded-full', info.data.fake_models ? 'bg-warning' : 'bg-good')}
                  aria-hidden
                />
                {info.data.fake_models ? 'Offline fake models' : 'Live models'}
              </div>
              <div className="truncate" title={info.data.llm_model}>
                LLM · <span className="num">{info.data.llm_model}</span>
              </div>
              <div className="truncate">code · {info.data.code_backend === 'stub' ? 'stub index' : 'raptor.code'}</div>
            </div>
          )}
          <ThemeSwitch />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* phone top bar */}
        <header className="flex items-center justify-between border-b border-line bg-surface/80 px-4 py-2.5 backdrop-blur md:hidden">
          <NavLink to="/" className="flex items-center gap-2">
            <Logo className="size-6" />
            <span className="font-display text-xl font-semibold">raptor</span>
          </NavLink>
          <ThemeSwitch compact />
        </header>
        <main id="main" className="min-h-0 flex-1 overflow-y-auto pb-16 md:pb-0">
          <Outlet />
        </main>
        {/* phone bottom nav */}
        <nav
          aria-label="Main"
          className="fixed inset-x-0 bottom-0 z-40 flex border-t border-line bg-surface/95 backdrop-blur md:hidden"
        >
          {NAV.map((n) => {
            const active = n.match(pathname)
            return (
              <NavLink
                key={n.to}
                to={n.to}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'flex flex-1 flex-col items-center gap-0.5 py-2 text-[11px] text-muted',
                  active && 'text-ink',
                )}
              >
                <n.icon className={cn('size-5', active && 'text-accent')} />
                {n.label}
              </NavLink>
            )
          })}
        </nav>
      </div>
    </div>
  )
}

export function PageHeader({
  eyebrow,
  title,
  children,
  actions,
}: {
  eyebrow?: React.ReactNode
  title: React.ReactNode
  children?: React.ReactNode
  actions?: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-4 px-4 pt-6 pb-4 sm:flex-row sm:items-end sm:justify-between sm:px-8 sm:pt-8">
      <div className="min-w-0">
        {eyebrow && <div className="mb-1.5 text-xs font-medium tracking-[0.14em] text-muted uppercase">{eyebrow}</div>}
        <h1 className="font-display text-3xl leading-tight font-semibold tracking-tight text-ink text-balance sm:text-4xl">
          {title}
        </h1>
        {children && <div className="mt-2 max-w-2xl text-sm text-ink-2">{children}</div>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}
