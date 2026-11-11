import { createContext, createElement, type ReactNode, useCallback, useContext, useEffect, useState } from 'react'

export type ThemePref = 'light' | 'dark' | 'system'
const KEY = 'raptor-theme'

function read(): ThemePref {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

function apply(pref: ThemePref) {
  const dark = pref === 'system' ? window.matchMedia('(prefers-color-scheme: dark)').matches : pref === 'dark'
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
}

function useThemeState() {
  const [pref, setPref] = useState<ThemePref>(read)

  useEffect(() => {
    apply(pref)
    if (pref !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const on = () => apply('system')
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [pref])

  const set = useCallback((p: ThemePref) => {
    try {
      if (p === 'system') localStorage.removeItem(KEY)
      else localStorage.setItem(KEY, p)
    } catch {
      /* private mode */
    }
    setPref(p)
  }, [])

  return { pref, set }
}

const ThemeContext = createContext<ReturnType<typeof useThemeState> | null>(null)

export function ThemeProvider({ children }: { children: ReactNode }) {
  return createElement(ThemeContext.Provider, { value: useThemeState() }, children)
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>')
  return ctx
}

/** The resolved theme, kept in sync with <html data-theme>. */
export function useResolvedTheme(): 'light' | 'dark' {
  const get = () => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')
  const [t, setT] = useState<'light' | 'dark'>(get)
  useEffect(() => {
    const obs = new MutationObserver(() => setT(get()))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => obs.disconnect()
  }, [])
  return t
}
