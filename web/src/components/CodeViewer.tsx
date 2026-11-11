import type { HighlighterCore } from 'shiki/core'
import { useEffect, useRef, useState } from 'react'
import { useCodeFile } from '@/api/hooks'
import { ErrorNote, Skeleton } from '@/components/ui/primitives'

export interface LineRange {
  start: number
  end: number
}

const LANGS = new Set(['python', 'javascript', 'typescript', 'tsx', 'json', 'yaml', 'markdown', 'toml', 'bash'])

// Fine-grained bundle: only the languages the explorer shows, with the JS regex engine (no WASM).
let highlighter: Promise<HighlighterCore> | undefined
function loadHighlighter() {
  highlighter ??= (async () => {
    const [{ createHighlighterCore }, { createJavaScriptRegexEngine }] = await Promise.all([
      import('shiki/core'),
      import('shiki/engine/javascript'),
    ])
    return createHighlighterCore({
      themes: [import('shiki/themes/github-light.mjs'), import('shiki/themes/github-dark.mjs')],
      langs: [
        import('shiki/langs/python.mjs'),
        import('shiki/langs/javascript.mjs'),
        import('shiki/langs/typescript.mjs'),
        import('shiki/langs/tsx.mjs'),
        import('shiki/langs/json.mjs'),
        import('shiki/langs/yaml.mjs'),
        import('shiki/langs/markdown.mjs'),
        import('shiki/langs/toml.mjs'),
        import('shiki/langs/bash.mjs'),
      ],
      engine: createJavaScriptRegexEngine(),
    })
  })()
  return highlighter
}

/** Highlighted source with optional line range emphasis; scrolls the range into view. */
export function CodeViewer({ repoId, path, range }: { repoId: string; path: string; range?: LineRange | null }) {
  const file = useCodeFile(repoId, path)
  const [html, setHtml] = useState<string | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const text = file.data?.text
  const language = file.data?.language

  useEffect(() => {
    if (text === undefined) return
    let cancelled = false
    setHtml(null)
    void loadHighlighter()
      .then((hl) =>
        hl.codeToHtml(text, {
          lang: language && LANGS.has(language) ? language : 'text',
          themes: { light: 'github-light', dark: 'github-dark' },
          transformers: [
            {
              line(node, line) {
                node.properties['data-line'] = line
                if (range && line >= range.start && line <= range.end) this.addClassToHast(node, 'cited')
              },
            },
          ],
        }),
      )
      .then((h) => {
        if (!cancelled) setHtml(h)
      })
      .catch(() => {
        if (!cancelled) setHtml(null)
      })
    return () => {
      cancelled = true
    }
  }, [text, language, range?.start, range?.end]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!html || !range) return
    const el = box.current?.querySelector<HTMLElement>(`[data-line="${range.start}"]`)
    el?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [html, range?.start]) // eslint-disable-line react-hooks/exhaustive-deps

  if (file.isError) return <ErrorNote error={file.error} />
  if (file.isLoading) return <Skeleton className="m-4 h-64" />
  return (
    <div ref={box} className="code-view scrollbar-thin h-full overflow-auto" data-testid="code-viewer">
      {html ? (
        <div dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        // Plain fallback while the highlighter loads (or if the language is unknown).
        <pre className="p-4 font-mono text-[12.5px] leading-5 text-ink-2">{text}</pre>
      )}
    </div>
  )
}
