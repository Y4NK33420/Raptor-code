import { CornerDownLeft, Sparkles, Square } from 'lucide-react'
import { useEffect, useState } from 'react'
import { urls } from '@/api/client'
import type { RetrievalEvent } from '@/api/types'
import { LayerMix, RetrievedList, StreamingText } from '@/components/Retrieval'
import { Button } from '@/components/ui/button'
import { Badge, ErrorNote, Textarea } from '@/components/ui/primitives'
import { useAsk } from '@/hooks/useStreams'

const SUGGESTIONS = [
  'What is the main conflict in this text?',
  'Who are the key characters and how are they related?',
  'Summarise the whole text in three sentences.',
]

/** Ask a question of one index: streams the answer and reports what was retrieved. */
export function AskPanel({
  indexId,
  onRetrieval,
  onPick,
  selected,
}: {
  indexId: string
  onRetrieval: (r: RetrievalEvent | undefined) => void
  onPick: (id: number) => void
  selected: number | null
}) {
  const [q, setQ] = useState('')
  const ask = useAsk(urls.query(indexId))
  const s = ask.state
  const busy = s.status === 'retrieving' || s.status === 'streaming'

  // Lift the retrieval to the parent (it lights the nodes up in the tree).
  useEffect(() => {
    onRetrieval(s.retrieval)
  }, [s.retrieval, onRetrieval])

  const submit = (question = q) => {
    const text = question.trim()
    if (!text || busy) return
    setQ(text)
    void ask.run({ question: text })
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <form
        className="space-y-2 border-b border-line p-4"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <Textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit()
          }}
          rows={3}
          aria-label="Question"
          placeholder="Ask anything about this index…"
        />
        <div className="flex items-center justify-between gap-2">
          <span className="hidden text-[11px] text-muted sm:inline">Ctrl + Enter to send</span>
          {busy ? (
            <Button type="button" variant="secondary" size="sm" onClick={ask.stop}>
              <Square /> Stop
            </Button>
          ) : (
            <Button type="submit" variant="primary" size="sm" disabled={!q.trim()}>
              <CornerDownLeft /> Ask
            </Button>
          )}
        </div>
      </form>
      <div className="scrollbar-thin min-h-0 flex-1 space-y-5 overflow-y-auto p-4">
        {s.status === 'idle' && (
          <div className="space-y-2">
            <p className="flex items-center gap-1.5 text-xs font-medium text-muted">
              <Sparkles className="size-3.5" /> Try
            </p>
            {SUGGESTIONS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => submit(t)}
                className="block w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-left text-sm text-ink-2 hover:border-line-strong hover:text-ink"
              >
                {t}
              </button>
            ))}
          </div>
        )}
        {s.status === 'retrieving' && <p className="text-sm text-muted">Searching the tree…</p>}
        {s.retrieval && (
          <section className="space-y-3" aria-label="Retrieval">
            <LayerMix retrieval={s.retrieval} />
            <p className="text-[11px] text-muted">
              {s.retrieval.nodes.length} nodes lit up in the tree · retrieved in{' '}
              <span className="num">{s.retrieval.latency_s.toFixed(2)}s</span>
            </p>
          </section>
        )}
        {(s.text || s.status === 'streaming') && (
          <section aria-label="Answer" aria-live="polite" data-testid="answer">
            <h3 className="mb-1.5 flex items-center gap-2 text-[11px] font-semibold tracking-wider text-muted uppercase">
              Answer
              {s.final && <Badge>{s.final.latency_s.toFixed(1)}s</Badge>}
            </h3>
            <StreamingText text={s.text} active={s.status === 'streaming' && !s.final} />
          </section>
        )}
        {s.status === 'error' && <ErrorNote error={s.error} />}
        {s.retrieval && (
          <section>
            <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Retrieved nodes</h3>
            <RetrievedList retrieval={s.retrieval} onPick={onPick} selected={selected} />
          </section>
        )}
      </div>
    </div>
  )
}
