import { Check, Swords, X } from 'lucide-react'
import { useState } from 'react'
import { urls } from '@/api/client'
import { useStory } from '@/api/hooks'
import type { Side } from '@/api/types'
import { LayerMix, RetrievedList, StreamingText } from '@/components/Retrieval'
import { Button } from '@/components/ui/button'
import { Badge, Card, ErrorNote, Skeleton, Textarea } from '@/components/ui/primitives'
import { Select } from '@/components/ui/radix'
import { type AnswerStream, useCompare } from '@/hooks/useStreams'
import { cn } from '@/lib/utils'

const SIDES: { side: Side; title: string; blurb: string }[] = [
  { side: 'flat', title: 'Flat RAG', blurb: 'Top chunks only, no hierarchy' },
  { side: 'raptor', title: 'RAPTOR', blurb: 'Chunks and summaries from every layer' },
]

function Verdict({ correct }: { correct?: boolean | null }) {
  if (correct === null || correct === undefined) return null
  return correct ? (
    <Badge tone="good">
      <Check className="size-3" /> correct
    </Badge>
  ) : (
    <Badge tone="critical">
      <X className="size-3" /> wrong
    </Badge>
  )
}

function SideCard({
  side,
  title,
  blurb,
  stream,
  onPick,
}: {
  side: Side
  title: string
  blurb: string
  stream: AnswerStream
  onPick: (id: number) => void
}) {
  const active = stream.status === 'streaming' || stream.status === 'retrieving'
  return (
    <Card className="flex min-w-0 flex-col" data-testid={`arena-${side}`}>
      <div className="flex items-start justify-between gap-2 border-b border-line px-4 py-3">
        <div>
          <h3 className="font-display text-lg leading-6 font-semibold">{title}</h3>
          <p className="text-xs text-muted">{blurb}</p>
        </div>
        <div className="flex items-center gap-1.5">
          {stream.final && <Badge>{stream.final.latency_s.toFixed(1)}s</Badge>}
          <Verdict correct={stream.final?.correct} />
        </div>
      </div>
      <div className="space-y-4 p-4">
        {stream.status === 'idle' && <p className="text-sm text-muted">Waiting for a question.</p>}
        {stream.retrieval && <LayerMix retrieval={stream.retrieval} />}
        {(stream.text || active) && <StreamingText text={stream.text} active={active && !stream.final} />}
        {stream.status === 'error' && <ErrorNote error={stream.error} />}
        {stream.retrieval && (
          <details>
            <summary className="cursor-pointer text-xs font-medium text-ink-2 select-none hover:text-ink">
              {stream.retrieval.nodes.length} retrieved nodes
            </summary>
            <RetrievedList className="mt-2" retrieval={stream.retrieval} onPick={onPick} />
          </details>
        )}
      </div>
    </Card>
  )
}

/** Flat RAG vs RAPTOR with the same context budget; QuALITY questions bring options and the gold answer. */
export function ArenaPanel({
  indexId,
  setId,
  onPick,
}: {
  indexId: string
  setId?: string | null
  onPick: (id: number) => void
}) {
  const story = useStory(setId)
  const cmp = useCompare(urls.compare(indexId))
  const [qid, setQid] = useState<string>('custom')
  const [custom, setCustom] = useState('')
  const questions = story.data?.questions ?? []
  const chosen = questions.find((q) => q.question_id === qid)
  const running = cmp.state.status === 'running'
  const canRun = !running && (!!chosen || !!custom.trim())

  const run = () => {
    if (chosen) void cmp.run({ question_id: chosen.question_id, set_id: setId ?? undefined, top_k: 10, max_tokens: 2000 })
    else void cmp.run({ question: custom.trim(), top_k: 10, max_tokens: 2000 })
  }
  const q = cmp.state.question
  const options = q?.options ?? chosen?.options
  const gold = q?.gold ?? chosen?.gold

  return (
    <div className="scrollbar-thin h-full overflow-y-auto p-4 sm:p-6">
      <div className="mx-auto max-w-6xl space-y-5">
        <Card className="space-y-3 p-4">
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
            <div className="space-y-2">
              {setId && (
                <div>
                  <span className="mb-1 block text-xs font-medium text-ink-2">QuALITY question</span>
                  {story.isLoading ? (
                    <Skeleton className="h-9" />
                  ) : (
                    <Select
                      label="QuALITY question"
                      className="w-full"
                      value={qid}
                      onValueChange={setQid}
                      options={[
                        { value: 'custom', label: 'Write my own question' },
                        ...questions.map((x) => ({
                          value: x.question_id,
                          label: x.question,
                          hint: x.difficult ? 'hard question' : undefined,
                        })),
                      ]}
                    />
                  )}
                </div>
              )}
              {chosen ? (
                <p className="text-sm text-ink">{chosen.question}</p>
              ) : (
                <Textarea
                  rows={2}
                  aria-label="Custom question"
                  value={custom}
                  onChange={(e) => setCustom(e.target.value)}
                  placeholder="Ask both systems the same question…"
                />
              )}
            </div>
            <Button variant="primary" onClick={run} disabled={!canRun}>
              <Swords /> {running ? 'Running…' : 'Compare'}
            </Button>
          </div>
          {options && (
            <ol className="grid gap-1.5 sm:grid-cols-2" aria-label="Answer options">
              {options.map((o, i) => (
                <li
                  key={i}
                  className={cn(
                    'flex gap-2 rounded-lg border px-3 py-1.5 text-sm',
                    gold === i + 1 ? 'border-good/50 bg-good/10 text-ink' : 'border-line text-ink-2',
                  )}
                >
                  <span className="num text-muted">{i + 1}</span>
                  <span>{o}</span>
                  {gold === i + 1 && (
                    <Badge tone="good" className="ml-auto self-start">
                      gold
                    </Badge>
                  )}
                </li>
              ))}
            </ol>
          )}
          {q?.budget && (
            <p className="text-[11px] text-muted">
              Same budget for both: top {q.budget.top_k} nodes, {q.budget.max_tokens} tokens.
            </p>
          )}
        </Card>
        {cmp.state.status === 'error' && <ErrorNote error={cmp.state.error} />}
        <div className="grid gap-4 lg:grid-cols-2">
          {SIDES.map((s) => (
            <SideCard key={s.side} {...s} stream={cmp.state.sides[s.side]} onPick={onPick} />
          ))}
        </div>
      </div>
    </div>
  )
}
