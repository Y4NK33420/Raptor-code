import { useCallback, useEffect, useRef, useState } from 'react'
import { streamSse } from '@/api/sse'
import type {
  AnswerEvent,
  CodeHit,
  CompareQuestionEvent,
  CompareRequest,
  QueryRequest,
  RetrievalEvent,
  Side,
} from '@/api/types'

export interface AnswerStream {
  status: 'idle' | 'retrieving' | 'streaming' | 'done' | 'error'
  retrieval?: RetrievalEvent
  text: string
  final?: AnswerEvent
  error?: string
}

const idle = (): AnswerStream => ({ status: 'idle', text: '' })

function useAbortable() {
  const ref = useRef<AbortController | null>(null)
  useEffect(() => () => ref.current?.abort(), [])
  const next = () => {
    ref.current?.abort()
    ref.current = new AbortController()
    return ref.current
  }
  const stop = () => ref.current?.abort()
  return { next, stop }
}

/** Streaming RAPTOR question answering over one index. */
export function useAsk(url: string) {
  const [state, setState] = useState<AnswerStream>(idle)
  const ctl = useAbortable()
  const run = useCallback(
    async (body: QueryRequest) => {
      const c = ctl.next()
      setState({ status: 'retrieving', text: '' })
      try {
        await streamSse(url, { method: 'POST', body, signal: c.signal }, (m) => {
          if (m.event === 'retrieval')
            setState((s) => ({ ...s, status: 'streaming', retrieval: m.data as RetrievalEvent }))
          else if (m.event === 'token')
            setState((s) => ({ ...s, text: s.text + (m.data as { text: string }).text }))
          else if (m.event === 'answer') setState((s) => ({ ...s, final: m.data as AnswerEvent }))
          else if (m.event === 'error')
            setState((s) => ({ ...s, status: 'error', error: (m.data as { message: string }).message }))
          else if (m.event === 'done') setState((s) => (s.status === 'error' ? s : { ...s, status: 'done' }))
        })
      } catch (e) {
        if (!c.signal.aborted) setState((s) => ({ ...s, status: 'error', error: String((e as Error).message ?? e) }))
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [url],
  )
  const reset = useCallback(() => {
    ctl.stop()
    setState(idle())
  }, [ctl])
  return { state, run, reset, stop: ctl.stop }
}

export interface CompareStream {
  status: 'idle' | 'running' | 'done' | 'error'
  question?: CompareQuestionEvent
  sides: Record<Side, AnswerStream>
  error?: string
}

/** Flat RAG vs RAPTOR, both answers streamed side by side. */
export function useCompare(url: string) {
  const blank = (): CompareStream => ({ status: 'idle', sides: { flat: idle(), raptor: idle() } })
  const [state, setState] = useState<CompareStream>(blank)
  const ctl = useAbortable()
  const patch = (side: Side, fn: (s: AnswerStream) => AnswerStream) =>
    setState((st) => ({ ...st, sides: { ...st.sides, [side]: fn(st.sides[side]) } }))

  const run = useCallback(
    async (body: CompareRequest) => {
      const c = ctl.next()
      setState({
        status: 'running',
        sides: { flat: { status: 'retrieving', text: '' }, raptor: { status: 'retrieving', text: '' } },
      })
      try {
        await streamSse(url, { method: 'POST', body, signal: c.signal }, (m) => {
          const d = m.data as { side?: Side }
          if (m.event === 'question') setState((s) => ({ ...s, question: m.data as CompareQuestionEvent }))
          else if (m.event === 'retrieval' && d.side)
            patch(d.side, (s) => ({ ...s, status: 'streaming', retrieval: m.data as RetrievalEvent }))
          else if (m.event === 'token' && d.side)
            patch(d.side, (s) => ({ ...s, text: s.text + (m.data as { text: string }).text }))
          else if (m.event === 'answer' && d.side)
            patch(d.side, (s) => ({ ...s, status: 'done', final: m.data as AnswerEvent }))
          else if (m.event === 'error') {
            const msg = (m.data as { message: string }).message
            if (d.side) patch(d.side, (s) => ({ ...s, status: 'error', error: msg }))
            else setState((s) => ({ ...s, status: 'error', error: msg }))
          } else if (m.event === 'done') setState((s) => (s.status === 'error' ? s : { ...s, status: 'done' }))
        })
      } catch (e) {
        if (!c.signal.aborted) setState((s) => ({ ...s, status: 'error', error: String((e as Error).message ?? e) }))
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [url],
  )
  return { state, run, stop: ctl.stop }
}

export interface CodeAskStream {
  status: 'idle' | 'streaming' | 'done' | 'error'
  text: string
  citations: CodeHit[]
  error?: string
}

export function useCodeAsk(url: string) {
  const [state, setState] = useState<CodeAskStream>({ status: 'idle', text: '', citations: [] })
  const ctl = useAbortable()
  const run = useCallback(
    async (question: string) => {
      const c = ctl.next()
      setState({ status: 'streaming', text: '', citations: [] })
      try {
        await streamSse(url, { method: 'POST', body: { question, k: 6 }, signal: c.signal }, (m) => {
          if (m.event === 'token') setState((s) => ({ ...s, text: s.text + (m.data as { text: string }).text }))
          else if (m.event === 'citations')
            setState((s) => ({ ...s, citations: (m.data as { citations: CodeHit[] }).citations }))
          else if (m.event === 'error')
            setState((s) => ({ ...s, status: 'error', error: (m.data as { message: string }).message }))
          else if (m.event === 'done') setState((s) => (s.status === 'error' ? s : { ...s, status: 'done' }))
        })
      } catch (e) {
        if (!c.signal.aborted) setState((s) => ({ ...s, status: 'error', error: String((e as Error).message ?? e) }))
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [url],
  )
  return { state, run }
}
