export interface SseMessage<T = unknown> {
  event: string
  data: T
  id?: string
}

/** Incremental parser for a text/event-stream body. Feed it chunks; it returns complete events. */
export class SseParser {
  private buf = ''

  push(chunk: string): SseMessage[] {
    this.buf += chunk.replace(/\r\n/g, '\n')
    const out: SseMessage[] = []
    let sep = this.buf.indexOf('\n\n')
    while (sep >= 0) {
      const block = this.buf.slice(0, sep)
      this.buf = this.buf.slice(sep + 2)
      const msg = parseBlock(block)
      if (msg) out.push(msg)
      sep = this.buf.indexOf('\n\n')
    }
    return out
  }
}

export function parseBlock(block: string): SseMessage | null {
  let event = 'message'
  let id: string | undefined
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue
    const i = line.indexOf(':')
    const field = i < 0 ? line : line.slice(0, i)
    const value = i < 0 ? '' : line.slice(i + 1).replace(/^ /, '')
    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
    else if (field === 'id') id = value
  }
  if (!data.length) return null
  const raw = data.join('\n')
  let parsed: unknown = raw
  try {
    parsed = JSON.parse(raw)
  } catch {
    /* plain-text payload */
  }
  return { event, data: parsed, id }
}

/** GET or POST a URL and call `onEvent` for every server-sent event until the stream ends. */
export async function streamSse(
  url: string,
  opts: {
    method?: 'GET' | 'POST'
    body?: unknown
    signal?: AbortSignal
    headers?: Record<string, string>
  },
  onEvent: (msg: SseMessage) => void,
): Promise<void> {
  const res = await fetch(url, {
    method: opts.method ?? 'GET',
    headers: {
      Accept: 'text/event-stream',
      ...(opts.body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...opts.headers,
    },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    signal: opts.signal,
  })
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  const parser = new SseParser()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    for (const msg of parser.push(value)) onEvent(msg)
  }
}
