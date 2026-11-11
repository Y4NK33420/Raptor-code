import { useMutation } from '@tanstack/react-query'
import { BookOpen, Check, ClipboardType, Search, Upload } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { api } from '@/api/client'
import { usePresets, useStories } from '@/api/hooks'
import type { Preset } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Badge, ErrorNote, Input, Label, Skeleton, Textarea } from '@/components/ui/primitives'
import { Dialog, DialogContent, Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/radix'
import { cn, compact } from '@/lib/utils'

type Source = 'text' | 'file' | 'quality'

/** Rough LLM-call estimate: chunks ≈ tokens / chunk size; each layer has ~1/5 as many nodes. */
export function estimateCalls(chars: number, preset: Preset | undefined): number {
  if (!preset || chars <= 0) return 0
  const cfg = preset.config as { chunking: { max_tokens: number }; builder: { max_layers: number } }
  let nodes = Math.max(1, Math.ceil(chars / 4 / cfg.chunking.max_tokens))
  let calls = 0
  for (let l = 0; l < cfg.builder.max_layers && nodes > 11; l++) {
    nodes = Math.max(1, Math.round(nodes / 5))
    calls += nodes
  }
  return calls
}

export function NewIndexDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const navigate = useNavigate()
  const presets = usePresets()
  const stories = useStories()
  const [source, setSource] = useState<Source>('quality')
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [fileChars, setFileChars] = useState(0)
  const [setId, setSetId] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [preset, setPreset] = useState('quick')
  const [name, setName] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const story = stories.data?.find((s) => s.set_id === setId)
  const chars = source === 'text' ? text.length : source === 'file' ? fileChars : (story?.word_count ?? 0) * 5.6
  const chosenPreset = presets.data?.find((p) => p.id === preset)
  const calls = estimateCalls(chars, chosenPreset)

  const visibleStories = useMemo(() => {
    const seen = new Set<string>()
    const f = filter.toLowerCase()
    return (stories.data ?? []).filter((s) => {
      if (seen.has(s.article_id)) return false // the same article appears in several question sets
      seen.add(s.article_id)
      return !f || s.title.toLowerCase().includes(f) || s.author.toLowerCase().includes(f)
    })
  }, [stories.data, filter])

  const ready =
    (source === 'text' && text.trim().length >= 20) ||
    (source === 'file' && !!file) ||
    (source === 'quality' && !!setId)

  const create = useMutation({
    mutationFn: () => {
      const form = new FormData()
      form.set('source_type', source)
      form.set('preset', preset)
      if (name.trim()) form.set('name', name.trim())
      if (source === 'text') form.set('text', text)
      if (source === 'file' && file) form.set('file', file)
      if (source === 'quality' && setId) form.set('set_id', setId)
      return api.createIndex(form)
    },
    onSuccess: (r) => {
      onOpenChange(false)
      navigate(`/build/${r.job_id}?index=${encodeURIComponent(r.index_id)}`)
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        title="New index"
        description="Pick a document and a strategy. The build streams live, layer by layer."
        className="sm:w-[min(760px,calc(100vw-32px))]"
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            if (ready) create.mutate()
          }}
          className="space-y-5"
        >
          <Tabs value={source} onValueChange={(v) => setSource(v as Source)}>
            <TabsList className="w-full sm:w-auto">
              <TabsTrigger value="quality" className="flex-1">
                <BookOpen /> QuALITY story
              </TabsTrigger>
              <TabsTrigger value="text" className="flex-1">
                <ClipboardType /> Paste
              </TabsTrigger>
              <TabsTrigger value="file" className="flex-1">
                <Upload /> Upload
              </TabsTrigger>
            </TabsList>

            <TabsContent value="quality" className="mt-3">
              <div className="relative mb-2">
                <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted" />
                <Input
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="Search stories by title or author"
                  className="pl-9"
                  aria-label="Search stories"
                />
              </div>
              <div
                role="radiogroup"
                aria-label="QuALITY story"
                className="scrollbar-thin max-h-60 space-y-1 overflow-y-auto rounded-xl border border-line p-1"
              >
                {stories.isLoading && <Skeleton className="h-40" />}
                {stories.data?.length === 0 && (
                  <p className="p-4 text-sm text-muted">
                    QuALITY data not found. Run <code>python scripts/download_quality.py</code>.
                  </p>
                )}
                {visibleStories.map((s) => (
                  <button
                    type="button"
                    role="radio"
                    aria-checked={setId === s.set_id}
                    key={s.set_id}
                    onClick={() => setSetId(s.set_id)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition hover:bg-surface-2',
                      setId === s.set_id && 'bg-accent-soft hover:bg-accent-soft',
                    )}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-ink">{s.title}</div>
                      <div className="truncate text-xs text-muted">
                        {s.author} · <span className="num">{compact(s.word_count)}</span> words ·{' '}
                        {s.n_questions} questions
                      </div>
                    </div>
                    {s.index_ids.length > 0 && <Badge tone="outline">indexed</Badge>}
                    {setId === s.set_id && <Check className="size-4 text-accent" />}
                  </button>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="text" className="mt-3">
              <Label htmlFor="ni-text">Document text</Label>
              <Textarea
                id="ni-text"
                rows={8}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste an article, a chapter, meeting notes…"
              />
              <p className="mt-1 text-xs text-muted">
                <span className="num">{compact(text.length)}</span> characters
              </p>
            </TabsContent>

            <TabsContent value="file" className="mt-3">
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  const f = e.dataTransfer.files[0]
                  if (f) {
                    setFile(f)
                    setFileChars(f.size)
                  }
                }}
                className="flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed border-line-strong p-8 text-center transition hover:border-accent hover:bg-accent-soft/40"
              >
                <Upload className="size-6 text-muted" />
                <span className="text-sm text-ink">{file ? file.name : 'Drop a .txt or .md file, or click to choose'}</span>
                {file && <span className="num text-xs text-muted">{compact(file.size)} bytes</span>}
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".txt,.md,.markdown,text/plain,text/markdown"
                className="hidden"
                aria-label="Choose file"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null
                  setFile(f)
                  setFileChars(f?.size ?? 0)
                }}
              />
            </TabsContent>
          </Tabs>

          <fieldset>
            <legend className="mb-2 text-[13px] font-medium text-ink-2">Strategy preset</legend>
            <div className="grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Strategy preset">
              {presets.data?.map((p) => (
                <button
                  type="button"
                  role="radio"
                  aria-checked={preset === p.id}
                  key={p.id}
                  onClick={() => setPreset(p.id)}
                  className={cn(
                    'rounded-xl border border-line p-3 text-left transition hover:border-line-strong',
                    preset === p.id && 'border-accent bg-accent-soft/50 ring-1 ring-accent',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-ink">{p.label}</span>
                    <span className="num text-[11px] text-muted">
                      {(p.config as { chunking: { max_tokens: number } }).chunking.max_tokens} tok ·{' '}
                      {(p.config as { builder: { max_layers: number } }).builder.max_layers} layers
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-muted">{p.description}</p>
                </button>
              ))}
            </div>
          </fieldset>

          <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
            <div>
              <Label htmlFor="ni-name">Name (optional)</Label>
              <Input
                id="ni-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={story?.title ?? file?.name.replace(/\.\w+$/, '') ?? 'Pasted text'}
              />
            </div>
            <p className="text-xs text-muted sm:pb-3 sm:text-right" aria-live="polite">
              {calls > 0 ? (
                <>
                  ≈ <span className="num text-ink">{calls}</span> summary calls
                </>
              ) : (
                ' '
              )}
            </p>
          </div>

          {create.error && <ErrorNote error={create.error} />}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" disabled={!ready || create.isPending}>
              {create.isPending ? 'Starting…' : 'Build index'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
