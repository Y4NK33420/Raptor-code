import { useMutation, useQueryClient } from '@tanstack/react-query'
import { BookOpen, FileText, Plus, Sparkles, Trash2 } from 'lucide-react'
import { motion } from 'motion/react'
import { useState } from 'react'
import { Link } from 'react-router'
import { api } from '@/api/client'
import { useIndexes } from '@/api/hooks'
import type { IndexInfo } from '@/api/types'
import { PageHeader } from '@/components/layout/AppShell'
import { NewIndexDialog } from '@/components/NewIndexDialog'
import { Strata } from '@/components/Strata'
import { Button } from '@/components/ui/button'
import { Badge, Card, Empty, ErrorNote, Skeleton } from '@/components/ui/primitives'
import { Tip } from '@/components/ui/radix'
import { compact, timeAgo } from '@/lib/utils'

function IndexCard({ index, i }: { index: IndexInfo; i: number }) {
  const qc = useQueryClient()
  const del = useMutation({
    mutationFn: () => api.deleteIndex(index.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['indexes'] }),
  })
  const words = Math.round(index.leaf_chars / 5.6)
  const src = index.source
  return (
    <motion.li
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(i * 0.04, 0.4) }}
      className="list-none"
    >
      <Card className="group relative flex h-full flex-col overflow-hidden transition hover:border-line-strong">
        <Link
          to={`/indexes/${encodeURIComponent(index.id)}`}
          className="flex flex-1 flex-col p-4 outline-none after:absolute after:inset-0 after:rounded-2xl focus-visible:after:ring-2 focus-visible:after:ring-accent"
          aria-label={`Open ${index.name}`}
        >
          <div className="mb-3 flex items-center gap-1.5">
            {src.type === 'quality' ? (
              <Badge tone="accent">
                <BookOpen className="size-3" /> QuALITY
              </Badge>
            ) : (
              <Badge>
                <FileText className="size-3" /> {src.filename ? 'Upload' : 'Text'}
              </Badge>
            )}
            {index.origin === 'demo' && <Badge tone="outline">demo</Badge>}
            {index.preset && <Badge tone="outline">{index.preset}</Badge>}
          </div>
          <h3 className="font-display text-xl leading-snug font-semibold text-ink">{index.name}</h3>
          {src.type === 'quality' && src.title && src.title !== index.name && (
            <p className="mt-0.5 text-sm text-muted">{src.title}</p>
          )}
          <div className="my-5 flex h-14 items-end">
            <Strata counts={index.layer_counts} className="w-full" />
          </div>
          <dl className="mt-auto grid grid-cols-3 gap-2 border-t border-line pt-3 text-xs">
            <div>
              <dt className="text-muted">nodes</dt>
              <dd className="num text-sm text-ink">{index.num_nodes}</dd>
            </div>
            <div>
              <dt className="text-muted">layers</dt>
              <dd className="num text-sm text-ink">{index.num_layers + 1}</dd>
            </div>
            <div>
              <dt className="text-muted">words</dt>
              <dd className="num text-sm text-ink">{compact(words)}</dd>
            </div>
          </dl>
          <p className="mt-2 text-[11px] text-muted">{timeAgo(index.created_at)}</p>
        </Link>
        {index.origin === 'user' && (
          <Tip content="Delete index">
            <Button
              variant="danger"
              size="icon-sm"
              className="absolute top-3 right-3 z-10 opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
              aria-label={`Delete ${index.name}`}
              disabled={del.isPending}
              onClick={() => {
                if (window.confirm(`Delete "${index.name}"? This cannot be undone.`)) del.mutate()
              }}
            >
              <Trash2 />
            </Button>
          </Tip>
        )}
      </Card>
    </motion.li>
  )
}

export function LibraryPage() {
  const indexes = useIndexes()
  const [open, setOpen] = useState(false)
  const total = indexes.data?.reduce((a, i) => a + i.num_nodes, 0) ?? 0
  return (
    <div className="mx-auto max-w-7xl pb-10">
      <PageHeader
        eyebrow="Library"
        title="Every document, as a tree of summaries"
        actions={
          <Button variant="primary" size="lg" onClick={() => setOpen(true)}>
            <Plus /> New index
          </Button>
        }
      >
        RAPTOR chunks a document, clusters similar chunks, and summarises each cluster, layer after
        layer. Questions are answered from whichever layers fit best.
        {indexes.data && (
          <span className="text-muted">
            {' '}
            {indexes.data.length} indexes · <span className="num">{compact(total)}</span> nodes.
          </span>
        )}
      </PageHeader>
      <div className="px-4 sm:px-8">
        {indexes.error && <ErrorNote error={indexes.error} />}
        {indexes.isLoading && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-64 rounded-2xl" />
            ))}
          </div>
        )}
        {indexes.data && indexes.data.length === 0 && (
          <Card>
            <Empty icon={<Sparkles />} title="No indexes yet">
              Build one from pasted text, a file, or a QuALITY story.
            </Empty>
          </Card>
        )}
        {indexes.data && indexes.data.length > 0 && (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" aria-label="Indexes">
            {indexes.data.map((ix, i) => (
              <IndexCard key={ix.id} index={ix} i={i} />
            ))}
          </ul>
        )}
      </div>
      <NewIndexDialog open={open} onOpenChange={setOpen} />
    </div>
  )
}
