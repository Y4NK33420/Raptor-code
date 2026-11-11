import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/radix'
import { LayerMix, RetrievedList } from './Retrieval'
import { Strata } from './Strata'
import { CitedText } from '@/pages/Code'
import type { RetrievalEvent } from '@/api/types'

const retrieval: RetrievalEvent = {
  nodes: [
    { id: 1, layer: 0, score: 0.9, tokens: 100, preview: 'a leaf about the ball' },
    { id: 9, layer: 1, score: 0.5, tokens: 100, preview: 'a summary of the story' },
  ],
  layer_counts: { '0': 1, '1': 1 },
  context_tokens: 200,
  latency_s: 0.1,
}

describe('LayerMix', () => {
  it('reports the share of context that came from summaries', () => {
    render(
      <TooltipProvider>
        <LayerMix retrieval={retrieval} />
      </TooltipProvider>,
    )
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByRole('img')).toHaveAccessibleName(/Leaves 50%, Layer 1 50%/)
  })
})

describe('RetrievedList', () => {
  it('lets keyboard users pick a node', async () => {
    const onPick = vi.fn()
    render(<RetrievedList retrieval={retrieval} onPick={onPick} />)
    await userEvent.click(screen.getByText('a summary of the story'))
    expect(onPick).toHaveBeenCalledWith(9)
  })
})

describe('Strata', () => {
  it('describes tree shape for assistive tech', () => {
    render(<Strata counts={{ '0': 20, '1': 4 }} animate={false} />)
    expect(screen.getByRole('img')).toHaveAccessibleName(/layer 1 has 4 nodes/)
  })
})

describe('CitedText', () => {
  it('turns path:line citations into buttons that open that range', async () => {
    const onOpen = vi.fn()
    render(<CitedText text="See shop/checkout.py:19-32 and models.py:5." onOpen={onOpen} />)
    await userEvent.click(screen.getByRole('button', { name: 'shop/checkout.py:19-32' }))
    expect(onOpen).toHaveBeenCalledWith('shop/checkout.py', { start: 19, end: 32 })
    await userEvent.click(screen.getByRole('button', { name: 'models.py:5' }))
    expect(onOpen).toHaveBeenLastCalledWith('models.py', { start: 5, end: 5 })
  })
})
