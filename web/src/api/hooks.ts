import { useQuery } from '@tanstack/react-query'
import { api } from './client'

const forever = { staleTime: Infinity }

export const useInfo = () => useQuery({ queryKey: ['info'], queryFn: api.info, ...forever })
export const usePresets = () => useQuery({ queryKey: ['presets'], queryFn: api.presets, ...forever })
export const useIndexes = () => useQuery({ queryKey: ['indexes'], queryFn: api.indexes })
export const useIndex = (id: string) =>
  useQuery({ queryKey: ['index', id], queryFn: () => api.index(id) })
export const useTree = (id: string) =>
  useQuery({ queryKey: ['tree', id], queryFn: () => api.tree(id), staleTime: 60_000 })
export const useNode = (id: string, nid: number | null) =>
  useQuery({
    queryKey: ['node', id, nid],
    queryFn: () => api.node(id, nid as number),
    enabled: nid !== null,
    staleTime: 60_000,
  })
export const useProjection = (id: string, enabled = true) =>
  useQuery({ queryKey: ['projection', id], queryFn: () => api.projection(id), enabled, ...forever })
export const useStories = () => useQuery({ queryKey: ['stories'], queryFn: api.stories })
export const useStory = (setId: string | null | undefined) =>
  useQuery({
    queryKey: ['story', setId],
    queryFn: () => api.story(setId as string),
    enabled: !!setId,
    ...forever,
  })
export const useRuns = () => useQuery({ queryKey: ['runs'], queryFn: api.runs })
export const useRun = (id: string | null) =>
  useQuery({ queryKey: ['run', id], queryFn: () => api.run(id as string), enabled: !!id })
export const useRunCompare = (a: string | null, b: string | null) =>
  useQuery({
    queryKey: ['run-compare', a, b],
    queryFn: () => api.compareRuns(a as string, b as string),
    enabled: !!a && !!b && a !== b,
  })
export const useCodeRepos = () => useQuery({ queryKey: ['code'], queryFn: api.codeRepos })
export const useCodeFiles = (id: string | undefined) =>
  useQuery({ queryKey: ['code-files', id], queryFn: () => api.codeFiles(id!), enabled: !!id })
export const useCodeFile = (id: string | undefined, path: string | null) =>
  useQuery({
    queryKey: ['code-file', id, path],
    queryFn: () => api.codeFile(id!, path!),
    enabled: !!id && !!path,
    ...forever,
  })
export const useCodeSymbol = (id: string | undefined, nid: number | null) =>
  useQuery({
    queryKey: ['code-symbol', id, nid],
    queryFn: () => api.codeSymbol(id!, nid!),
    enabled: !!id && nid !== null,
  })
export const useCodeGraph = (id: string | undefined, kinds: string) =>
  useQuery({
    queryKey: ['code-graph', id, kinds],
    queryFn: () => api.codeGraph(id!, kinds),
    enabled: !!id,
  })
