import type * as T from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<R>(path: string, init?: RequestInit): Promise<R> {
  const res = await fetch(`/api${path}`, init)
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, msg)
  }
  if (res.status === 204) return undefined as R
  return (await res.json()) as R
}

const qs = (params: Record<string, string | undefined>) => {
  const s = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined) s.set(k, v)
  return s.toString()
}
const enc = encodeURIComponent

export const api = {
  info: () => request<T.ServerInfo>('/info'),
  presets: () => request<T.Preset[]>('/presets'),
  indexes: () => request<T.IndexInfo[]>('/indexes'),
  index: (id: string) => request<T.IndexInfo>(`/indexes/${enc(id)}`),
  deleteIndex: (id: string) => request<void>(`/indexes/${enc(id)}`, { method: 'DELETE' }),
  createIndex: (form: FormData) =>
    request<T.BuildAccepted>('/indexes', { method: 'POST', body: form }),
  job: (id: string) => request<T.JobInfo>(`/jobs/${enc(id)}`),
  jobs: () => request<T.JobInfo[]>('/jobs'),
  tree: (id: string) => request<T.TreeData>(`/indexes/${enc(id)}/tree`),
  node: (id: string, nid: number) => request<T.NodeDetail>(`/indexes/${enc(id)}/nodes/${nid}`),
  projection: (id: string) => request<T.Projection>(`/indexes/${enc(id)}/projection`),
  stories: () => request<T.QualityStory[]>('/quality/stories'),
  story: (setId: string) => request<T.QualityStoryDetail>(`/quality/stories/${enc(setId)}`),
  runs: () => request<T.RunSummary[]>('/runs'),
  run: (id: string) => request<T.RunDetail>(`/runs/${enc(id)}`),
  compareRuns: (a: string, b: string) => request<T.RunCompare>(`/runs/compare?${qs({ a, b })}`),
  codeRepos: () => request<T.CodeRepo[]>('/code'),
  codeFiles: (id: string) => request<T.CodeFile[]>(`/code/${enc(id)}/files`),
  codeFile: (id: string, path: string) =>
    request<T.CodeFileText>(`/code/${enc(id)}/file?${qs({ path })}`),
  codeSymbol: (id: string, nid: number) => request<T.CodeSymbol>(`/code/${enc(id)}/symbol/${nid}`),
  codeGraph: (id: string, kinds?: string) =>
    request<T.CodeGraph>(`/code/${enc(id)}/graph?${qs({ kinds })}`),
}

export const urls = {
  jobEvents: (id: string) => `/api/jobs/${enc(id)}/events`,
  query: (id: string) => `/api/indexes/${enc(id)}/query`,
  compare: (id: string) => `/api/indexes/${enc(id)}/compare`,
  codeAsk: (id: string) => `/api/code/${enc(id)}/ask`,
}
