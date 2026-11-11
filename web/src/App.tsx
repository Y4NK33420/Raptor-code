import { lazy, Suspense } from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router'
import { AppShell } from './components/layout/AppShell'
import { Skeleton } from './components/ui/primitives'
import { LibraryPage } from './pages/Library'

const BuildPage = lazy(() => import('./pages/Build'))
const WorkspacePage = lazy(() => import('./pages/Workspace'))
const ExperimentsPage = lazy(() => import('./pages/Experiments'))
const CodePage = lazy(() => import('./pages/Code'))

function Loading() {
  return (
    <div className="space-y-3 p-8">
      <Skeleton className="h-9 w-64" />
      <Skeleton className="h-[60vh] w-full" />
    </div>
  )
}

const wrap = (el: React.ReactNode) => <Suspense fallback={<Loading />}>{el}</Suspense>

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: '/', element: <LibraryPage /> },
      { path: '/build/:jobId', element: wrap(<BuildPage />) },
      { path: '/indexes/:id/:tab?', element: wrap(<WorkspacePage />) },
      { path: '/experiments', element: wrap(<ExperimentsPage />) },
      { path: '/code/:repoId?', element: wrap(<CodePage />) },
      {
        path: '*',
        element: (
          <div className="p-10 text-center">
            <p className="font-display text-2xl">Nothing grows here.</p>
            <a className="mt-2 inline-block text-accent underline" href="/">
              Back to the library
            </a>
          </div>
        ),
      },
    ],
  },
])

export function App() {
  return <RouterProvider router={router} />
}
