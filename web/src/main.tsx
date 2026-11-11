import '@fontsource-variable/fraunces'
import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import '@xyflow/react/dist/style.css'
import './index.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { TooltipProvider } from './components/ui/radix'
import { ThemeProvider } from './hooks/useTheme'

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <App />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
)
