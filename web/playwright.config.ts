import { resolve } from 'node:path'
import { defineConfig } from '@playwright/test'

const exe = resolve(import.meta.dirname, '..', '.venv', process.platform === 'win32' ? 'Scripts/raptor.exe' : 'bin/raptor')

const port = 8791
export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  use: { baseURL: `http://127.0.0.1:${port}`, channel: process.env.PW_CHANNEL || undefined },
  webServer: {
    command: `"${exe}" serve --fake-models --fake-delay 0.005 --port ${port} --web-dist dist`,
    url: `http://127.0.0.1:${port}/api/info`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
})
