import { expect, test } from '@playwright/test'

const STORY = `Once upon a time a girl named Ella lived with her stepmother. ${'The stepmother was cruel and made Ella clean the hearth every day. '.repeat(30)}
A fairy godmother sent Ella to the royal ball in a gown of silver. ${'The prince danced with Ella until midnight and then she fled. '.repeat(30)}
The prince found her by a glass slipper and they were married. ${'The kingdom celebrated for three days with music and feasts. '.repeat(30)}`

test('build an index, ask, and see retrieved nodes light up', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: /new index/i }).first().click()
  await page.getByRole('tab', { name: /paste/i }).click()
  await page.getByLabel('Document text').fill(STORY)
  await page.getByRole('button', { name: 'Build index' }).click()

  await expect(page).toHaveURL(/\/build\//)
  await expect(page.getByText('complete', { exact: true })).toBeVisible({ timeout: 60_000 })
  await page.getByRole('link', { name: /explore the tree/i }).first().click()

  await expect(page).toHaveURL(/\/indexes\//)
  await page.getByLabel('Question').fill('Who does Ella live with?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.getByTestId('answer')).toContainText(/\w+/, { timeout: 30_000 })
  await expect(page.getByText(/nodes lit up in the tree/)).toBeVisible()
  await expect(page.getByLabel('Retrieved nodes').getByRole('button').first()).toBeVisible()
})
