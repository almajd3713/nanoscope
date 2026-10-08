import { chromium, type FullConfig } from "@playwright/test";

// Sign in once through the login link the server prints, and keep the cookie for every test.
export default async function globalSetup(config: FullConfig): Promise<void> {
  const baseURL = config.projects[0]?.use.baseURL ?? "http://127.0.0.1:18765";
  const token = process.env["E2E_TOKEN"] ?? "e2e-token";
  const browser = await chromium.launch();
  const context = await browser.newContext({ baseURL });
  const page = await context.newPage();
  await page.goto(`/login?token=${encodeURIComponent(token)}`);
  await context.storageState({ path: "e2e/.auth.json" });
  await browser.close();
}
