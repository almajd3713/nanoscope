import { defineConfig } from "@playwright/test";

// The tests run against a compose stack that is already up (e2e/stack.sh up). Beyond loopback the
// API wants its token, so the setup signs in once through the login link and saves the cookie.
const port = process.env["E2E_PORT"] ?? "18765";

export default defineConfig({
  testDir: "e2e",
  timeout: 360_000,
  expect: { timeout: 15_000 },
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: process.env["E2E_URL"] ?? `http://127.0.0.1:${port}`,
    storageState: "e2e/.auth.json",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    // lesson 1 goes first: it counts the clicks of a fresh start
    { name: "first-run", testMatch: /lesson1\.spec\.ts/, use: { browserName: "chromium" } },
    { name: "rest", testIgnore: /lesson1\.spec\.ts/, dependencies: ["first-run"], use: { browserName: "chromium" } },
  ],
});
