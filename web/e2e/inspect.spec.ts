import { expect, test } from "@playwright/test";

// Plan done-when: the inspect page shows attention maps from the learner's own run at any
// archived step. The run is a 2-layer Modern trained for a few steps with two archived steps; the
// page then reads each of them through an inspect job a worker runs.
const KWARGS = { max_steps: 12, eval_interval: 6, checkpoint_interval: 6, sample_interval: 12, batch_size: 4, d_model: 32, n_layers: 2, n_heads: 2 };

test("the inspect page shows attention maps from the learner's own run at an archived step", async ({ page, request }) => {
  const sent = await request.post("/api/runs", {
    data: { model: "nanoscope.models.modern:Modern", preset: "tinystories-5min", seed: 0, kwargs: KWARGS, compile: false, wandb: false, checkpoint_steps: [4, 8] },
  });
  expect(sent.ok(), await sent.text()).toBe(true);
  const { ref } = (await sent.json()) as { ref: string };
  await expect
    .poll(async () => ((await (await request.get(`/api/runs/${ref}`)).json()) as { status?: { state: string } }).status?.state, { timeout: 300_000, intervals: [2000] })
    .toBe("done");

  // from the run page
  await page.goto(`/runs/${ref}`);
  await page.getByRole("link", { name: "Inspect" }).click();
  await page.waitForURL(/\/inspect\//);

  // the latest checkpoint first: a map for every head of every layer
  await expect(page.getByRole("heading", { name: "Attention", exact: true })).toBeVisible({ timeout: 120_000 });
  for (const module of ["blocks.0.attn", "blocks.1.attn"]) {
    for (const head of [0, 1]) await expect(page.getByRole("button", { name: `${module} head ${head}` })).toBeVisible();
  }
  await expect(page.getByRole("table", { name: "Logit lens by layer" })).toBeVisible();
  await expect(page.getByText("from the step 12 checkpoint")).toBeVisible();

  // an archived step: the same prompt again, read at step 4
  await page.getByRole("radio", { name: "4", exact: true }).click();
  await expect(page.getByText("from the step 4 checkpoint")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole("button", { name: "blocks.1.attn head 1" })).toBeVisible();

  // across steps: one job per archived step
  await page.getByRole("radio", { name: "Across steps" }).click();
  await expect(page.getByText(/2 of 2 jobs done/)).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("step", { exact: true }).first()).toBeVisible();
});
