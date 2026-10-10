import { expect, test } from "@playwright/test";

// Plan done-when (P17.19): the induction-heads lesson shows a curve over checkpoints from the
// learner's own run. The lesson's repeated-sequence prompt goes through the same inspect page:
// the prompt is a phrase said twice, and "Across steps" reads one head's attention at every
// archived step, one inspect job per step.
const KWARGS = { max_steps: 12, eval_interval: 6, checkpoint_interval: 6, sample_interval: 12, batch_size: 4, d_model: 32, n_layers: 2, n_heads: 2 };
const PHRASE = "the cat sat on the mat and the cat sat on the mat and";

test("a repeated prompt is inspected at every archived step of the learner's own run", async ({ page, request }) => {
  const sent = await request.post("/api/runs", {
    data: { model: "nanoscope.models.modern:Modern", preset: "tinystories-5min", seed: 0, kwargs: KWARGS, compile: false, wandb: false, checkpoint_steps: [4, 8] },
  });
  expect(sent.ok(), await sent.text()).toBe(true);
  const { ref } = (await sent.json()) as { ref: string };
  await expect
    .poll(async () => ((await (await request.get(`/api/runs/${ref}`)).json()) as { status?: { state: string } }).status?.state, { timeout: 300_000, intervals: [2000] })
    .toBe("done");

  await page.goto(`/inspect/${ref}`);
  await page.getByLabel("Prompt", { exact: true }).fill(PHRASE);
  await page.getByRole("button", { name: "Inspect" }).click();
  await expect(page.getByRole("button", { name: "blocks.1.attn head 0" })).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/\d+ tokens; at most 64/)).toBeVisible();

  // the curve: the same head at both archived steps
  await page.getByRole("button", { name: "blocks.1.attn head 0" }).click();
  await page.getByRole("radio", { name: "Across steps" }).click();
  await expect(page.getByText(/2 of 2 jobs done/)).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/blocks\.1\.attn head 0, the same prompt at every archived step/)).toBeVisible();
});
