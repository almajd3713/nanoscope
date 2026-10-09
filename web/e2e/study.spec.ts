import { expect, test } from "@playwright/test";
import { readZip, saveStudy, studyToml, waitStudyDone } from "./study";

const NAME = `e2e-forest-${Date.now().toString(36)}`;

type Row = { label: string; delta: { mean: number; ci95_low: number | null; ci95_high: number | null } | null };

// Plan done-when: the forest plot shows the numbers in the study's results.json.
test("the forest plot values equal results.json", async ({ page, request }) => {
  await saveStudy(request, studyToml(NAME, "explore"));
  const started = await request.post(`/api/studies/${NAME}/run`);
  expect(started.ok(), await started.text()).toBe(true);

  await page.goto(`/studies/${NAME}`);
  await expect(page.getByRole("region", { name: "Variants and seeds" })).toBeVisible();
  await waitStudyDone(request, NAME, 6);

  const bundle = await request.get(`/api/studies/${NAME}/bundle.zip`);
  const results = JSON.parse(readZip(await bundle.body())["results.json"]!) as { rows: Row[] };
  const expected = results.rows
    .filter((r) => r.delta)
    .map((r) => {
      const d = r.delta!;
      const ci = d.ci95_low !== null ? ` [${d.ci95_low.toFixed(3)}, ${d.ci95_high!.toFixed(3)}]` : "";
      return `${r.label}: ${d.mean.toFixed(3)}${ci}`;
    })
    .join("; ");
  expect(expected).not.toBe("");

  const plot = page.getByRole("region", { name: "Comparison" }).getByRole("img", { name: /Δ val_bpb vs small/ });
  await expect(plot).toHaveAttribute("aria-label", new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  // every finished run is in the grid with its final value
  await expect(page.getByRole("link", { name: `studies/${NAME}/wide/seed-2` })).toContainText("done");
});
