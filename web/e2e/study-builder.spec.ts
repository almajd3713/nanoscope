import { expect, test } from "@playwright/test";
import { saveFile } from "./helpers";
import { follow } from "./study";

const NAME = `e2e-built-${Date.now().toString(36)}`;
// The same study as the form below, written in Python.
const PY = `from nanoscope import Study, Tokens
from nanoscope.models import Bigram

study = Study("${NAME}", preset="tinystories-5min", seeds=[0, 1, 2], budget=Tokens(20480), baseline="small")
study.add("small", Bigram, d_model=8)
study.add("wide", Bigram, d_model=32)
`;

// Plan done-when: a study made in the builder is the study the library derives from the equivalent
// .py file, so it queues the same runs.
test("a builder-made study has the spec of its .py twin and queues the same runs", async ({ page, request }) => {
  await saveFile(request, `studies/${NAME.replace(/-/g, "_")}.py`, PY);
  const queued = await request.post("/api/studies/from-file", { data: { file: `studies/${NAME.replace(/-/g, "_")}.py` } });
  expect(queued.ok(), await queued.text()).toBe(true);
  const twin = await follow<{ spec: Record<string, unknown> }>(request, ((await queued.json()) as { id: number }).id);

  await page.goto("/studies/new");
  await page.getByLabel("Name", { exact: true }).fill(NAME);
  await page.getByLabel("Budget (tokens)").fill("20480");
  const add = page.getByRole("button", { name: "Add variant" });
  await add.click();
  await add.click();
  const names = page.getByLabel(/^Name of variant/);
  await names.nth(0).fill("small");
  await names.nth(1).fill("wide");
  await page.getByLabel("Keywords of small").fill("d_model = 8");
  await page.getByLabel("Keywords of wide").fill("d_model = 32");
  for (const variant of ["small", "wide"]) await page.getByLabel(`Model of ${variant}`).selectOption("nanoscope.models.bigram:Bigram");
  await page.getByRole("combobox", { name: /^Baseline/ }).selectOption("small");

  const save = page.getByRole("button", { name: "Save as TOML" });
  await expect(save).toBeEnabled();
  await save.click();
  await expect(page).toHaveURL(new RegExp(`/studies/${NAME}/edit$`));

  const built = (await (await request.get(`/api/studies/${NAME}/spec`)).json()) as { spec: Record<string, unknown> };
  expect(built.spec).toEqual(twin.spec);

  await page.getByRole("button", { name: "Train 6 runs" }).click();
  await expect(page).toHaveURL(new RegExp(`/studies/${NAME}$`));
  await expect
    .poll(async () => {
      const all = (await (await request.get("/api/studies")).json()) as { name: string; runs_total: number }[];
      return all.find((s) => s.name === NAME)?.runs_total;
    }, { timeout: 60_000 })
    .toBe(6);
  const wanted = ["small", "wide"].flatMap((v) => [0, 1, 2].map((s) => `studies/${NAME}/${v}/seed-${s}`)).sort();
  await expect
    .poll(async () => {
      const jobs = (await (await request.get("/api/jobs?kind=run&limit=200")).json()) as { ref: string }[];
      return jobs.map((j) => j.ref).filter((r) => r?.startsWith(`studies/${NAME}/`)).sort();
    }, { timeout: 60_000 })
    .toEqual(wanted);
});
