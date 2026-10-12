import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import realReport from "../../test/fixtures/study-report.json";
import realRuns from "../../test/fixtures/study-runs.json";
import realSpec from "../../test/fixtures/study-spec.json";
import reportMd from "../../test/fixtures/study-report.md?raw";
import realUpload from "../../test/fixtures/study-card-upload.json";
import { installEventSource, lastSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { Study } from "./Study";

// Real API output for a finished record-mode study "toy" (variants small and wide, three seeds, one
// prediction): web/test/fixtures/study-*.json, written by a Python test against a real tiny run. Runs
// that are still going are those same entries with a different state and step.
const SPEC = realSpec.spec;
const report = realReport;
const base = realRuns;
const run = (variant: string, seed: number, state: string, val_bpb: number | null = null, step = 0) => ({
  ...base[0]!, ref: `studies/toy/${variant}/seed-${seed}`, state, step, val_bpb,
});
const finalRow = report.comparison.rows[1]!;
const prediction = report.predictions[0]!;

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  setLevel("Research");
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

const allRuns = (state: string) => (state === "done" ? base : ["small", "wide"].flatMap((v) => [0, 1, 2].map((s) => run(v, s, state))));

function open(runs: object[], extra: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/studies/toy/spec": { body: { name: "toy", path: "studies/toy.toml", spec: SPEC, toml: "" } },
    "GET /api/runs": { body: runs },
    "GET /api/studies/toy/report": { body: report },
    "GET /api/studies/toy/report.md": { body: reportMd },
    ...extra,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/studies/toy"]}>
        <Routes>
          <Route path="/studies/:name" element={<Study />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

const noReport = { "GET /api/studies/toy/report": { status: 422, problem: true, body: { title: "Cannot be done as asked", detail: "needs finished runs" } } };

describe("Study page", () => {
  it("fills the variant × seed grid live and counts finished runs", async () => {
    const done = base[0]!;
    open([done, run("small", 1, "running", null, 7)], noReport);
    expect(await screen.findByText("7 / 20")).toBeTruthy();
    expect(screen.getByLabelText(done.ref).textContent).toContain(done.val_bpb!.toFixed(3));
    expect(screen.getByText("of 6")).toBeTruthy();
    expect(screen.getByText(/1 running · 4 queued/)).toBeTruthy();
    act(() => lastSource().emit("step", { ref: "studies/toy/small/seed-1", step: 12 }));
    expect(await screen.findByText("12 / 20")).toBeTruthy();
    act(() => lastSource().emit("state", { ref: "studies/toy/small/seed-1", state: "done", step: 20, max_steps: 20 }));
    act(() => lastSource().emit("eval", { ref: "studies/toy/small/seed-1", step: 20, val_bpb: 2.5 }));
    await waitFor(() => expect(screen.getByLabelText("studies/toy/small/seed-1").textContent).toContain("2.500"));
    expect(screen.queryByRole("region", { name: "Comparison" })).toBeNull();
  });

  it("shows the comparison, forest plot, predictions, report and bundle once runs have finished", async () => {
    open(allRuns("done"));
    const comparison = await screen.findByRole("region", { name: "Comparison" });
    expect(within(comparison).getByText(finalRow.text.delta)).toBeTruthy();
    expect(within(comparison).getAllByText(/no shipped baselines with 3 or more seeds/).length).toBeGreaterThan(0);
    expect(within(comparison).getByRole("img", { name: /Δ val_bpb vs small, 95% CI/ })).toBeTruthy();
    const predictions = screen.getByRole("region", { name: "Predictions" });
    expect(within(predictions).getByText(prediction.predicted.toFixed(3))).toBeTruthy();
    expect(within(predictions).getByText(new RegExp(`committed in ${report.manifest.preregistration_commit.slice(0, 7)} before the first run`))).toBeTruthy();
    expect(await screen.findByText("Study: toy")).toBeTruthy();
    const link = screen.getAllByRole("link", { name: /bundle/i }).at(-1) as HTMLAnchorElement;
    expect(link.getAttribute("href")).toBe("/api/studies/toy/bundle.zip");
    expect(screen.queryByRole("button", { name: "Stop study" })).toBeNull();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("stops a running study", async () => {
    const seen = open([run("small", 0, "running", null, 4)], {
      ...noReport,
      "POST /api/studies/toy/stop": { body: { study: "toy", stopping: [] } },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Stop study" }));
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path === "/api/studies/toy/stop")).toBe(true));
  });

  it("offers the ablation card for a finished record study only", async () => {
    open(allRuns("done"), { "GET /api/studies/toy/card/upload": { body: realUpload } });
    await userEvent.click(await screen.findByRole("button", { name: "Export ablation card…" }));
    expect(await screen.findByText("Ablation card for toy")).toBeTruthy();
  });

  it("hides the card and builder controls below Research", async () => {
    setLevel("Tinker");
    open(allRuns("done"));
    await screen.findByRole("region", { name: "Comparison" });
    expect(screen.queryByRole("button", { name: "Export ablation card…" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Open in builder" })).toBeNull();
  });

  it("has no card button while runs are still going", async () => {
    open([run("small", 0, "running", null, 5)], noReport);
    await screen.findByRole("button", { name: "Stop study" });
    expect(screen.queryByRole("button", { name: "Export ablation card…" })).toBeNull();
  });

  it("shows the plan's grid for a study that is queued and has no run folder yet", async () => {
    open([], { "GET /api/studies/toy/report": { status: 404, problem: true, body: { title: "Not found", detail: "no study 'toy' has run" } } });
    expect(await screen.findByText("Variants × seeds")).toBeTruthy();
    expect(screen.getAllByText("queued").length).toBeGreaterThanOrEqual(6);
    expect(screen.queryByText(/no study 'toy' has run/)).toBeNull();
  });
});
