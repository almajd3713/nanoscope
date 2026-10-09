import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installEventSource, lastSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { Study } from "./Study";

const SPEC = {
  schema: 1, name: "toy", preset: "tinystories-5min", match: "params", baseline: "small", seeds: [0, 1, 2], mode: "record", tolerance: 0.02,
  budget: { tokens: 4000000 },
  variants: [
    { name: "small", model: "nanoscope.models.bigram:Bigram" },
    { name: "no-rope", model: "nanoscope.models.modern:Modern" },
  ],
};
const run = (variant: string, seed: number, state: string, val_bpb: number | null = null, step = 0) => ({
  ref: `studies/toy/${variant}/seed-${seed}`, state, step, max_steps: 1954, val_bpb, updated: null, stale: false, error: null, model: "Bigram", started_at: "2026-10-08T21:02:00+00:00",
});
const row = (label: string, verdict: string, delta: object | null, text: object) => ({
  label, source: `runs/studies/toy/${label}`, seeds: [0, 1, 2], params: 1000, params_kind: "non_embedding", tokens: 4000000, verdict, delta,
  text: { params: "787,840", tokens: "4.0M", value: "0.885 ± 0.009", delta: "(baseline)", verdict, statement: null, ...text },
});
const report = {
  schema: 1, nanoscope: "0.5", study: "toy", mode: "record",
  manifest: { commit: "a".repeat(40), study_file: "studies/toy.toml", study_file_commit: "a99fe9aa0e70b58a5d011120093e0935157b4f35", study_file_committed_at: "x", started_at: "2026-10-08T21:02:11+00:00", committed_via: "nanoscope", preregistration_commit: "a99fe9a" },
  predictions: [{ variant: "no-rope", metric: "val_bpb", predicted: 1.09, actual: { mean: 1.009, ci95_low: 0.99, ci95_high: 1.03 }, error: -0.0743 }],
  rows: [],
  notes: [],
  comparison: {
    metric: "val_bpb", baseline: "small", title: "bits per byte on the first 200 validation documents", params_header: "Non-emb params",
    notes: ["1 variant is compared with one baseline."], curves: [], precision_plan: { text: "With 3 seeds per model, a difference is known to about ±0.034 bpb." },
    noise_floor: { text: "Seed noise on this preset is about 0.014 bpb (standard deviation between seeds of the same model, from the shipped baselines)." },
    rows: [
      row("small", "baseline", null, {}),
      row("no-rope", "worse", { mean: 0.124, ci95_low: 0.113, ci95_high: 0.136, n: 3, paired: true }, { value: "1.009 ± 0.010", delta: "+0.124 [+0.113, +0.136]", verdict: "worse", params: "787,840" }),
    ],
  },
};

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

const allRuns = (state: string) => ["small", "no-rope"].flatMap((v) => [0, 1, 2].map((s) => run(v, s, state, state === "done" ? 0.9 + s / 100 : null, state === "done" ? 1954 : 0)));

function open(runs: object[], extra: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/studies/toy/spec": { body: { name: "toy", path: "studies/toy.toml", spec: SPEC, toml: "" } },
    "GET /api/runs": { body: runs },
    "GET /api/studies/toy/report": { body: report },
    "GET /api/studies/toy/report.md": { body: "# Study: toy\n\n## Results\n" },
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

describe("Study page", () => {
  it("fills the variant × seed grid live and counts finished runs", async () => {
    open([run("small", 0, "done", 0.883, 1954), run("small", 1, "running", null, 559)], { "GET /api/studies/toy/report": { status: 422, problem: true, body: { title: "Cannot be done as asked", detail: "needs finished runs" } } });
    expect(await screen.findByText("559 / 1954")).toBeTruthy();
    expect(screen.getByLabelText("studies/toy/small/seed-0").textContent).toContain("0.883");
    expect(screen.getByText("of 6")).toBeTruthy();
    expect(screen.getByText(/1 running · 4 queued/)).toBeTruthy();
    act(() => lastSource().emit("step", { ref: "studies/toy/small/seed-1", step: 700 }));
    expect(await screen.findByText("700 / 1954")).toBeTruthy();
    act(() => lastSource().emit("state", { ref: "studies/toy/small/seed-1", state: "done", step: 1954, max_steps: 1954 }));
    act(() => lastSource().emit("eval", { ref: "studies/toy/small/seed-1", step: 1954, val_bpb: 0.881 }));
    await waitFor(() => expect(screen.getByLabelText("studies/toy/small/seed-1").textContent).toContain("0.881"));
    expect(screen.queryByRole("region", { name: "Comparison" })).toBeNull();
  });

  it("shows the comparison, forest plot, predictions, report and bundle once runs have finished", async () => {
    open(allRuns("done"));
    const comparison = await screen.findByRole("region", { name: "Comparison" });
    expect(within(comparison).getByText("+0.124 [+0.113, +0.136]")).toBeTruthy();
    expect(within(comparison).getByText("1 variant is compared with one baseline.")).toBeTruthy();
    expect(within(comparison).getByText(/Seed noise on this preset is about 0\.014 bpb/)).toBeTruthy();
    expect(within(comparison).getByRole("img", { name: /Δ val_bpb vs small, 95% CI/ })).toBeTruthy();
    const predictions = screen.getByRole("region", { name: "Predictions" });
    expect(within(predictions).getByText("1.090")).toBeTruthy();
    expect(within(predictions).getByText("−7.4%")).toBeTruthy();
    expect(within(predictions).getByText(/committed in a99fe9a before the first run/)).toBeTruthy();
    expect(await screen.findByText("Study: toy")).toBeTruthy();
    const link = screen.getAllByRole("link", { name: /bundle/i }).at(-1) as HTMLAnchorElement;
    expect(link.getAttribute("href")).toBe("/api/studies/toy/bundle.zip");
    expect(screen.queryByRole("button", { name: "Stop study" })).toBeNull();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("stops a running study", async () => {
    const seen = open([run("small", 0, "running", null, 40)], {
      "GET /api/studies/toy/report": { status: 422, problem: true, body: { title: "x", detail: "y" } },
      "POST /api/studies/toy/stop": { body: { study: "toy", stopping: [] } },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Stop study" }));
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path === "/api/studies/toy/stop")).toBe(true));
  });

  it("offers the ablation card for a finished record study only", async () => {
    open(allRuns("done"), { "GET /api/studies/toy/card/upload": { body: { repo: "a/b", repo_type: "dataset", path_in_repo: "cards/toy.json", content: "{}", commit_message: "m", exported_at: "t" } } });
    await userEvent.click(await screen.findByRole("button", { name: "Export ablation card…" }));
    expect(await screen.findByText("Ablation card for toy")).toBeTruthy();
  });

  it("has no card button while runs are still going", async () => {
    open([run("small", 0, "running", null, 5)], { "GET /api/studies/toy/report": { status: 422, problem: true, body: { title: "x", detail: "y" } } });
    await screen.findByRole("button", { name: "Stop study" });
    expect(screen.queryByRole("button", { name: "Export ablation card…" })).toBeNull();
  });
});
