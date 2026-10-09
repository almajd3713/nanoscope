import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import models from "../../test/fixtures/models.json";
import presets from "../../test/fixtures/presets.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { StudyBuilder } from "./StudyBuilder";

// A spec with every key the builder edits and some it does not.
const SPEC = {
  schema: 1,
  name: "m1-ablation",
  preset: "tinystories-5min",
  overrides: { max_steps: 100 },
  budget: { tokens: 4000000 },
  match: "params",
  match_knob: "ffn_hidden",
  match_to: "modern",
  range: [64, 1024, 8],
  baseline: "modern",
  seeds: [0, 1, 2],
  mode: "record",
  tolerance: 0.02,
  variants: [
    { name: "modern", model: "nanoscope.models.modern:Modern", kwargs: { ffn_hidden: 384 } },
    { name: "no-rope", model: "nanoscope.models.modern:Modern", kwargs: { rope: false } },
  ],
  predictions: { modern: { val_bpb: 1.06 } },
};
const TOML = 'schema = 1\nname = "m1-ablation"\n';
const job = (id: number, state: string, result: unknown = null) => ({ id, kind: "x", state, lane: "interactive", owner: "local", payload: {}, result, error: null, created_at: 1 });
const hardware = { devices: [{ name: "cpu", kind: "cpu", memory_total: null, memory_free: null }], cpu_count: 4, torch: "2", workers: [] };
const ok = { ok: true, problems: [], refs: [], estimate: null, toml: TOML };

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

function open(at: string, routes: Parameters<typeof mockApi>[0]) {
  const seen = mockApi({
    "GET /api/models": { body: models },
    "GET /api/presets": { body: presets },
    "GET /api/hardware": { body: hardware },
    "POST /api/validate/study": { body: ok },
    "POST /api/studies/sizes": { status: 202, body: job(9, "queued") },
    "GET /api/jobs/9": { body: job(9, "failed") },
    "POST /api/studies": { status: 201, body: { name: "m1-ablation" } },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[at]}>
        <Routes>
          <Route path="/studies/new" element={<StudyBuilder />} />
          <Route path="/studies/:name/edit" element={<StudyBuilder />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

const posted = (seen: ReturnType<typeof mockApi>) =>
  seen.find((s) => s.method === "POST" && s.path === "/api/studies")!.body as { spec: unknown; overwrite: boolean };

describe("StudyBuilder round trip", () => {
  it("saves a TOML spec back exactly as it was read, keeping what it has no field for", async () => {
    const seen = open("/studies/m1-ablation/edit", {
      "GET /api/studies/m1-ablation/spec": { body: { name: "m1-ablation", path: "studies/m1-ablation.toml", spec: SPEC, toml: TOML } },
    });
    await screen.findByDisplayValue("no-rope");
    await waitFor(() => expect(screen.getByText(/saved/)).toBeTruthy());
    await userEvent.click(screen.getByRole("button", { name: "Save as TOML" }));
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path === "/api/studies")).toBe(true));
    expect(posted(seen).spec).toEqual(SPEC);
  });

  it("opens a Python study through a worker and writes the same spec the CLI prints", async () => {
    const seen = open("/studies/new?file=studies/m1.py", {
      "POST /api/studies/from-file": { status: 202, body: job(4, "queued") },
      "GET /api/jobs/4": { body: job(4, "done", { spec: SPEC, toml: TOML }) },
    });
    await screen.findByDisplayValue("no-rope");
    expect(seen.find((s) => s.path === "/api/studies/from-file")!.body).toEqual({ file: "studies/m1.py" });
    expect((screen.getByLabelText("Name") as HTMLInputElement).value).toBe("m1-ablation");
    await userEvent.click(screen.getByRole("button", { name: "Save as TOML" }));
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path === "/api/studies")).toBe(true));
    expect(posted(seen).spec).toEqual(SPEC);
    expect(posted(seen).overwrite).toBe(false);
  });
});

describe("StudyBuilder open a Python study", () => {
  it("asks for the file and reads it", async () => {
    open("/studies/new", {
      "POST /api/studies/from-file": { status: 202, body: job(4, "queued") },
      "GET /api/jobs/4": { body: job(4, "running") },
    });
    await userEvent.type(await screen.findByLabelText("Start from a Python study"), "studies/m1.py");
    await userEvent.click(screen.getByRole("button", { name: "Open" }));
    expect(await screen.findByText("Reading studies/m1.py…")).toBeTruthy();
  });
});
