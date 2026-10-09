import { render, screen, waitFor, within } from "@testing-library/react";
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

const SPEC = {
  schema: 1,
  name: "toy",
  preset: "tinystories-5min",
  match: "params",
  baseline: "small",
  seeds: [0, 1],
  mode: "explore",
  tolerance: 0.02,
  variants: [
    { name: "small", model: "nanoscope.models.bigram:Bigram", kwargs: { d_model: 8 } },
    { name: "wide", model: "nanoscope.models.bigram:Bigram", kwargs: { d_model: 32 } },
  ],
};
const TOML = 'schema = 1\nname = "toy"\n';
const sizes = {
  reference: "small",
  tolerance: 0.02,
  variants: [
    { name: "small", kwargs: { d_model: 8 }, non_embedding_params: 1000, flops_per_token: 6000, delta: 0, within: true },
    { name: "wide", kwargs: { d_model: 32 }, non_embedding_params: 1084, flops_per_token: 6504, delta: 0.084, within: false },
  ],
};
const job = (state: string, result: unknown = null) => ({ id: 5, kind: "sizes", state, lane: "interactive", owner: "local", payload: {}, result, error: null, created_at: 1 });
const hardware = {
  devices: [{ name: "cpu", kind: "cpu", memory_total: null, memory_free: null }, { name: "cuda:0", kind: "cuda", memory_total: 8 * 1024 ** 3, memory_free: 6 * 1024 ** 3 }],
  cpu_count: 16, torch: "2", workers: [],
};
const ok = { ok: true, problems: [], refs: [], estimate: { text: "about 2 min on cuda:0", complete: true, wall_seconds: 120, total_seconds: 240, runs: 4 }, toml: TOML };

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

function open(at: string, routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/models": { body: models },
    "GET /api/presets": { body: presets },
    "GET /api/hardware": { body: hardware },
    "GET /api/studies/toy/spec": { body: { name: "toy", path: "studies/toy.toml", spec: SPEC, toml: TOML } },
    "POST /api/validate/study": { body: (sent: unknown) => ({ ...ok, toml: JSON.stringify((sent as { spec: unknown }).spec) === JSON.stringify(SPEC) ? TOML : "changed" }) },
    "POST /api/studies/sizes": { status: 202, body: job("queued") },
    "GET /api/jobs/5": { body: job("done", sizes) },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[at]}>
        <Routes>
          <Route path="/studies/new" element={<StudyBuilder />} />
          <Route path="/studies/:name/edit" element={<StudyBuilder />} />
          <Route path="/studies" element={<p>Studies list</p>} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

describe("StudyBuilder", () => {
  it("shows a saved study's variants with sizes from a worker's job, red outside the tolerance", async () => {
    open("/studies/toy/edit");
    expect(await screen.findByDisplayValue("wide")).toBeTruthy();
    expect(await screen.findByText("1,084")).toBeTruthy();
    expect(screen.getByText("+8.4%")).toBeTruthy();
    expect(screen.getByText("wide is outside the size tolerance")).toBeTruthy();
    expect(screen.getByText(/8\.4% above small; match = params allows 2%/)).toBeTruthy();
    expect(screen.getByText("about 2 min on cuda:0")).toBeTruthy();
    expect(screen.getByText(/schema = 1/)).toBeTruthy();
  });

  it("sends the typed spec for judging, with devices and runs per device for the estimate", async () => {
    const seen = open("/studies/toy/edit");
    await screen.findByDisplayValue("wide");
    await userEvent.clear(screen.getByLabelText("Keywords of wide"));
    await userEvent.type(screen.getByLabelText("Keywords of wide"), "d_model = 16");
    await waitFor(() => {
      const sent = seen.filter((s) => s.path === "/api/validate/study").at(-1)?.body as { spec: { variants: { kwargs: unknown }[] }; devices: string[]; workers_per_device: number };
      expect(sent.spec.variants[1]!.kwargs).toEqual({ d_model: 16 });
      expect(sent.devices).toEqual(["cuda:0"]);
      expect(sent.workers_per_device).toBe(1);
    });
    await waitFor(() => expect(screen.getByText(/unsaved changes/)).toBeTruthy());
  });

  it("adds and removes variants, and keeps the baseline pointing at one that exists", async () => {
    open("/studies/toy/edit");
    await screen.findByDisplayValue("wide");
    await userEvent.click(screen.getByRole("button", { name: "Add variant" }));
    expect(screen.getByDisplayValue("variant-3")).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Remove small" }));
    expect(screen.queryByDisplayValue("small")).toBeNull();
    expect((screen.getByRole("combobox", { name: /^Baseline/ }) as HTMLSelectElement).value).toBe("");
  });

  it("saves over the existing file, and a new study as a new file", async () => {
    const seen = open("/studies/toy/edit");
    await screen.findByDisplayValue("wide");
    await userEvent.click(screen.getByRole("button", { name: "Save as TOML" }));
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path === "/api/studies")).toBe(true));
    const sent = seen.find((s) => s.method === "POST" && s.path === "/api/studies")!.body as { overwrite: boolean; spec: { name: string } };
    expect(sent.overwrite).toBe(true);
    expect(sent.spec.name).toBe("toy");
  });

  it("puts the library's words on a problem and disables Save", async () => {
    open("/studies/new", {
      "POST /api/validate/study": {
        body: { ok: false, problems: [{ code: "invalid_spec", field: "variants", message: "a study needs at least one variant", hint: null }], refs: [], estimate: null, toml: null },
      },
    });
    expect(await screen.findByText("a study needs at least one variant")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Save as TOML" }) as HTMLButtonElement).disabled).toBe(true);
    const plan = screen.getByRole("region", { name: "Plan" });
    expect((within(plan).getByLabelText("Name") as HTMLInputElement).disabled).toBe(false);
  });

  const git = (over: object = {}) => ({
    repo: true, root: "/ws", branch: "main", head: "a99fe9aa0e70b58a5d011120093e0935157b4f35", clean: true, changed: [],
    identity: true, path: "studies/toy.toml", path_committed: true, ...over,
  });
  const record = { ...SPEC, mode: "record" };
  const spec = (mode: object) => ({ "GET /api/studies/toy/spec": { body: { name: "toy", path: "studies/toy.toml", spec: mode, toml: TOML } } });

  it("trains an explore study from the saved file", async () => {
    const seen = open("/studies/toy/edit", { "GET /api/git/status": { body: git({ clean: false, changed: ["x"] }) }, "POST /api/studies/toy/run": { status: 202, body: { study: "toy", job: job("queued") } } });
    await screen.findByDisplayValue("wide");
    const button = await screen.findByRole("button", { name: "Train 4 runs" }) as HTMLButtonElement;
    await waitFor(() => expect(button.disabled).toBe(false));
    await userEvent.click(button);
    await waitFor(() => expect(seen.some((s) => s.method === "POST" && s.path === "/api/studies/toy/run")).toBe(true));
  });

  it("will not train in record mode until the spec is committed in a clean tree, and says which files", async () => {
    open("/studies/toy/edit", { ...spec(record), "POST /api/validate/study": { body: ok }, "GET /api/git/status": { body: git({ clean: false, changed: ["studies/toy.toml", "notes/ideas.md"], path_committed: false }) } });
    await screen.findByDisplayValue("wide");
    expect(await screen.findByText("2 uncommitted changes")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Train 4 runs" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Commit the preregistration to train in record mode")).toBeTruthy();
  });

  it("enables record mode for a committed spec in a clean tree", async () => {
    open("/studies/toy/edit", { ...spec(record), "POST /api/validate/study": { body: ok }, "GET /api/git/status": { body: git() } });
    await screen.findByDisplayValue("wide");
    await screen.findByText("clean");
    const button = screen.getByRole("button", { name: "Train 4 runs" }) as HTMLButtonElement;
    await waitFor(() => expect(button.disabled).toBe(false));
    expect(screen.getByText(/Every run records the preregistration commit/)).toBeTruthy();
  });
});
