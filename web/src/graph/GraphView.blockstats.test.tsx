import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import { installEventSource, MockEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { Model } from "../pages/Model";
import type { GClass } from "./flow";
import { GraphView, layerChips } from "./GraphView";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

const cls = modern.classes[0] as unknown as GClass;
const frame = (el: React.ReactNode) => <div style={{ width: 800, height: 600 }}>{el}</div>;

describe("layer colours from block statistics", () => {
  it("ranks each layer's gradient norm from 1 (lowest) to 5 (highest)", () => {
    const chips = layerChips([
      { name: "blocks.0", grad_norm: 0.724 },
      { name: "blocks.1", grad_norm: 0.13 },
      { name: "blocks.2", grad_norm: 0.114 },
      { name: "blocks.3", grad_norm: 0.089 },
    ]);
    expect(chips.map((c) => c.heat)).toEqual([5, 1, 1, 1]);
    expect(chips[0]).toEqual({ name: "0", heat: 5, title: "blocks.0: grad_norm 0.724 (5 of 5)" });
    expect(layerChips([{ name: "blocks.0", grad_norm: 1 }])[0]!.heat).toBe(1);
  });

  it("shows a chip per layer in the layer group", async () => {
    render(frame(<GraphView cls={cls} layerStats={[{ name: "blocks.0", grad_norm: 0.7 }, { name: "blocks.1", grad_norm: 0.1 }]} />));
    await waitFor(() => expect(screen.getByLabelText("blocks.0: grad_norm 0.7 (5 of 5)")).toBeTruthy());
    expect(screen.getByLabelText("blocks.1: grad_norm 0.1 (1 of 5)")).toBeTruthy();
  });

  it("the layer group steps n_layers with + and −", async () => {
    const onAddLayer = vi.fn();
    const onRemoveLayer = vi.fn();
    render(frame(<GraphView cls={cls} onAddLayer={onAddLayer} onRemoveLayer={onRemoveLayer} />));
    await screen.findByLabelText("Attention in attn");
    fireEvent.click(screen.getByLabelText("Add a layer"));
    fireEvent.click(screen.getByLabelText("Remove a layer"));
    expect(onAddLayer).toHaveBeenCalledOnce();
    expect(onRemoveLayer).toHaveBeenCalledOnce();
  });
});

describe("the model page follows the blockstats of a run of this model", () => {
  beforeEach(() => {
    localStorage.clear();
    resetLevelCache();
    installEventSource();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("reads the newest line, then each new one as the run writes it", async () => {
    const FILE = "models/my_lm.py";
    mockApi({
      [`GET /api/files/${FILE}`]: { body: { path: FILE, content: "x", etag: "e1" } },
      [`POST /api/files/${FILE}/graph`]: { body: { ...modern, path: FILE, etag: "e1" } },
      [`POST /api/files/${FILE}/lint`]: { body: [] },
      "GET /api/blocks": { body: blocks },
      "GET /api/presets": { body: [] },
      "GET /api/jobs": { body: [] },
      [`POST /api/models/${FILE}:MyModern/describe`]: { status: 202, body: { id: 3, state: "queued" } },
      "GET /api/jobs/3": { body: { id: 3, state: "done" } },
      "GET /api/runs": { body: [{ ref: "tinystories-5min/other/seed-0", model: "Other", state: "done" }, { ref: "tinystories-5min/mymodern/seed-0", model: "MyModern", state: "running" }] },
      "GET /api/runs/tinystories-5min/mymodern/seed-0/blockstats": { body: [{ step: 10, blocks: [{ name: "blocks.0", grad_norm: 0.95 }, { name: "blocks.1", grad_norm: 0.21 }] }] },
    });
    render(
      <Providers>
        <MemoryRouter initialEntries={[`/model/${FILE}`]}>
          <Routes>
            <Route path="/model/*" element={<Model />} />
          </Routes>
        </MemoryRouter>
      </Providers>,
    );
    await waitFor(() => expect(screen.getByLabelText("blocks.0: grad_norm 0.95 (5 of 5)")).toBeTruthy());
    expect(screen.getByText(/Layer colours: gradient norm at step 10 of/)).toBeTruthy();
    const stream = MockEventSource.all.filter((s) => s.url === "/api/runs/tinystories-5min/mymodern/seed-0/events").at(-1)!;
    act(() => stream.emit("blockstats", { step: 20, blocks: [{ name: "blocks.0", grad_norm: 0.1 }, { name: "blocks.1", grad_norm: 0.9 }] }));
    await waitFor(() => expect(screen.getByLabelText("blocks.1: grad_norm 0.9 (5 of 5)")).toBeTruthy());
    expect(screen.getByLabelText("blocks.0: grad_norm 0.1 (1 of 5)")).toBeTruthy();
    expect(screen.getByText(/at step 20 of/)).toBeTruthy();
  });
});
