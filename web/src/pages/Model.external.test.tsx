import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import graph from "../../test/fixtures/graph-modern_like.json";
import { installEventSource, MockEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache } from "../app/level";
import { Providers } from "../app/providers";
import { Model } from "./Model";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

const FILE = "models/my_lm.py";

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

describe("Model page and the workspace watcher", () => {
  it("draws the graph again when the file is changed by something else", async () => {
    let layers = 4;
    mockApi({
      [`GET /api/files/${FILE}`]: { body: () => ({ path: FILE, content: "x", etag: `e${layers}` }) },
      [`POST /api/files/${FILE}/graph`]: {
        body: () => {
          const doc = structuredClone(graph) as unknown as { classes: { args: Record<string, { value?: number }> }[] };
          doc.classes[0]!.args["n_layers"]!.value = layers;
          return { ...doc, path: FILE, etag: `e${layers}` };
        },
      },
      [`POST /api/files/${FILE}/lint`]: { body: [] },
      "GET /api/blocks": { body: blocks },
      "GET /api/presets": { body: [] },
      [`POST /api/models/${FILE}:MyModern/describe`]: { status: 202, body: { id: 3, state: "queued" } },
      "GET /api/jobs/3": { body: { id: 3, state: "done" } },
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
    await screen.findByText("4 × layer");
    layers = 6;
    for (const source of MockEventSource.all.filter((s) => s.url === "/api/files/events")) {
      act(() => source.emit("change", { path: "other.py", kind: "modified", etag: "z" })); // not ours: ignored
      act(() => source.emit("change", { path: FILE, kind: "modified", etag: "e6" }));
    }
    await waitFor(() => expect(screen.getByText("6 × layer")).toBeTruthy());
    expect(screen.queryByText("4 × layer")).toBeNull();
  });
});
