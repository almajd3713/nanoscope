import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import graph from "../../test/fixtures/graph-modern_like.json";
import lesson04 from "../../test/fixtures/lesson-04.json";
import starterStructure from "../../test/fixtures/structure-04-starter.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { Model } from "./Model";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

const FILE = "lessons/foundations/04-multi-head/starter.py";
const SOURCE = "import torch\n\n\nclass MultiHead(nn.Module):\n    pass\n";
const REASON = "output differs from the reference: max abs diff 0.31 on inputs of shape [[2, 6, 16]] (tolerance 1e-05)";

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    [`GET /api/files/${FILE}`]: { body: { path: FILE, content: SOURCE, etag: "e1" } },
    [`POST /api/files/${FILE}/graph`]: { body: { ...graph, path: FILE, etag: "e1" } },
    [`POST /api/files/${FILE}/lint`]: { body: [] },
    "GET /api/blocks": { body: blocks },
    "GET /api/presets": { body: [] },
    "GET /api/curricula/foundations/04-multi-head": { body: { ...lesson04, state: "started" } },
    "GET /api/jobs": { body: [] },
    [`POST /api/models/${FILE}:MyModern/describe`]: { status: 202, body: { id: 3, state: "queued" } },
    "GET /api/jobs/3": { body: { id: 3, state: "done" } },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={[`/model/${FILE}`]}>
        <Routes>
          <Route path="/model/*" element={<Model />} />
          <Route path="/learn/:path/:lesson" element={<p>lesson page</p>} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

describe("Model page for a lesson file", () => {
  it("shows the lesson, a way back to it, and the check button", async () => {
    open();
    const bar = await screen.findByRole("region", { name: "Lesson" });
    expect(within(bar).getByRole("link", { name: lesson04.title }).getAttribute("href")).toBe("/learn/foundations/04-multi-head");
    expect(within(bar).getByRole("button", { name: "Run the check" })).toBeTruthy();
  });

  it("the lesson's file is editable at the Learn level", async () => {
    open();
    const box = await screen.findByLabelText(`Source of ${FILE}`);
    expect((box as HTMLTextAreaElement).readOnly).toBe(false);
  });

  it("a file outside the lessons has no lesson bar", async () => {
    mockApi({
      "GET /api/files/m.py": { body: { path: "m.py", content: SOURCE, etag: "e1" } },
      "POST /api/files/m.py/graph": { body: { ...graph, path: "m.py", etag: "e1" } },
      "POST /api/files/m.py/lint": { body: [] },
      "GET /api/blocks": { body: blocks },
      "GET /api/presets": { body: [] },
      "POST /api/models/m.py:MyModern/describe": { status: 202, body: { id: 3, state: "queued" } },
      "GET /api/jobs/3": { body: { id: 3, state: "done" } },
    });
    render(
      <Providers>
        <MemoryRouter initialEntries={["/model/m.py"]}>
          <Routes>
            <Route path="/model/*" element={<Model />} />
          </Routes>
        </MemoryRouter>
      </Providers>,
    );
    await screen.findByRole("heading", { name: "MyModern" });
    expect(screen.queryByRole("region", { name: "Lesson" })).toBeNull();
  });

  it("runs the check from here and puts a failed equivalence result on its class", async () => {
    act(() => setLevel("Tinker"));
    let queued = false;
    const seen = open({
      "POST /api/curricula/foundations/04-multi-head/check": {
        status: 202,
        body: () => {
          queued = true;
          return { id: 9, kind: "check", state: "queued", payload: { lesson: "foundations/04-multi-head" } };
        },
      },
      "GET /api/jobs": {
        body: () =>
          queued
            ? [{
                id: 9, kind: "check", state: "done", lane: "interactive", owner: "local", payload: { lesson: "foundations/04-multi-head" }, error: null, created_at: 1,
                result: { passed: false, check_id: "c", lesson: "foundations/04-multi-head", checks: [
                  { id: "built", passed: true, reason: "MultiHead builds" },
                  { id: "same-as-reference", passed: false, reason: REASON },
                ] },
              }]
            : [],
      },
    });
    await userEvent.click(await screen.findByRole("button", { name: "Run the check" }));
    await waitFor(() => expect(seen.some((s) => s.path.endsWith("/check") && s.method === "POST")).toBe(true));
    // the reasons are shown word for word, and the marker sits on `class MultiHead` (line 4)
    expect(await screen.findByText(REASON, { selector: "span" })).toBeTruthy();
    expect(await screen.findByText(`nanoscope 4:1 ${REASON}`)).toBeTruthy();
  });

  it("a lesson file that is plain PyTorch shows what its code builds, with the lesson's actions and an editable file", async () => {
    const seen = open({
      [`POST /api/files/${FILE}/graph`]: { body: { ...graph, path: FILE, etag: "e1", classes: [] } },
      "GET /api/models": { body: [{ name: "MultiHead", ref: `${FILE}:MultiHead`, source: FILE, graph: false }] },
      [`POST /api/models/${FILE}:MultiHead/structure`]: { status: 202, body: { id: 9, state: "queued" } },
      "GET /api/jobs/9": { body: { id: 9, state: "done", result: starterStructure } },
    });
    const view = await screen.findByRole("region", { name: "Structure" });
    // the starter as shipped: built with the lesson's sizes, nothing in it yet, forward not written
    expect(await within(view).findByText("MultiHead has no layers yet.", { exact: false })).toBeTruthy();
    expect(within(view).getByText("MultiHead(d_model=16, context_length=8, n_heads=4)")).toBeTruthy();
    expect(within(view).getByText("NotImplementedError: multi-head attention", { exact: false })).toBeTruthy();
    expect(within(view).getByRole("button", { name: "line 30" })).toBeTruthy();
    // the lesson's own check says how to build the class: its first inputs go with the job
    const sent = seen.find((x) => x.path === `/api/models/${FILE}:MultiHead/structure`);
    expect(sent?.body).toEqual({ args: {}, inputs: lesson04.checks[1]!.inputs });
    expect(within(view).queryByRole("button", { name: "Start a model from blocks" })).toBeNull();
    expect(within(view).getByText("This lesson is written in plain PyTorch", { exact: false })).toBeTruthy();
    const bar = await screen.findByRole("region", { name: "Lesson" });
    expect(within(bar).getByRole("button", { name: "Run the check" })).toBeTruthy();
    expect(((await screen.findByLabelText(`Source of ${FILE}`)) as HTMLTextAreaElement).value).toBe(SOURCE);
  });

  it("a plain lesson file that has a blocks way in links to it while the file is as shipped", async () => {
    open({
      [`POST /api/files/${FILE}/graph`]: { body: { ...graph, path: FILE, etag: "e1", classes: [] } },
      "GET /api/curricula/foundations/04-multi-head": { body: { ...lesson04, state: "started", template: { file: "template.py", summary: "Slots." }, starter: "code" } },
      "GET /api/models": { body: [{ name: "MultiHead", ref: `${FILE}:MultiHead`, source: FILE, graph: false }] },
      [`POST /api/models/${FILE}:MultiHead/structure`]: { status: 202, body: { id: 9, state: "queued" } },
      "GET /api/jobs/9": { body: { id: 9, state: "done", result: starterStructure } },
    });
    const link = await screen.findByRole("link", { name: "Build it from blocks on the lesson page" });
    expect(link.getAttribute("href")).toBe("/learn/foundations/04-multi-head");
  });
});
