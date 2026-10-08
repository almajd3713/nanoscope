import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import graph from "../../test/fixtures/graph-modern_like.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { DRAG_TYPE } from "../graph/dnd";
import { Model } from "./Model";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

const FILE = "models/my_lm.py";
const SOURCE = "class MyModern(Decoder):\n    pass\n";
const doc = { ...graph, path: FILE, etag: "e1" };

beforeEach(() => {
  localStorage.clear();
  resetLevelCache();
  installEventSource();
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
});
afterEach(() => vi.unstubAllGlobals());

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    [`GET /api/files/${FILE}`]: { body: { path: FILE, content: SOURCE, etag: "e1" } },
    [`POST /api/files/${FILE}/graph`]: { body: doc },
    [`POST /api/files/${FILE}/lint`]: { body: [] },
    "GET /api/blocks": { body: blocks },
    "GET /api/presets": { body: [{ name: "tinystories-5min" }] },
    [`POST /api/models/${FILE}:MyModern/describe`]: { status: 202, body: { id: 3, state: "queued" } },
    "GET /api/jobs/3": { body: { id: 3, state: "done" } },
    ...routes,
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
  return seen;
}

const drag = (name: string) => ({ types: [DRAG_TYPE], getData: (k: string) => (k === DRAG_TYPE ? name : "") });

describe("Model page", () => {
  it("at Learn shows the palette and the graph but not the code", async () => {
    open();
    expect(await screen.findByLabelText("Attention in attn")).toBeTruthy();
    expect(screen.getByRole("navigation", { name: "Block palette" })).toBeTruthy();
    expect(screen.queryByLabelText(`Editor for ${FILE}`)).toBeNull();
  });

  it("from Tinker up shows the editor beside the graph, on the same file", async () => {
    act(() => setLevel("Tinker"));
    open();
    expect(await screen.findByLabelText("Attention in attn")).toBeTruthy();
    expect(await screen.findByLabelText(`Editor for ${FILE}`)).toBeTruthy();
    expect(((await screen.findByLabelText(`Source of ${FILE}`)) as HTMLTextAreaElement).value).toBe(SOURCE);
  });

  it("dropping a palette block on a slot patches the file with the ETag it read, and redraws", async () => {
    const swapped = structuredClone(doc);
    ((swapped.classes[0]!.args as Record<string, unknown>)["block"] as { args: Record<string, unknown> }).args["norm"] = { kind: "block", block: "LayerNorm", args: {}, span: null, family: "norm" };
    const seen = open({ [`POST /api/files/${FILE}/graph/patch`]: { body: { ...swapped, etag: "e2" } } });
    const norm = await screen.findByLabelText("RMSNorm in norm");
    await waitFor(() => expect(screen.getByRole("heading", { name: "MyModern" })).toBeTruthy());
    fireEvent.drop(norm, { dataTransfer: drag("LayerNorm") });
    await waitFor(() => expect(seen.some((s) => s.path.endsWith("/graph/patch"))).toBe(true));
    const sent = seen.find((s) => s.path.endsWith("/graph/patch"))!;
    expect(sent.body).toEqual({ edits: [{ op: "replace_block", class: "MyModern", path: ["block", "norm"], node: { kind: "block", block: "LayerNorm", args: {}, span: null } }] });
    expect(await screen.findByLabelText("LayerNorm in norm")).toBeTruthy();
  });

  it("refuses a locked block on the page and sends nothing", async () => {
    const seen = open();
    const pos = await screen.findByLabelText("RoPE in pos");
    fireEvent.drop(pos, { dataTransfer: drag("RoPE") });
    expect((await screen.findByRole("alert")).textContent).toContain("RoPE is locked until you build it yourself");
    expect(seen.some((s) => s.path.endsWith("/graph/patch"))).toBe(false);
  });

  it("shows the server's refusal word for word when a patch is refused", async () => {
    open({
      [`POST /api/files/${FILE}/graph/patch`]: {
        status: 422,
        problem: true,
        body: { title: "Locked until you build it", status: 422, detail: "block:NoPE is locked until you build it yourself in the lesson foundations/09-nope." },
      },
    });
    const pos = await screen.findByLabelText("RoPE in pos");
    fireEvent.drop(pos, { dataTransfer: drag("NoPE") });
    expect((await screen.findByRole("alert")).textContent).toContain("block:NoPE is locked until you build it yourself in the lesson foundations/09-nope.");
  });

  it("selecting a box opens the inspector, and an edit there is one set_arg", async () => {
    const seen = open({ [`POST /api/files/${FILE}/graph/patch`]: { body: { ...doc, etag: "e2" } } });
    fireEvent.click(await screen.findByLabelText("Attention in attn"));
    const field = await screen.findByLabelText("n_heads");
    await userEvent.clear(field);
    await userEvent.type(field, "8{Enter}");
    await waitFor(() => expect(seen.some((s) => s.path.endsWith("/graph/patch"))).toBe(true));
    expect(seen.find((s) => s.path.endsWith("/graph/patch"))!.body).toEqual({
      edits: [{ op: "set_arg", class: "MyModern", path: ["block", "attn"], arg: "n_heads", value: { kind: "literal", value: 8, span: null } }],
    });
  });

  it("selecting a box jumps the editor to its line", async () => {
    act(() => setLevel("Tinker"));
    open();
    fireEvent.click(await screen.findByLabelText("Attention in attn"));
    expect((await screen.findByTestId("revealed")).textContent).toBe("9");
  });

  it("starts with nothing to undo", async () => {
    open();
    await screen.findByLabelText("Attention in attn");
    expect((screen.getByRole("button", { name: "Undo" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Redo" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("says so when the file has no model", async () => {
    open({ [`POST /api/files/${FILE}/graph`]: { body: { ...doc, classes: [] } } });
    expect(await screen.findByText("No model in this file")).toBeTruthy();
  });

  it("shows the library's message when the file cannot be read", async () => {
    open({ [`POST /api/files/${FILE}/graph`]: { status: 404, problem: true, body: { title: "Not found", status: 404, detail: `no such file: ${FILE}` } } });
    expect(await screen.findByText(`no such file: ${FILE}`)).toBeTruthy();
  });
});
