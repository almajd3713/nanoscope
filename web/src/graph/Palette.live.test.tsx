import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import { installEventSource, MockEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { useBlocks } from "./api";
import { Palette } from "./Palette";

function LivePalette() {
  const found = useBlocks();
  return <Palette blocks={found.data?.blocks ?? []} />;
}

beforeEach(installEventSource);
afterEach(() => vi.unstubAllGlobals());

const stream = (path: string) => MockEventSource.all.filter((s) => s.url === path).at(-1)!;

describe("Palette stays current", () => {
  it("unlocks a block when a lesson check earns it, without a reload", async () => {
    let earned = false;
    const body = () => ({
      ...blocks,
      blocks: blocks.blocks.map((b) => (earned && b.name === "RoPE" ? { ...b, lock: { ...b.lock, locked: false, how: "earned" } } : b)),
    });
    mockApi({ "GET /api/blocks": { body } });
    render(
      <Providers>
        <MemoryRouter>
          <LivePalette />
        </MemoryRouter>
      </Providers>,
    );
    const rope = async () => (await screen.findByText("RoPE")).closest("li")!;
    expect((await rope()).getAttribute("draggable")).toBe("false");
    earned = true;
    act(() => stream("/api/learn/events").emit("unlocks", { unlocks: { "block:RoPE": {} } }));
    await waitFor(async () => expect((await rope()).getAttribute("draggable")).toBe("true"));
  });

  it("lists a block registered in a workspace file as soon as the file changes", async () => {
    let extra = false;
    const mine = { name: "ScaledMLP", family: "mlp", args: [], user: true, certified: false, certification: { state: "uncertified" }, lock: { lockable: false, locked: false, lesson: null, how: null } };
    mockApi({ "GET /api/blocks": { body: () => ({ ...blocks, blocks: extra ? [...blocks.blocks, mine] : blocks.blocks }) } });
    render(
      <Providers>
        <MemoryRouter>
          <LivePalette />
        </MemoryRouter>
      </Providers>,
    );
    await screen.findByText("RoPE");
    expect(screen.queryByText("ScaledMLP")).toBeNull();
    extra = true;
    act(() => stream("/api/files/events").emit("change", { path: "notes.md", kind: "modified", etag: "x" }));
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByText("ScaledMLP")).toBeNull(); // not a python file: nothing to re-read
    act(() => stream("/api/files/events").emit("change", { path: "blocks/mine.py", kind: "added", etag: "y" }));
    expect(await screen.findByText("ScaledMLP")).toBeTruthy();
    expect(screen.getByText("not certified")).toBeTruthy();
  });
});
