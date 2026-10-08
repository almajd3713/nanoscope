import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import { DRAG_TYPE, Palette, type PaletteBlock } from "./Palette";

const catalog = blocks.blocks as unknown as PaletteBlock[];
const mine: PaletteBlock = {
  name: "ScaledMLP", family: "mlp", args: [], user: true, certified: false, certification: { state: "stale" }, lock: { locked: false, lesson: null },
};

function show(extra: PaletteBlock[] = []) {
  render(
    <MemoryRouter>
      <Palette blocks={[...catalog, ...extra]} />
    </MemoryRouter>,
  );
}

describe("Palette", () => {
  it("groups blocks by family and leaves out what nanoscope adds itself", () => {
    show();
    const groups = screen.getAllByRole("region").map((g) => g.getAttribute("aria-label"));
    expect(groups).toEqual(["attention", "embedding", "mlp", "norm", "positional", "primitive", "structure", "template", "Your blocks"]);
    expect(screen.queryByText("Decoder")).toBeNull();
    expect(screen.queryByText("TokenEmbedding")).toBeNull();
    expect(within(screen.getByRole("region", { name: "norm" })).getAllByRole("listitem").map((li) => li.textContent)).toEqual(["LayerNorm", "RMSNorm"]);
  });

  it("an unlocked block drags, carrying its name", () => {
    show();
    const item = screen.getByText("RMSNorm").closest("li")!;
    expect(item.getAttribute("draggable")).toBe("true");
    const data: Record<string, string> = {};
    fireEvent.dragStart(item, { dataTransfer: { setData: (k: string, v: string) => (data[k] = v), effectAllowed: "" } });
    expect(data[DRAG_TYPE]).toBe("RMSNorm");
  });

  it("a locked block is greyed, cannot be dragged, and links to its lesson", () => {
    show();
    const item = screen.getByText("RoPE").closest("li")!;
    expect(item.getAttribute("draggable")).toBe("false");
    expect(item.getAttribute("aria-disabled")).toBe("true");
    expect(within(item).getByLabelText("locked")).toBeTruthy();
    expect(within(item).getByRole("link", { name: "modern-block/02-rope" }).getAttribute("href")).toBe("/learn/modern-block/02-rope");
    let started = false;
    const prevented = !fireEvent.dragStart(item, { dataTransfer: { setData: () => (started = true), effectAllowed: "" } });
    expect(prevented).toBe(true);
    expect(started).toBe(false);
  });

  it("lists your own blocks last with their certification", () => {
    show([mine, { ...mine, name: "Sealed", certified: true, certification: { state: "certified" } }]);
    const own = screen.getByRole("region", { name: "Your blocks" });
    expect(within(own).getByText("ScaledMLP")).toBeTruthy();
    expect(within(own).getByText("changed since it was certified")).toBeTruthy();
    expect(within(within(own).getByText("Sealed").closest("li")!).getByLabelText("certified")).toBeTruthy();
  });

  it("says where your own blocks come from when there are none", () => {
    show();
    expect(screen.getByText(/register_block/)).toBeTruthy();
  });

  it("filters by name", async () => {
    show();
    await userEvent.type(screen.getByLabelText("Search blocks"), "norm");
    expect(screen.getAllByRole("listitem").map((li) => li.textContent)).toEqual(["LayerNorm", "RMSNorm"]);
  });
});
