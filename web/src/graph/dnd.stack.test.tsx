import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import layerPattern from "../../test/fixtures/graph-layer_pattern.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import { DRAG_TYPE } from "./dnd";
import type { GClass } from "./flow";
import { DRAG_LAYER, StackPanel } from "./StackPanel";

const uniform = modern.classes[0] as unknown as GClass;
const patterned = layerPattern.classes[0] as unknown as GClass;
const dragOf = (type: string, value: string) => ({ types: [type], getData: (k: string) => (k === type ? value : "") });

function show(cls: GClass) {
  const onEdit = vi.fn();
  render(<StackPanel cls={cls} onEdit={onEdit} />);
  return onEdit;
}

describe("StackPanel", () => {
  it("dropping a Block on the stack adds a layer: n_layers + 1 for one block", () => {
    const onEdit = show(uniform);
    const zone = screen.getByRole("group", { name: "Add a layer" });
    expect(fireEvent.dragOver(zone, { dataTransfer: dragOf(DRAG_TYPE, "Block") })).toBe(false);
    fireEvent.drop(zone, { dataTransfer: dragOf(DRAG_TYPE, "Block") });
    expect(onEdit).toHaveBeenCalledWith([{ op: "add_layer", class: "MyModern" }]);
  });

  it("on a pattern it adds a copy of the last block, and refuses what is not a Block", () => {
    const onEdit = show(patterned);
    const zone = screen.getByRole("group", { name: "Add a layer" });
    fireEvent.drop(zone, { dataTransfer: dragOf(DRAG_TYPE, "RMSNorm") });
    expect(onEdit).not.toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).toBe("RMSNorm is not a layer: drop a Block to add a layer");
    fireEvent.drop(zone, { dataTransfer: dragOf(DRAG_TYPE, "Block") });
    expect(onEdit.mock.calls[0]![0][0]).toMatchObject({ op: "add_layer", class: "SlidingGlobal", path: ["pattern"] });
  });

  it("dragging a layer to the trash removes it, as does its button", async () => {
    const onEdit = show(patterned);
    fireEvent.drop(screen.getByRole("group", { name: "Remove a layer by dropping it here" }), { dataTransfer: dragOf(DRAG_LAYER, "1") });
    expect(onEdit).toHaveBeenLastCalledWith([{ op: "remove_layer", class: "SlidingGlobal", path: ["pattern"], index: 1 }]);
    await userEvent.click(screen.getByRole("button", { name: "Remove layer 0" }));
    expect(onEdit).toHaveBeenLastCalledWith([{ op: "remove_layer", class: "SlidingGlobal", path: ["pattern"], index: 0 }]);
  });

  it("one block's layers are removed from n_layers", async () => {
    const onEdit = show(uniform);
    await userEvent.click(screen.getByRole("button", { name: "Remove a layer" }));
    expect(onEdit).toHaveBeenCalledWith([{ op: "remove_layer", class: "MyModern" }]);
  });

  it("the pattern editor previews what it writes and sends one set_pattern", async () => {
    const onEdit = show(uniform);
    const preview = screen.getByLabelText("This writes");
    expect(Array.from(preview.children).map((c) => c.textContent)).toEqual(["sliding", "sliding", "sliding", "global"]);
    await userEvent.click(screen.getByRole("button", { name: "Write pattern" }));
    const [edit] = onEdit.mock.calls[0]![0];
    expect(edit).toMatchObject({ op: "set_pattern", class: "MyModern" });
    expect(edit.items).toHaveLength(4);
    expect(edit.items[0].args.attn.args.window.value).toBe(64);
    expect(edit.items[3].args.attn.args.window).toBeUndefined();
  });

  it("a pattern that is not one cannot be written, and says why", async () => {
    const onEdit = show(uniform);
    const field = screen.getByLabelText("Pattern");
    await userEvent.clear(field);
    await userEvent.type(field, "fast:global 3:1");
    expect(screen.getByText("fast is not a layer kind here: use sliding or global")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Write pattern" }) as HTMLButtonElement).disabled).toBe(true);
    expect(onEdit).not.toHaveBeenCalled();
  });
});
