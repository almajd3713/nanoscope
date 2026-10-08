import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import filled from "../../test/fixtures/graph-template_fill.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import type { CatalogBlock } from "../editor/completions";
import { DRAG_TYPE } from "./dnd";
import type { GClass } from "./flow";
import { TemplateCanvas } from "./TemplateCanvas";

const cls = filled.classes[0] as unknown as GClass;
const catalog = blocks.blocks as unknown as CatalogBlock[];
const dragOf = (name: string) => ({ types: [DRAG_TYPE], getData: (k: string) => (k === DRAG_TYPE ? name : "") });

describe("TemplateCanvas", () => {
  it("shows every slot of the template, and the nested one inside its owner", () => {
    render(<TemplateCanvas cls={cls} blocks={catalog} onEdit={() => {}} />);
    const outer = screen.getByRole("region", { name: "BlockTemplate template" });
    expect(within(outer).getAllByRole("group").map((g) => g.getAttribute("aria-label")).filter((l) => /^(norm1|attn|norm2|mlp) slot$/.test(l ?? ""))).toEqual(["norm1 slot", "attn slot", "norm2 slot", "mlp slot"]);
    const inner = within(outer).getByRole("region", { name: "AttentionTemplate template" });
    expect(within(inner).getAllByRole("group").map((g) => g.getAttribute("aria-label"))).toEqual(
      ["q", "k", "v", "scores", "mask", "normalize", "mix", "out"].map((s) => `${s} slot`),
    );
  });

  it("an empty slot says so and a filled one names its block", () => {
    render(<TemplateCanvas cls={cls} blocks={catalog} onEdit={() => {}} />);
    expect(within(screen.getByRole("group", { name: "mask slot" })).getByText("drop a block here")).toBeTruthy();
    expect(within(screen.getByRole("group", { name: "scores slot" })).getByText("ScaledDotScores(…)")).toBeTruthy();
  });

  it("dropping a primitive on a slot is one fill_slot at the template's path", () => {
    const onEdit = vi.fn();
    render(<TemplateCanvas cls={cls} blocks={catalog} onEdit={onEdit} />);
    const slot = screen.getByRole("group", { name: "mask slot" });
    expect(fireEvent.dragOver(slot, { dataTransfer: dragOf("CausalMask") })).toBe(false);
    fireEvent.drop(slot, { dataTransfer: dragOf("CausalMask") });
    expect(onEdit).toHaveBeenCalledWith([{ op: "fill_slot", class: "FromPrimitives", path: ["block", "attn"], slot: "mask", node: { kind: "block", block: "CausalMask", args: {}, span: null } }]);
  });

  it("a locked block is refused with the lesson, and nothing is sent", () => {
    const onEdit = vi.fn();
    render(<TemplateCanvas cls={cls} blocks={catalog} onEdit={onEdit} />);
    fireEvent.drop(screen.getByRole("group", { name: "mlp slot" }), { dataTransfer: dragOf("RoPE") });
    expect(onEdit).not.toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).toContain("RoPE is locked until you build it yourself in the lesson modern-block/02-rope");
  });

  it("a filled slot can be emptied", async () => {
    const onEdit = vi.fn();
    render(<TemplateCanvas cls={cls} blocks={catalog} onEdit={onEdit} />);
    await userEvent.click(screen.getByRole("button", { name: "Empty the scores slot" }));
    expect(onEdit).toHaveBeenCalledWith([{ op: "fill_slot", class: "FromPrimitives", path: ["block", "attn"], slot: "scores", node: null }]);
  });

  it("says when a class has no template", () => {
    render(<TemplateCanvas cls={modern.classes[0] as unknown as GClass} blocks={catalog} onEdit={() => {}} />);
    expect(screen.getByText(/has no template to fill/)).toBeTruthy();
  });
});
