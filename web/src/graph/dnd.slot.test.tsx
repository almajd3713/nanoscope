import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import opaque from "../../test/fixtures/graph-opaque_block.json";
import type { CatalogBlock } from "../editor/completions";
import { carriesBlock, DRAG_TYPE, planDrop } from "./dnd";
import { buildFlow, type GClass } from "./flow";
import { GraphView } from "./GraphView";

const cls = modern.classes[0] as unknown as GClass;
const catalog = blocks.blocks as unknown as CatalogBlock[];
const box = (name: string, doc = cls) => buildFlow(doc).boxes.find((b) => b.name === name)!;

describe("planDrop", () => {
  it("a block of the same family becomes one replace_block at the slot's path", () => {
    const plan = planDrop("MyModern", box("RMSNorm"), "LayerNorm", catalog);
    expect(plan).toEqual({ ok: true, edit: { op: "replace_block", class: "MyModern", path: ["block", "norm"], node: { kind: "block", block: "LayerNorm", args: {}, span: null } } });
  });

  it("refuses a locked block before sending anything, naming the lesson", () => {
    const plan = planDrop("MyModern", box("NoPE") ?? box("RoPE"), "RoPE", catalog);
    expect(plan.ok).toBe(false);
    expect(plan.ok === false && plan.message).toBe("RoPE is locked until you build it yourself in the lesson modern-block/02-rope");
  });

  it("refuses a block that does not fit the slot", () => {
    const plan = planDrop("MyModern", box("RMSNorm"), "GELUMLP", catalog);
    expect(plan).toEqual({ ok: false, message: "GELUMLP is a mlp block; the norm takes a norm block" });
  });

  it("refuses the box that is already there, nanoscope's own boxes and calls it cannot edit", () => {
    expect(planDrop("MyModern", box("RMSNorm"), "RMSNorm", catalog)).toEqual({ ok: false, message: "RMSNorm is already there" });
    expect(planDrop("MyModern", box("Head"), "Linear", catalog).ok).toBe(false);
    const odd = box("Gated", opaque.classes[0] as unknown as GClass);
    expect(planDrop("WithCustom", odd, "SwiGLU", catalog)).toEqual({ ok: false, message: "Gated is a call the graph cannot edit; change it in the code" });
  });
});

describe("dropping on the graph", () => {
  it("a palette drag over a slot is accepted, and the drop reports the slot and the block", async () => {
    const onDropBlock = vi.fn();
    render(<div style={{ width: 800, height: 600 }}><GraphView cls={cls} onDropBlock={onDropBlock} /></div>);
    const norm = await screen.findByLabelText("RMSNorm in norm");
    const data = { [DRAG_TYPE]: "LayerNorm" };
    const dataTransfer = { types: [DRAG_TYPE], getData: (k: string) => data[k as keyof typeof data] ?? "", dropEffect: "" };
    expect(carriesBlock({ dataTransfer: dataTransfer as unknown as DataTransfer })).toBe(true);
    expect(fireEvent.dragOver(norm, { dataTransfer })).toBe(false); // preventDefault: it is a drop target
    fireEvent.drop(norm, { dataTransfer });
    await waitFor(() => expect(onDropBlock).toHaveBeenCalledTimes(1));
    expect(onDropBlock.mock.calls[0]![0]).toMatchObject({ name: "RMSNorm", path: ["block", "norm"] });
    expect(onDropBlock.mock.calls[0]![1]).toBe("LayerNorm");
  });

  it("ignores a drag that is not a palette block, and nanoscope's own boxes", async () => {
    const onDropBlock = vi.fn();
    render(<div style={{ width: 800, height: 600 }}><GraphView cls={cls} onDropBlock={onDropBlock} /></div>);
    const norm = await screen.findByLabelText("RMSNorm in norm");
    const other = { types: ["text/plain"], getData: (k: string) => (k === "text/plain" ? "x" : "") };
    expect(fireEvent.dragOver(norm, { dataTransfer: other })).toBe(true); // not prevented
    fireEvent.drop(norm, { dataTransfer: other });
    const head = await screen.findByLabelText("Head in head");
    fireEvent.drop(head, { dataTransfer: { types: [DRAG_TYPE], getData: () => "Linear" } });
    expect(onDropBlock).not.toHaveBeenCalled();
  });
});
