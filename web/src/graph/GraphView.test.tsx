import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import modern from "../../test/fixtures/graph-modern_like.json";
import codeOnly from "../../test/fixtures/graph-code_only.json";
import type { GClass } from "./flow";
import { GraphView } from "./GraphView";
import { layout } from "./layout";
import { buildFlow } from "./flow";

const cls = (doc: { classes: unknown[] }, i = 0) => doc.classes[i] as unknown as GClass;

describe("layout", () => {
  it("places every box, the layer boxes inside their group, with no saved positions", async () => {
    const flow = buildFlow(cls(modern));
    const placed = await layout(flow, "detailed");
    for (const b of flow.boxes) expect(placed.boxes[b.id], b.id).toBeTruthy();
    expect(placed.groups["layers"]!.height).toBeGreaterThan(placed.boxes["block/attn"]!.height);
    // data flows downward: embedding above the layers above the head
    expect(placed.boxes["tok_emb"]!.y).toBeLessThan(placed.groups["layers"]!.y);
    expect(placed.groups["layers"]!.y).toBeLessThan(placed.boxes["head"]!.y);
    // inside the group, the norm comes before attention
    expect(placed.boxes["block/norm"]!.y).toBeLessThan(placed.boxes["block/attn"]!.y);
  });

  it("is the same twice: nothing is remembered", async () => {
    const flow = buildFlow(cls(modern));
    expect(await layout(flow, "detailed")).toEqual(await layout(flow, "detailed"));
  });

  it("makes boxes taller as the depth goes up", async () => {
    const flow = buildFlow(cls(modern));
    const surface = await layout(flow, "surface");
    const detailed = await layout(flow, "detailed");
    expect(detailed.boxes["block/attn"]!.height).toBeGreaterThan(surface.boxes["block/attn"]!.height);
  });
});

describe("GraphView", () => {
  it("draws the class's blocks", async () => {
    render(<div style={{ width: 800, height: 600 }}><GraphView cls={cls(modern)} /></div>);
    await waitFor(() => expect(screen.getByLabelText("Attention in attn")).toBeTruthy());
    for (const label of ["TokenEmbedding in tok_emb", "RMSNorm in norm", "RoPE in pos", "SwiGLU in mlp", "RMSNorm in final_norm", "Head in head"]) {
      expect(screen.getByLabelText(label), label).toBeTruthy();
    }
    expect(screen.getByText("4 × layer")).toBeTruthy();
    expect(screen.getByText("n_kv_heads=2")).toBeTruthy();
  });

  it("draws again when the parsed class changes", async () => {
    const changed = structuredClone(cls(modern)) as GClass;
    (changed.args!["n_layers"] as { value: number }).value = 7;
    const { rerender } = render(<div style={{ width: 800, height: 600 }}><GraphView cls={cls(modern)} /></div>);
    await screen.findByText("4 × layer");
    rerender(<div style={{ width: 800, height: 600 }}><GraphView cls={changed} /></div>);
    await screen.findByText("7 × layer");
    expect(screen.queryByText("4 × layer")).toBeNull();
  });

  it("reports the box that is clicked", async () => {
    const onSelect = vi.fn();
    render(<div style={{ width: 800, height: 600 }}><GraphView cls={cls(modern)} onSelect={onSelect} /></div>);
    const node = await screen.findByLabelText("Attention in attn");
    fireEvent.click(node); // d3-zoom's mouse handling needs a real window; the click is what we listen for
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: "Attention", path: ["block", "attn"] }));
  });

  it("says a code-only class is code only, with the library's reason", () => {
    render(<GraphView cls={cls(codeOnly, 1)} />);
    expect(screen.getByText(/is code only: __init__ has an assignment/)).toBeTruthy();
  });
});
