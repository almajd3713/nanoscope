import { describe, expect, it } from "vitest";
import layerPattern from "../../test/fixtures/graph-layer_pattern.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import opaque from "../../test/fixtures/graph-opaque_block.json";
import oneHead from "../../test/fixtures/graph-filled_template.json";
import codeOnly from "../../test/fixtures/graph-code_only.json";
import { buildFlow, type GClass } from "./flow";

const cls = (doc: { classes: unknown[] }, i = 0) => doc.classes[i] as unknown as GClass;

describe("buildFlow", () => {
  it("lays a decoder out as embedding, a layer group, final norm and head", () => {
    const flow = buildFlow(cls(modern));
    expect(flow.boxes.map((b) => b.name)).toEqual(["TokenEmbedding", "RMSNorm", "Attention", "RoPE", "SwiGLU", "RMSNorm", "Head"]);
    expect(flow.groups).toEqual([{ id: "layers", label: "4 × layer", path: ["block"] }]);
    const inLayer = flow.boxes.filter((b) => b.parent === "layers").map((b) => b.slot);
    expect(inLayer).toEqual(["norm", "attn", "pos", "mlp"]); // the layer's own order, pos hanging off attn
  });

  it("remembers each box's path, which is how an edit names it", () => {
    const flow = buildFlow(cls(modern));
    const byName = (n: string) => flow.boxes.find((b) => b.name === n)!;
    expect(byName("Attention").path).toEqual(["block", "attn"]);
    expect(byName("RoPE").path).toEqual(["block", "attn", "pos"]);
    expect(flow.boxes.filter((b) => b.name === "RMSNorm").map((b) => b.path)).toEqual([["block", "norm"], ["final_norm"]]);
    expect(byName("Head").path).toBeNull();
  });

  it("shows plain arguments as written and hangs nested blocks off their owner", () => {
    const flow = buildFlow(cls(modern));
    const attn = flow.boxes.find((b) => b.name === "Attention")!;
    expect(attn.args).toEqual([["n_heads", "4"], ["n_kv_heads", "2"], ["qk_norm", "True"]]);
    expect(flow.arrows.find((a) => a.kind === "arg")).toMatchObject({ from: "block/attn", to: "block/attn/pos" });
    // data flows norm -> attn -> mlp inside the layer, and the layer sits between embedding and final norm
    const flows = flow.arrows.filter((a) => a.kind === "flow").map((a) => `${a.from}>${a.to}`);
    expect(flows).toContain("block/norm>block/attn");
    expect(flows).toContain("block/attn>block/mlp");
    expect(flows).toContain("tok_emb>block/norm");
    expect(flows).toContain("block/mlp>final_norm");
    expect(flows).toContain("final_norm>head");
  });

  it("follows a layer pattern item by item", () => {
    const flow = buildFlow(cls(layerPattern));
    expect(flow.groups[0]!.label).toBe("pattern of 2, repeated through 6 layers");
    const paths = flow.boxes.filter((b) => b.parent === "layers").map((b) => b.id);
    expect(paths.filter((p) => p.startsWith("pattern/0/")).length).toBeGreaterThan(0);
    expect(paths.filter((p) => p.startsWith("pattern/1/")).length).toBeGreaterThan(0);
    expect(flow.arrows.some((a) => a.from.startsWith("pattern/0/") && a.to.startsWith("pattern/1/"))).toBe(true);
  });

  it("keeps calls it cannot edit as opaque boxes", () => {
    const flow = buildFlow(cls(opaque));
    const odd = flow.boxes.filter((b) => b.kind === "opaque");
    expect(odd.map((b) => [b.slot, b.name])).toEqual([["attn", "Gated"], ["mlp", "mylib.FancyMLP"]]);
    expect(odd[0]!.line).toBe(11);
  });

  it("draws nothing for a class that is code only", () => {
    const flow = buildFlow(cls(codeOnly, 1));
    expect(flow).toEqual({ boxes: [], groups: [], arrows: [] });
  });

  it("draws a class that fills a template as its filled slots in order, empty slots left out", () => {
    const flow = buildFlow(cls(oneHead));
    expect(flow.boxes.map((b) => [b.slot, b.name])).toEqual([
      ["q", "Linear"], ["k", "Linear"], ["v", "Linear"], ["scores", "ScaledDotScores"], ["normalize", "Softmax"], ["out", "Linear"],
    ]);
    expect(flow.boxes[0]!.path).toEqual(["q"]);
    expect(flow.arrows.map((a) => a.id)).toEqual(["q>k", "k>v", "v>scores", "scores>normalize", "normalize>out"]);
  });
});
