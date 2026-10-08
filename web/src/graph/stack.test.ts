import { describe, expect, it } from "vitest";
import layerPattern from "../../test/fixtures/graph-layer_pattern.json";
import modern from "../../test/fixtures/graph-modern_like.json";
import type { GClass } from "./flow";
import { addLayerEdit, parsePattern, patternItems, removeLayerEdit, stackOf } from "./stack";

const uniform = modern.classes[0] as unknown as GClass;
const patterned = layerPattern.classes[0] as unknown as GClass;

describe("stackOf", () => {
  it("one block repeated, or a pattern", () => {
    expect(stackOf(uniform)).toMatchObject({ kind: "uniform", layers: 4 });
    const p = stackOf(patterned)!;
    expect(p).toMatchObject({ kind: "pattern", layers: 6 });
    expect(p.items.map((i) => i.label)).toHaveLength(2); // its first layer's window is a parameter, not a literal: global
  });
});

describe("edits", () => {
  it("a uniform stack changes n_layers; a pattern gains a copy of its last block or loses one", () => {
    expect(addLayerEdit(uniform)).toEqual({ op: "add_layer", class: "MyModern" });
    expect(removeLayerEdit(uniform, null)).toEqual({ op: "remove_layer", class: "MyModern" });
    const add = addLayerEdit(patterned)!;
    expect(add).toMatchObject({ op: "add_layer", class: "SlidingGlobal", path: ["pattern"] });
    expect((add["node"] as { kind: string }).kind).toBe("block");
    expect(removeLayerEdit(patterned, 1)).toEqual({ op: "remove_layer", class: "SlidingGlobal", path: ["pattern"], index: 1 });
  });
});

describe("parsePattern", () => {
  it("reads kind:kind a:b", () => {
    expect(parsePattern("sliding:global 3:1")).toEqual({ first: "sliding", second: "global", counts: [3, 1] });
    expect(parsePattern(" global:sliding 1:1 ")).toMatchObject({ first: "global" });
  });
  it("says what is wrong", () => {
    expect(parsePattern("sliding 3")).toMatch(/kind:kind a:b/);
    expect(parsePattern("local:global 3:1")).toBe("local is not a layer kind here: use sliding or global");
    expect(parsePattern("sliding:global 0:0")).toBe("a pattern needs at least one layer");
  });
});

describe("patternItems", () => {
  it("makes sliding layers with a window and global ones without", () => {
    const items = patternItems(uniform, { first: "sliding", second: "global", counts: [3, 1] }, 64);
    expect(Array.isArray(items)).toBe(true);
    const windows = (items as unknown as { args: { attn: { args: Record<string, { value?: number }> } } }[]).map((i) => i.args.attn.args["window"]?.value ?? null);
    expect(windows).toEqual([64, 64, 64, null]);
  });
  it("refuses what it cannot build", () => {
    expect(patternItems(uniform, { first: "sliding", second: "global", counts: [1, 1] }, 0)).toBe("the window is a whole number of tokens, at least 1");
    expect(patternItems({ ...uniform, args: {} }, { first: "sliding", second: "global", counts: [1, 1] }, 8)).toBe("this model has no layer block to build a pattern from");
  });
});
