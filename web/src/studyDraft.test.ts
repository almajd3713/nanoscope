import { describe, expect, it } from "vitest";
import { freshName, fromSpec, parseKwargs, parseSeedList, splitPairs, toSpec } from "./studyDraft";

const SPEC = {
  schema: 1,
  name: "m1-ablation",
  preset: "tinystories-5min",
  overrides: { max_steps: 100 },
  budget: { tokens: 4000000 },
  match: "params",
  baseline: "modern",
  seeds: [0, 1, 2],
  mode: "record",
  tolerance: 0.02,
  variants: [
    { name: "gpt2", model: "nanoscope.models.gpt2:GPT2" },
    { name: "modern", model: "nanoscope.models.modern:Modern", kwargs: { ffn_hidden: 384 } },
    { name: "no-rope", model: "nanoscope.models.modern:Modern", kwargs: { rope: false, ffn_hidden: 384 } },
  ],
  predictions: { gpt2: { val_bpb: 1.2 }, modern: { val_bpb: 1.06 } },
};

describe("study draft", () => {
  it("reads a spec and writes it back unchanged", () => {
    expect(toSpec(fromSpec(SPEC))).toEqual(SPEC);
  });

  it("keeps what the builder does not edit, such as a FLOPs budget and a match knob", () => {
    const spec = { ...SPEC, budget: { flops: 1e15 }, match_knob: "ffn_hidden", match_to: "modern", range: [64, 1024, 8] };
    const back = toSpec(fromSpec(spec));
    expect(back["budget"]).toEqual({ flops: 1e15 });
    expect(back["match_knob"]).toBe("ffn_hidden");
    expect(back["range"]).toEqual([64, 1024, 8]);
  });

  it("types keywords by what they look like, and splits only on commas outside brackets", () => {
    expect(splitPairs("a = [1, 2], b = 'x, y'")).toEqual([["a", "[1, 2]"], ["b", "'x, y'"]]);
    expect(parseKwargs("n_kv_heads = 4, rope = false, name = tied")).toEqual({ n_kv_heads: 4, rope: false, name: "tied" });
  });

  it("reads seeds as a count or a list and leaves nonsense for the library to refuse", () => {
    expect(parseSeedList("3")).toEqual([0, 1, 2]);
    expect(parseSeedList("0, 4,7")).toEqual([0, 4, 7]);
    expect(parseSeedList("a b")).toBe("a b");
  });

  it("drops an empty prediction and an empty budget", () => {
    const draft = fromSpec(SPEC);
    draft.budget = "";
    draft.variants[0]!.predicted = "";
    const spec = toSpec(draft);
    expect(spec["budget"]).toBeUndefined();
    expect(spec["predictions"]).toEqual({ modern: { val_bpb: 1.06 } });
  });

  it("names a new variant with the first free number", () => {
    expect(freshName([{ name: "variant-2", model: "m", kwargs: "", predicted: "" }])).toBe("variant-3");
  });
});
