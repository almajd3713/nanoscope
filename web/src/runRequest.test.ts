import { describe, expect, it } from "vitest";
import { buildKwargs, commands, parseSeeds, parseValue, seedList, showValue } from "./runRequest";

describe("parseValue", () => {
  it("gives a value the JSON type its annotation names", () => {
    expect(parseValue("128", "int")).toBe(128);
    expect(parseValue("0.003", "float")).toBe(0.003);
    expect(parseValue("1e-4", "float")).toBe(0.0001);
    expect(parseValue("gelu", "str")).toBe("gelu");
    expect(parseValue("true", "bool")).toBe(true);
    expect(parseValue("", "int | None")).toBeNull();
    expect(parseValue("None", "int | None")).toBeNull();
    expect(parseValue("4", "int | None")).toBe(4);
  });

  it("leaves what is not a number as typed, for the library to refuse", () => {
    expect(parseValue("ten", "int")).toBe("ten");
    expect(parseValue("", "int")).toBe("");
  });

  it("reads unannotated and complex values as JSON, else as typed", () => {
    expect(parseValue("0.01", null)).toBe(0.01);
    expect(parseValue("[0.9, 0.95]", "tuple[float, float]")).toEqual([0.9, 0.95]);
    expect(parseValue("fp16", null)).toBe("fp16");
  });
});

describe("seeds", () => {
  it("takes a count or a list", () => {
    expect(parseSeeds("3")).toBe(3);
    expect(parseSeeds("0, 4,7")).toEqual([0, 4, 7]);
    expect(parseSeeds("x")).toBe("x");
    expect(seedList(3)).toEqual([0, 1, 2]);
    expect(seedList([0, 4])).toEqual([0, 4]);
    expect(seedList("x")).toEqual([]);
  });
});

describe("buildKwargs", () => {
  const fields = [
    { name: "d_model", annotation: "int", default: 128 },
    { name: "n_heads", annotation: "int", default: 4 },
    { name: "qk_norm", annotation: "bool", default: true },
  ];
  it("sends only what differs from the default", () => {
    const text: Record<string, string> = { d_model: "128", n_heads: "8", qk_norm: "false" };
    expect(buildKwargs(fields, (n) => text[n] ?? "", [{ key: "lr", value: "0.01" }, { key: "", value: "1" }])).toEqual({
      n_heads: 8,
      qk_norm: false,
      lr: 0.01,
    });
  });

  it("shows defaults as the text a field holds", () => {
    expect(showValue(128)).toBe("128");
    expect(showValue(null)).toBe("");
    expect(showValue(true)).toBe("true");
    expect(showValue("gelu")).toBe("gelu");
  });
});

describe("commands", () => {
  it("writes the CLI and Python call for the same request", () => {
    const out = commands({
      model: "nanoscope.models.gpt2:GPT2",
      shippedClass: "GPT2",
      preset: "tinystories-5min",
      seeds: 3,
      kwargs: { n_heads: 8, lr: 0.01 },
    });
    expect(out.cli).toBe("nanoscope run nanoscope.models.gpt2:GPT2 --preset tinystories-5min --seeds 3 --set n_heads=8 lr=0.01");
    expect(out.python).toBe(
      'from nanoscope import run\nfrom nanoscope.models import GPT2\n\nrun(GPT2, preset="tinystories-5min", seeds=3, n_heads=8, lr=0.01)',
    );
  });

  it("has no Python snippet for a workspace model, and omits a single seed", () => {
    const out = commands({ model: "lessons/x/starter.py:MyBigram", shippedClass: null, preset: "p", seeds: 1, kwargs: {} });
    expect(out.cli).toBe("nanoscope run lessons/x/starter.py:MyBigram --preset p");
    expect(out.python).toBeUndefined();
  });
});
