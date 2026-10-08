import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const made: { opts: Record<string, unknown>; data: unknown[][] }[] = [];
vi.mock("uplot", () => ({
  default: class {
    setData() {}
    setSize() {}
    destroy() {}
    constructor(opts: Record<string, unknown>, data: unknown[][]) {
      made.push({ opts, data });
    }
  },
}));

import compare from "../../test/fixtures/compare.json";
import { CompareCurves } from "./CompareCurves";

beforeEach(() => {
  made.length = 0;
  for (const [name, value] of [["series-1", "#0072b2"], ["series-2", "#c4530a"], ["baseline", "#6b7384"], ["band", "rgba(1,2,3,.2)"]]) {
    document.documentElement.style.setProperty(`--${name}`, value as string);
  }
});

const sets = compare.curves.map((c) => ({ label: c.label, baseline: c.label === "GPT2", seeds: c.seeds as { seed: number; points: [number, number][] }[] }));

describe("CompareCurves", () => {
  it("draws every seed, seed 0 at full weight and the others quiet, in the set's colour", () => {
    render(<CompareCurves sets={sets} xLabel="tokens" yLabel="val_bpb (lower is better)" />);
    const series = made[0]!.opts["series"] as { label?: string; stroke?: string; width?: number; dash?: number[] }[];
    // x, then 3 sets x 3 seeds, then the band's two hidden series
    expect(series).toHaveLength(1 + 9 + 2);
    const bigram = series.filter((s) => s.label?.startsWith("Bigram"));
    expect(bigram.map((s) => s.width)).toEqual([1.5, 1, 1]);
    expect(new Set(bigram.map((s) => s.stroke))).toEqual(new Set(["#0072b2"]));
    const modern = series.filter((s) => s.label?.startsWith("Modern"));
    expect(new Set(modern.map((s) => s.stroke))).toEqual(new Set(["#c4530a"]));
  });

  it("draws the baseline's seeds dashed and neutral, with a band between their extremes", () => {
    render(<CompareCurves sets={sets} xLabel="tokens" yLabel="y" />);
    const series = made[0]!.opts["series"] as { label?: string; stroke?: string; dash?: number[] }[];
    const gpt2 = series.filter((s) => s.label?.startsWith("GPT2"));
    expect(gpt2).toHaveLength(3);
    for (const s of gpt2) {
      expect(s.stroke).toBe("#6b7384");
      expect(s.dash).toEqual([5, 4]);
    }
    expect(made[0]!.opts["bands"]).toEqual([{ series: [11, 10], fill: "rgba(1,2,3,.2)" }]);
    const data = made[0]!.data;
    const lows = data[10] as number[];
    const highs = data[11] as number[];
    expect(lows.every((v, i) => highs[i] !== null && v <= (highs[i] as number))).toBe(true);
  });
});
