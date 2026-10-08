import { describe, expect, it } from "vitest";
import { align, evalPoints } from "./curveData";

describe("curveData", () => {
  it("keeps only the rows with a validation bpb, in order", () => {
    const rows = [{ step: 10, loss: 7 }, { step: 50, val_bpb: 2 }, { step: 60, loss: 6 }, { step: 100, val_bpb: 1.5 }];
    expect(evalPoints(rows)).toEqual({ steps: [50, 100], values: [2, 1.5] });
  });

  it("aligns the run with the shipped seeds' range on a shared step axis", () => {
    const run = { steps: [50, 100], values: [2, 1.5] };
    const seeds = [
      { steps: [50, 100, 150], values: [2.1, 1.6, 1.4] },
      { steps: [50, 100, 150], values: [1.9, 1.4, 1.3] },
    ];
    expect(align(run, seeds)).toEqual({
      x: [50, 100, 150],
      run: [2, 1.5, null],
      low: [1.9, 1.4, 1.3],
      high: [2.1, 1.6, 1.4],
    });
  });

  it("draws no range where a seed has no eval, and none without baselines", () => {
    const out = align({ steps: [50], values: [2] }, [
      { steps: [50, 100], values: [2, 1.5] },
      { steps: [50], values: [2.2] },
    ]);
    expect(out.low).toEqual([2, null]);
    expect(align({ steps: [50], values: [2] }, []).low).toEqual([null]);
  });
});
