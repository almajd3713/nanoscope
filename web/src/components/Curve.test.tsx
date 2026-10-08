import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const made: { opts: Record<string, unknown>; data: unknown; setData: ReturnType<typeof vi.fn>; destroy: ReturnType<typeof vi.fn> }[] = [];

vi.mock("uplot", () => ({
  default: class {
    setData = vi.fn();
    destroy = vi.fn();
    setSize = vi.fn();
    constructor(opts: Record<string, unknown>, data: unknown) {
      made.push({ opts, data, setData: this.setData, destroy: this.destroy });
    }
  },
}));
vi.mock("uplot/dist/uPlot.min.css", () => ({}));

import { setChoice } from "../styles/theme";
import { Curve } from "./Curve";

beforeEach(() => {
  made.length = 0;
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  document.documentElement.style.setProperty("--series-1", "#0072b2");
  document.documentElement.style.setProperty("--band", "rgba(107,115,132,0.18)");
});
afterEach(() => vi.unstubAllGlobals());

const run = { steps: [50, 100], values: [1.9, 1.5] };
const baseline = {
  name: "shipped gpt2, 2 seeds",
  seeds: [
    { steps: [50, 100], values: [2, 1.6] },
    { steps: [50, 100], values: [1.8, 1.4] },
  ],
};

describe("Curve", () => {
  it("draws the run's eval points with the shipped seeds' range as a band, in token colours", () => {
    render(<Curve name="tinystories-5min/gpt2/seed-0" run={run} baseline={baseline} yLabel="Validation bpb (lower is better)" />);
    expect(made).toHaveLength(1);
    const series = made[0]!.opts["series"] as { stroke?: string }[];
    expect(series[1]?.stroke).toBe("#0072b2");
    expect(made[0]!.opts["bands"]).toEqual([{ series: [3, 2], fill: "rgba(107,115,132,0.18)" }]);
    expect(made[0]!.data).toEqual([[50, 100], [1.9, 1.5], [1.8, 1.4], [2, 1.6]]);
    expect(screen.getByText("shipped gpt2, 2 seeds")).toBeTruthy();
  });

  it("draws no band without a baseline, and still lists the values as a table", () => {
    render(<Curve name="mine" run={run} yLabel="Validation bpb (lower is better)" />);
    expect(made[0]!.opts["bands"]).toEqual([]);
    expect(screen.getByText("Values as a table")).toBeTruthy();
    expect(screen.getByText("1.900")).toBeTruthy();
    expect(screen.getByText("1.500")).toBeTruthy();
  });

  it("appends new points without rebuilding the plot", () => {
    const { rerender } = render(<Curve name="mine" run={run} yLabel="y" />);
    rerender(<Curve name="mine" run={{ steps: [50, 100, 150], values: [1.9, 1.5, 1.3] }} yLabel="y" />);
    expect(made).toHaveLength(1);
    expect(made[0]!.setData).toHaveBeenLastCalledWith([[50, 100, 150], [1.9, 1.5, 1.3], [null, null, null], [null, null, null]]);
  });

  it("rebuilds with the new token colours when the theme changes", () => {
    render(<Curve name="mine" run={run} yLabel="y" />);
    document.documentElement.style.setProperty("--series-1", "#4ea3e0");
    act(() => setChoice("dark"));
    expect(made).toHaveLength(2);
    expect(made[0]!.destroy).toHaveBeenCalled();
    expect((made[1]!.opts["series"] as { stroke?: string }[])[1]?.stroke).toBe("#4ea3e0");
  });
});
