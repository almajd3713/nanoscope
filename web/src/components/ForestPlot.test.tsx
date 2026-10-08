import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import compare from "../../test/fixtures/compare.json";
import { axis, ForestPlot, niceStep } from "./ForestPlot";

describe("axis", () => {
  it("picks round steps and always includes zero", () => {
    expect(niceStep(0.6)).toBeCloseTo(0.2);
    expect(niceStep(1)).toBeCloseTo(0.25);
    const a = axis([-0.176, -0.153, 0.38, 0.469]);
    expect(a.min).toBeLessThanOrEqual(-0.176);
    expect(a.max).toBeGreaterThanOrEqual(0.469);
    expect(a.ticks).toContain(0);
  });
});

describe("ForestPlot", () => {
  const rows = compare.rows.map((r) => ({ label: r.label, delta: r.delta }));

  it("draws one point per model with a whisker, and none for the baseline, from the API numbers", () => {
    const { container } = render(<ForestPlot caption="Δ val_bpb vs GPT2, 95% CI" unit="bpb" rows={rows} />);
    expect(container.querySelectorAll("rect")).toHaveLength(2);
    expect(container.querySelectorAll("line.ci, line[class*='ci']")).toHaveLength(2);
    expect(screen.getByText("no difference")).toBeTruthy();
    expect(screen.getByText("← better")).toBeTruthy();
    expect(screen.getByText("worse →")).toBeTruthy();
  });

  it("has a text equivalent with the numbers", () => {
    render(<ForestPlot caption="Δ val_bpb vs GPT2, 95% CI" unit="bpb" rows={rows} />);
    const label = screen.getByRole("img").getAttribute("aria-label")!;
    expect(label).toContain("Modern: -0.165 [-0.176, -0.153]");
    expect(label).toContain("Bigram: 0.424 [0.380, 0.469]");
  });

  it("draws a point without a whisker when there is no interval", () => {
    const { container } = render(
      <ForestPlot caption="c" unit="bpb" rows={[{ label: "A", delta: { mean: -0.1, ci95_low: null, ci95_high: null } }, { label: "B", delta: null }]} />,
    );
    expect(container.querySelectorAll("rect")).toHaveLength(1);
    expect(container.querySelectorAll("line[class*='ci']")).toHaveLength(0);
  });
});
