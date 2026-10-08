import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import samples from "../../test/fixtures/samples.json";
import { Samples } from "./Samples";

describe("Samples", () => {
  it("shows the newest sample and a tab per step, oldest first", () => {
    render(<Samples samples={samples} caption="every 50 steps, 200 tokens, temperature 0.8" />);
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(samples.map((s) => `step ${s.step}`));
    expect(screen.getByRole("tabpanel").textContent).toBe(samples[2]!.text);
    expect(screen.getByText("every 50 steps, 200 tokens, temperature 0.8")).toBeTruthy();
  });

  it("switches samples, and keeps the one you chose when a new one arrives", async () => {
    const { rerender } = render(<Samples samples={samples} />);
    await userEvent.click(screen.getByRole("tab", { name: `step ${samples[0]!.step}` }));
    expect(screen.getByRole("tabpanel").textContent).toBe(samples[0]!.text);
    rerender(<Samples samples={[...samples, { step: 999, text: "newest" }]} />);
    expect(screen.getByRole("tabpanel").textContent).toBe(samples[0]!.text);
    expect(screen.getByRole("tab", { name: "step 999" })).toBeTruthy();
  });

  it("follows the newest while you have not chosen", () => {
    const { rerender } = render(<Samples samples={samples} />);
    rerender(<Samples samples={[...samples, { step: 999, text: "newest" }]} />);
    expect(screen.getByRole("tabpanel").textContent).toBe("newest");
  });

  it("renders nothing before the first sample", () => {
    const { container } = render(<Samples samples={[]} />);
    expect(container.innerHTML).toBe("");
  });
});
