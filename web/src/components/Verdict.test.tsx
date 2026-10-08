import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Verdict } from "./Verdict";
import { StateTag } from "./StateTag";

describe("a verdict or state is never colour alone", () => {
  it.each(["better", "worse", "within noise", "no CI"])("%s has a glyph and its word", (verdict) => {
    const { container } = render(<Verdict verdict={verdict} />);
    expect(screen.getByText(verdict)).toBeTruthy();
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("the baseline is the word alone, and an unknown verdict is refused", () => {
    const { container } = render(<Verdict verdict="baseline" />);
    expect(screen.getByText("baseline")).toBeTruthy();
    expect(container.querySelector("svg")).toBeNull();
    expect(() => render(<Verdict verdict="great" />)).toThrow(/unknown verdict/);
  });

  it.each(["queued", "preparing", "running", "done", "stopped", "cancelling", "cancelled", "failed"])(
    "state %s has a glyph and its word",
    (state) => {
      const { container } = render(<StateTag state={state} />);
      expect(screen.getByText(state)).toBeTruthy();
      expect(container.querySelector("svg")).not.toBeNull();
    },
  );

  it("an unknown state is refused", () => {
    expect(() => render(<StateTag state="in progress" />)).toThrow(/unknown state/);
  });
});
