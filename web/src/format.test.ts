import { describe, expect, it } from "vitest";
import { computeText, minutesText } from "./format";

describe("format", () => {
  it("prints minutes like the library", () => {
    expect(minutesText(0.5)).toBe("0.5 min");
    expect(minutesText(15)).toBe("15 min");
    expect(minutesText(89)).toBe("89 min");
    expect(minutesText(90)).toBe("1.5 h");
    expect(minutesText(240)).toBe("4 h");
  });

  it("joins the variants, or prints a dash", () => {
    expect(computeText({ cpu: { estimate_minutes: 15 }, gpu: { estimate_minutes: 10 } })).toBe("cpu 15 min / gpu 10 min");
    expect(computeText({})).toBe("-");
  });
});
