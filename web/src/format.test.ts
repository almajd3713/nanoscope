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

import { clockText, countText, stampText } from "./format";

describe("format (numbers and time)", () => {
  it("attaches the SI suffix to the number", () => {
    expect(countText(13200000)).toBe("13.2M");
    expect(countText(726000)).toBe("726k");
    expect(countText(1250000)).toBe("1.25M");
    expect(countText(1.8e9)).toBe("1.8G");
    expect(countText(42)).toBe("42");
  });

  it("prints a clock", () => {
    expect(clockText(41)).toBe("0:41");
    expect(clockText(125)).toBe("2:05");
    expect(clockText(3725)).toBe("1:02:05");
    expect(clockText(-3)).toBe("0:00");
  });

  it("prints a stamp as ISO date and 24-hour time", () => {
    expect(stampText("2026-10-07T14:03:09")).toBe("2026-10-07 14:03");
    expect(stampText("not a date")).toBe("not a date");
  });
});
