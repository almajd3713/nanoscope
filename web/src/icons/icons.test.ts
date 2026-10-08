// @vitest-environment node
import { describe, expect, it } from "vitest";
import * as icons from "./index";

// The design system's Icons asset group (README table), in its names.
const ALLOWLIST = [
  "check", "x", "circle", "circle-half", "play", "pause", "prohibit", "minus", "info", "warning",
  "lock-simple", "lock-simple-open", "seal-check", "dots-six-vertical", "copy", "copy-simple",
  "arrow-clockwise", "stop", "plus", "trash", "pencil-simple", "arrow-square-out", "caret-down",
  "caret-right", "magnifying-glass", "cpu", "graphics-card", "sun", "moon",
];

const pascal = (name: string) =>
  name.split("-").map((p) => p[0]!.toUpperCase() + p.slice(1)).join("");

describe("icons", () => {
  it("allowlist has 29 icons", () => {
    expect(ALLOWLIST).toHaveLength(29);
  });

  it("exports exactly the allowlist", () => {
    expect(Object.keys(icons).sort()).toEqual(ALLOWLIST.map(pascal).sort());
  });

  it("every export is a component", () => {
    for (const [name, icon] of Object.entries(icons)) {
      expect(icon, name).toBeTruthy();
    }
  });
});
