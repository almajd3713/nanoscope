// @vitest-environment node
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import tokens from "../src/styles/tokens.json";
import { generate, resolveColor, type Tokens } from "./tokens";

const t = tokens as unknown as Tokens;

describe("tokens", () => {
  it("gives every color token a light and a dark value", () => {
    for (const c of t.color.tokens) {
      for (const theme of ["light", "dark"] as const) {
        expect(resolveColor(c, theme), `${c.name} ${theme}`).toBeTruthy();
      }
    }
  });

  it("resolves aliases to the token they name", () => {
    const focus = t.color.tokens.find((c) => c.name === "focus")!;
    expect(resolveColor(focus, "dark")).toBe("var(--accent)");
  });

  it("puts light on :root, dark on [data-theme] and under prefers-color-scheme", () => {
    const css = generate(t);
    expect(css).toContain(':root,\n[data-theme="light"] {');
    expect(css).toContain('[data-theme="dark"] {');
    expect(css).toContain(':root:not([data-theme="light"])');
    expect(css).toContain("--accent: #0a4bc4;");
    expect(css).toContain("--accent: #6e9bff;");
  });

  it("emits a class per type style, with tabular figures for values", () => {
    const css = generate(t);
    expect(css).toMatch(/\.title \{[^}]*font-size: 22px;/);
    expect(css).toMatch(/\.value \{[^}]*tabular-nums/);
  });

  it("matches the committed tokens.css", () => {
    const committed = readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8");
    expect(committed).toBe(generate(t));
  });
});
