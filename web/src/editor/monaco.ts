// Monaco, bundled (compose is offline: no CDN). Only the editor core, its features and the
// Python grammar are loaded; this module is imported lazily so pages without an editor never
// pay for it.
import * as monaco from "monaco-editor/editor/editor.api.js";
import "monaco-editor/features/register.all.js";
import "monaco-editor/languages/definitions/python/register.js";
import EditorWorker from "monaco-editor/editor/editor.worker.js?worker";
import { readToken, subscribe } from "../styles/theme";

(self as unknown as { MonacoEnvironment: unknown }).MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

export const THEME = "nanoscope";

// Monaco wants six hex digits for a token colour; the tokens may be #rgb or rgb(a)().
export function hex6(value: string): string {
  const v = value.trim().toLowerCase();
  const short = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/.exec(v);
  if (short) return short.slice(1).map((d) => d + d).join("");
  if (/^#[0-9a-f]{6}([0-9a-f]{2})?$/.test(v)) return v.slice(1, 7);
  const rgb = /^rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/.exec(v);
  if (rgb) return rgb.slice(1, 4).map((n) => Number(n).toString(16).padStart(2, "0")).join("");
  throw new Error(`cannot turn ${JSON.stringify(value)} into a colour`);
}

// The same for the `colors` map, keeping an alpha channel (the selection is translucent).
export function colorHex(value: string): string {
  const alpha = /^rgba\(\s*\d+[\s,]+\d+[\s,]+\d+[\s,/]+([\d.]+)\s*\)$/.exec(value.trim());
  const hexAlpha = /^#[0-9a-f]{6}([0-9a-f]{2})$/i.exec(value.trim());
  const a = alpha ? Math.round(Number(alpha[1]) * 255).toString(16).padStart(2, "0") : (hexAlpha?.[1] ?? "");
  return `#${hex6(value)}${a}`;
}

function rule(token: string, name: string, extra: Record<string, unknown> = {}) {
  return { token, foreground: hex6(readToken(name)), ...extra };
}

// Redefined whenever the page theme changes: the tokens are read from the computed style.
export function defineTheme(dark: boolean): void {
  monaco.editor.defineTheme(THEME, {
    base: dark ? "vs-dark" : "vs",
    inherit: true,
    rules: [
      rule("keyword", "syn-keyword"),
      rule("string", "syn-string"),
      rule("number", "syn-number"),
      rule("type", "syn-type"),
      rule("comment", "syn-comment", { fontStyle: "italic" }),
    ],
    colors: {
      "editor.background": colorHex(readToken("surface-raised")),
      "editor.foreground": colorHex(readToken("ink")),
      "editor.lineHighlightBackground": colorHex(readToken("surface")),
      "editor.selectionBackground": colorHex(readToken("selection")),
      "editorLineNumber.foreground": colorHex(readToken("ink-muted")),
      "editorCursor.foreground": colorHex(readToken("accent")),
      "editorWidget.background": colorHex(readToken("surface-raised")),
      "editorWidget.border": colorHex(readToken("line")),
      "editorError.foreground": colorHex(readToken("bad")),
      "editorWarning.foreground": colorHex(readToken("warn")),
      "editorInfo.foreground": colorHex(readToken("accent")),
    },
  });
  monaco.editor.setTheme(THEME);
}

let watching = false;
export function followTheme(isDark: () => boolean): void {
  defineTheme(isDark());
  if (watching) return;
  watching = true;
  subscribe((theme) => defineTheme(theme === "dark"));
}

export { monaco };
