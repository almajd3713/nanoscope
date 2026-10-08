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

function rule(token: string, name: string, extra: Record<string, unknown> = {}) {
  return { token, foreground: readToken(name).replace("#", ""), ...extra };
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
      "editor.background": readToken("surface-raised"),
      "editor.foreground": readToken("ink"),
      "editor.lineHighlightBackground": readToken("surface"),
      "editor.selectionBackground": readToken("selection"),
      "editorLineNumber.foreground": readToken("ink-muted"),
      "editorCursor.foreground": readToken("accent"),
      "editorWidget.background": readToken("surface-raised"),
      "editorWidget.border": readToken("line"),
      "editorError.foreground": readToken("bad"),
      "editorWarning.foreground": readToken("warn"),
      "editorInfo.foreground": readToken("accent"),
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
