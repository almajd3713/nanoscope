// Light, dark and system (the default) on <html data-theme>. "system" removes the attribute so
// tokens.css follows prefers-color-scheme. Monaco, uPlot and React Flow read colors with
// readToken() and redraw from subscribe().
export type ThemeChoice = "light" | "dark" | "system";
export type Theme = "light" | "dark";

const KEY = "nanoscope.theme";
const listeners = new Set<(theme: Theme) => void>();

function darkQuery(): MediaQueryList | null {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-color-scheme: dark)")
    : null;
}

export function getChoice(): ThemeChoice {
  try {
    const v = window.localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* storage can be blocked or throw; fall through to system */
  }
  return "system";
}

export function resolvedTheme(): Theme {
  const choice = getChoice();
  if (choice !== "system") return choice;
  return darkQuery()?.matches ? "dark" : "light";
}

function apply(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
}

function notify(): void {
  const theme = resolvedTheme();
  for (const fn of listeners) fn(theme);
}

export function setChoice(choice: ThemeChoice): void {
  try {
    if (choice === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, choice);
  } catch {
    /* the choice still applies for this page view */
  }
  apply(choice);
  // The stored value may be unreadable, so tell listeners what was just applied.
  const theme: Theme = choice === "system" ? (darkQuery()?.matches ? "dark" : "light") : choice;
  for (const fn of listeners) fn(theme);
}

let stopWatching: (() => void) | null = null;

// Call once at startup: applies the stored choice and follows the OS while on "system".
export function initTheme(): void {
  apply(getChoice());
  stopWatching?.();
  const query = darkQuery();
  if (!query) return;
  const onChange = () => {
    if (getChoice() === "system") notify();
  };
  query.addEventListener("change", onChange);
  stopWatching = () => query.removeEventListener("change", onChange);
}

export function subscribe(fn: (theme: Theme) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// The computed value of a token, e.g. readToken("surface-raised") -> "#ffffff".
export function readToken(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}
