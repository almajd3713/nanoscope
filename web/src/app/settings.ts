import { useSyncExternalStore } from "react";

// Browser-only settings, kept in localStorage inside try/catch and in memory, so a blocked
// store still works for the page view.
const SHOW_COMMANDS = "nanoscope.showCommands";
const listeners = new Set<() => void>();
let showCommands: boolean | null = null;

function readShowCommands(): boolean {
  try {
    return window.localStorage.getItem(SHOW_COMMANDS) === "1";
  } catch {
    return false; // off by default
  }
}

export function getShowCommands(): boolean {
  showCommands ??= readShowCommands();
  return showCommands;
}

export function setShowCommands(on: boolean): void {
  showCommands = on;
  try {
    window.localStorage.setItem(SHOW_COMMANDS, on ? "1" : "0");
  } catch {
    /* holds for this page view only */
  }
  for (const fn of listeners) fn();
}

// For tests: forget the in-memory value so the next read goes to storage again.
export function resetSettingsCache(): void {
  showCommands = null;
  for (const fn of listeners) fn();
}

export function useShowCommands(): boolean {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    getShowCommands,
  );
}
