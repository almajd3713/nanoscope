import { useSyncExternalStore } from "react";

export const LEVELS = ["Learn", "Tinker", "Research", "Extend"] as const;
export type Level = (typeof LEVELS)[number];

const KEY = "nanoscope.level";
const DEFAULT: Level = "Learn";
const listeners = new Set<() => void>();
// Kept in memory too, so the switch works when storage is blocked.
let current: Level | null = null;

function read(): Level {
  try {
    const v = window.localStorage.getItem(KEY);
    if (LEVELS.includes(v as Level)) return v as Level;
  } catch {
    /* a throwing store means Learn */
  }
  return DEFAULT;
}

export function getLevel(): Level {
  current ??= read();
  return current;
}

export function setLevel(level: Level): void {
  current = level;
  try {
    window.localStorage.setItem(KEY, level);
  } catch {
    /* the choice still holds for this page view */
  }
  for (const fn of listeners) fn();
}

// For tests: forget the in-memory value so the next read goes to storage again.
export function resetLevelCache(): void {
  current = null;
  for (const fn of listeners) fn();
}

export function useLevel(): Level {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    getLevel,
  );
}

// A control shows when the current level is at or above the lowest level that shows it.
export function atLeast(level: Level, minimum: Level): boolean {
  return LEVELS.indexOf(level) >= LEVELS.indexOf(minimum);
}
