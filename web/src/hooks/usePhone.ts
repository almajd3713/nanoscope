import { useSyncExternalStore } from "react";

// The design's phone breakpoint (--bp-phone). Only the run page, the runs list, the queue and lesson
// reading are drawn for a phone; the editor, the graph and the study builder show a notice instead.
const QUERY = "(max-width: 600px)";

function media(): MediaQueryList | null {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia(QUERY) : null;
}

function subscribe(onChange: () => void): () => void {
  const m = media();
  m?.addEventListener?.("change", onChange);
  return () => m?.removeEventListener?.("change", onChange);
}

export function usePhone(): boolean {
  return useSyncExternalStore(subscribe, () => media()?.matches ?? false, () => false);
}
