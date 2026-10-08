import { useSyncExternalStore } from "react";

// Set when any request comes back 401; the app then shows the login page instead of the screen.
let signedOut = false;
const listeners = new Set<() => void>();

export function markSignedOut(): void {
  if (signedOut) return;
  signedOut = true;
  for (const fn of listeners) fn();
}

export function resetAuth(): void {
  signedOut = false;
  for (const fn of listeners) fn();
}

export function useSignedOut(): boolean {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    () => signedOut,
  );
}
