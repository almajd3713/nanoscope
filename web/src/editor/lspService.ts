// The one connection to the language server this page view shares between editors. The editor
// asks for it when "Use the Python language server" is on; if the service is not running the
// state says `unavailable` and nothing else changes (ruff keeps checking).
import { useEffect, useSyncExternalStore } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useUseLsp } from "../app/settings";
import { LspClient, lspUrl, type ServerInfo } from "./lsp";

export type LspState =
  | { status: "off" }
  | { status: "connecting" }
  | { status: "connected"; client: LspClient; server: ServerInfo | null; root: string }
  | { status: "unavailable"; why: string };

let state: LspState = { status: "off" };
let attempt: { key: string; promise: Promise<void> } | null = null;
const listeners = new Set<() => void>();

function set(next: LspState): void {
  state = next;
  for (const fn of listeners) fn();
}

// Connects once per address; a closed connection or a refused one is retried only by a reload or
// by turning the setting off and on.
export function connectLsp(url: string, root: string): Promise<void> {
  const key = `${url}|${root}`;
  if (attempt?.key === key && state.status !== "off") return attempt.promise;
  set({ status: "connecting" });
  const promise = LspClient.connect(url, `file://${root}`).then(
    (client) => {
      client.onClose(() => {
        if (state.status === "connected" && state.client === client) set({ status: "unavailable", why: "the language server connection closed" });
      });
      set({ status: "connected", client, server: client.server, root });
    },
    (error: Error) => set({ status: "unavailable", why: error.message }),
  );
  attempt = { key, promise };
  return promise;
}

export function disconnectLsp(): void {
  if (state.status === "connected") state.client.close();
  attempt = null;
  set({ status: "off" });
}

// For tests.
export function resetLsp(): void {
  disconnectLsp();
}

const subscribe = (fn: () => void) => {
  listeners.add(fn);
  return () => listeners.delete(fn);
};

type ServerFacts = { workspace: string; lsp_port: number };

// The shared connection, while the setting is on.
export function useLspService(): LspState {
  const wanted = useUseLsp();
  const facts = useQuery({ queryKey: ["settings"], queryFn: () => unwrap(api.GET("/api/settings")) as unknown as Promise<ServerFacts> });
  const current = useSyncExternalStore(subscribe, () => state);
  const address = facts.data ? lspUrl(window.location, facts.data.lsp_port) : null;
  const root = facts.data?.workspace ?? null;
  useEffect(() => {
    if (!wanted) {
      if (state.status !== "off") disconnectLsp();
      return;
    }
    if (address && root) void connectLsp(address, root);
  }, [wanted, address, root]);
  if (!wanted) return { status: "off" };
  if (facts.isError) return { status: "unavailable", why: "the server's settings could not be read" };
  return current;
}
