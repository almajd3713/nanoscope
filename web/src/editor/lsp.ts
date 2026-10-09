// A small Language Server Protocol client over a WebSocket, for the editor's `lsp` service
// (basedpyright behind nanoscope/server/lsp_bridge.py). It speaks only what the editor shows:
// diagnostics, hover, completion and signature help. When the service is not running connect()
// rejects and the editor keeps ruff alone.
//
// Not monaco-languageclient: that library replaces Monaco's editor API with VS Code's services
// (a second editor bundle) to give features this editor does not have; this is a few hundred
// lines the page owns, with no extra dependency in an offline stack.

import type { Marker } from "./types";

export type Position = { line: number; character: number }; // 0-based, as the protocol says
export type Range = { start: Position; end: Position };
export type LspDiagnostic = { range: Range; severity?: 1 | 2 | 3 | 4; message: string; code?: string | number; source?: string };

export type Hover = { contents: string | { kind?: string; value: string } | (string | { value: string })[] } | null;
export type CompletionItem = {
  label: string;
  kind?: number;
  detail?: string;
  documentation?: string | { value: string };
  insertText?: string;
  textEdit?: { newText: string; range: Range };
};
export type SignatureHelp = {
  signatures: { label: string; documentation?: string | { value: string }; parameters?: { label: string | [number, number] }[] }[];
  activeSignature?: number;
  activeParameter?: number;
} | null;

export type ServerInfo = { name: string; version?: string };

type Json = Record<string, unknown>;
type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };

// What the browser's WebSocket offers, and what a test fakes.
export type Socket = {
  send(data: string): void;
  close(): void;
  onopen: ((event: unknown) => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: ((event: unknown) => void) | null;
  onclose: ((event: unknown) => void) | null;
};
export type SocketFactory = (url: string) => Socket;

const TIMEOUT_MS = 8000;

// basedpyright asks the client for its settings with workspace/configuration.
export const SETTINGS = { python: { analysis: { typeCheckingMode: "standard" } }, basedpyright: { analysis: { typeCheckingMode: "standard" } } };

export class LspClient {
  server: ServerInfo | null = null;
  private socket: Socket;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private versions = new Map<string, number>();
  private diagnosticsListeners = new Set<(uri: string, diagnostics: LspDiagnostic[]) => void>();
  private closeListeners = new Set<() => void>();
  closed = false;

  private constructor(socket: Socket) {
    this.socket = socket;
    socket.onmessage = (event) => this.receive(String(event.data));
    socket.onclose = () => this.shutdown();
    socket.onerror = () => this.shutdown();
  }

  // Opens the socket and does the initialize handshake; rejects when the service is not there.
  static connect(url: string, rootUri: string, makeSocket: SocketFactory = (u) => new WebSocket(u) as unknown as Socket): Promise<LspClient> {
    return new Promise((resolve, reject) => {
      let socket: Socket;
      try {
        socket = makeSocket(url);
      } catch (error) {
        reject(error instanceof Error ? error : new Error(String(error)));
        return;
      }
      const client = new LspClient(socket);
      const fail = (why: string) => {
        client.shutdown();
        reject(new Error(why));
      };
      const timer = setTimeout(() => fail(`no answer from the language server at ${url}`), TIMEOUT_MS);
      socket.onerror = () => {
        clearTimeout(timer);
        fail(`the language server at ${url} is not running`);
      };
      socket.onclose = () => {
        clearTimeout(timer);
        fail(`the language server at ${url} closed the connection`);
      };
      socket.onopen = () => {
        client
          .request("initialize", {
            processId: null,
            rootUri,
            workspaceFolders: [{ uri: rootUri, name: "workspace" }],
            capabilities: {
              textDocument: {
                synchronization: { didSave: false },
                hover: { contentFormat: ["markdown", "plaintext"] },
                completion: { completionItem: { documentationFormat: ["markdown", "plaintext"], snippetSupport: false } },
                signatureHelp: { signatureInformation: { documentationFormat: ["markdown", "plaintext"], parameterInformation: { labelOffsetSupport: true } } },
                publishDiagnostics: {},
              },
              workspace: { configuration: true },
            },
          })
          .then((result) => {
            clearTimeout(timer);
            client.server = ((result as Json)["serverInfo"] as ServerInfo | undefined) ?? null;
            socket.onerror = () => client.shutdown();
            socket.onclose = () => client.shutdown();
            client.notify("initialized", {});
            resolve(client);
          })
          .catch((error: Error) => {
            clearTimeout(timer);
            fail(error.message);
          });
      };
    });
  }

  private shutdown(): void {
    if (this.closed) return;
    this.closed = true;
    for (const p of this.pending.values()) p.reject(new Error("the language server connection closed"));
    this.pending.clear();
    for (const fn of this.closeListeners) fn();
  }

  onClose(fn: () => void): () => void {
    this.closeListeners.add(fn);
    return () => this.closeListeners.delete(fn);
  }

  onDiagnostics(fn: (uri: string, diagnostics: LspDiagnostic[]) => void): () => void {
    this.diagnosticsListeners.add(fn);
    return () => this.diagnosticsListeners.delete(fn);
  }

  private send(message: Json): void {
    if (!this.closed) this.socket.send(JSON.stringify({ jsonrpc: "2.0", ...message }));
  }

  private request(method: string, params: unknown): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ id, method, params });
    });
  }

  private notify(method: string, params: unknown): void {
    this.send({ method, params });
  }

  private receive(text: string): void {
    let message: Json;
    try {
      message = JSON.parse(text) as Json;
    } catch {
      return;
    }
    const id = message["id"] as number | string | undefined;
    const method = message["method"] as string | undefined;
    if (method === undefined && typeof id === "number") {
      const pending = this.pending.get(id);
      this.pending.delete(id);
      if (!pending) return;
      const error = message["error"] as { message?: string } | undefined;
      if (error) pending.reject(new Error(error.message ?? "the language server refused the request"));
      else pending.resolve(message["result"]);
      return;
    }
    if (method === "textDocument/publishDiagnostics") {
      const params = message["params"] as { uri: string; diagnostics: LspDiagnostic[] };
      for (const fn of this.diagnosticsListeners) fn(params.uri, params.diagnostics);
      return;
    }
    if (id !== undefined && method !== undefined) {
      // a request from the server: answer so it never waits
      if (method === "workspace/configuration") {
        const items = ((message["params"] as { items?: { section?: string }[] })["items"] ?? []).map((item) => {
          const section = item.section ?? "";
          return section.split(".").reduce<unknown>((value, key) => (value as Json | undefined)?.[key], SETTINGS) ?? null;
        });
        this.send({ id, result: items });
      } else {
        this.send({ id, result: null });
      }
    }
  }

  open(uri: string, text: string): void {
    this.versions.set(uri, 1);
    this.notify("textDocument/didOpen", { textDocument: { uri, languageId: "python", version: 1, text } });
  }

  change(uri: string, text: string): void {
    const version = (this.versions.get(uri) ?? 1) + 1;
    this.versions.set(uri, version);
    this.notify("textDocument/didChange", { textDocument: { uri, version }, contentChanges: [{ text }] });
  }

  closeDocument(uri: string): void {
    this.versions.delete(uri);
    this.notify("textDocument/didClose", { textDocument: { uri } });
  }

  hover(uri: string, position: Position): Promise<Hover> {
    return this.request("textDocument/hover", { textDocument: { uri }, position }) as Promise<Hover>;
  }

  async completion(uri: string, position: Position): Promise<CompletionItem[]> {
    const result = (await this.request("textDocument/completion", { textDocument: { uri }, position })) as CompletionItem[] | { items: CompletionItem[] } | null;
    return Array.isArray(result) ? result : (result?.items ?? []);
  }

  signatureHelp(uri: string, position: Position): Promise<SignatureHelp> {
    return this.request("textDocument/signatureHelp", { textDocument: { uri }, position }) as Promise<SignatureHelp>;
  }

  close(): void {
    if (this.closed) return;
    this.request("shutdown", null).catch(() => undefined);
    this.notify("exit", null);
    this.socket.close();
    this.shutdown();
  }
}

// The text a hover or documentation field carries, as Markdown.
export function markdownOf(value: string | { value: string } | (string | { value: string })[] | undefined | null): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(markdownOf).filter(Boolean).join("\n\n");
  return value.value;
}

const SEVERITY = { 1: "error", 2: "warning", 3: "info", 4: "info" } as const;

// A server diagnostic as an editor marker: positions turn from 0-based to 1-based, and the
// server's own name (basedpyright) says who found it.
export function toMarker(d: LspDiagnostic): Marker {
  return {
    line: d.range.start.line + 1,
    column: d.range.start.character + 1,
    endLine: d.range.end.line + 1,
    endColumn: d.range.end.character + 1,
    message: d.message,
    severity: SEVERITY[d.severity ?? 1],
    source: d.source ?? "basedpyright",
    code: d.code === undefined ? null : String(d.code),
  };
}

// file:///nanoscope/workspace/models/my_lm.py for a workspace path
export function documentUri(root: string, path: string): string {
  return `file://${root.replace(/\/$/, "")}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

export function lspUrl(location: { protocol: string; hostname: string }, port: number): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const host = location.hostname.includes(":") ? `[${location.hostname}]` : location.hostname;
  return `${scheme}://${host}:${port}/lsp`;
}
