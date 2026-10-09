import { describe, expect, it, vi } from "vitest";
import { documentUri, LspClient, lspUrl, markdownOf, toMarker, type Socket } from "./lsp";

// A server that answers like basedpyright does, over a fake socket.
class FakeSocket implements Socket {
  onopen: Socket["onopen"] = null;
  onmessage: Socket["onmessage"] = null;
  onerror: Socket["onerror"] = null;
  onclose: Socket["onclose"] = null;
  sent: { id?: number; method: string; params: Record<string, unknown> }[] = [];
  constructor(private handler: (message: FakeSocket["sent"][number], reply: (m: object) => void) => void = () => undefined) {}
  send(data: string) {
    const message = JSON.parse(data) as FakeSocket["sent"][number];
    this.sent.push(message);
    this.handler(message, (m) => this.onmessage?.({ data: JSON.stringify({ jsonrpc: "2.0", ...m }) }));
  }
  close() {
    this.onclose?.({});
  }
}

const initialize = (message: FakeSocket["sent"][number], reply: (m: object) => void) => {
  if (message.method === "initialize") reply({ id: message.id, result: { capabilities: {}, serverInfo: { name: "basedpyright", version: "1.40.2" } } });
};

describe("language server client", () => {
  it("falls back when the service is not running: connect rejects and says so", async () => {
    const socket = new FakeSocket();
    const attempt = LspClient.connect("ws://127.0.0.1:8767/lsp", "file:///w", () => socket);
    socket.onerror?.({});
    await expect(attempt).rejects.toThrow("the language server at ws://127.0.0.1:8767/lsp is not running");
  });

  it("falls back when the connection closes before the handshake or never answers", async () => {
    const closed = new FakeSocket();
    const first = LspClient.connect("ws://x/lsp", "file:///w", () => closed);
    closed.close();
    await expect(first).rejects.toThrow("closed the connection");

    vi.useFakeTimers();
    const silent = new FakeSocket();
    const second = LspClient.connect("ws://x/lsp", "file:///w", () => silent);
    silent.onopen?.({});
    const caught = expect(second).rejects.toThrow("no answer from the language server");
    await vi.advanceTimersByTimeAsync(9000);
    await caught;
    vi.useRealTimers();
  });

  it("falls back when the socket cannot even be made", async () => {
    await expect(
      LspClient.connect("ws://x/lsp", "file:///w", () => {
        throw new Error("blocked by the page's policy");
      }),
    ).rejects.toThrow("blocked by the page's policy");
  });

  it("initializes, learns the server's name and answers its configuration request", async () => {
    const socket = new FakeSocket(initialize);
    const attempt = LspClient.connect("ws://x/lsp", "file:///nanoscope/workspace", () => socket);
    socket.onopen?.({});
    const client = await attempt;
    expect(client.server).toEqual({ name: "basedpyright", version: "1.40.2" });
    expect(socket.sent.map((m) => m.method)).toEqual(["initialize", "initialized"]);
    socket.onmessage?.({ data: JSON.stringify({ jsonrpc: "2.0", id: 9, method: "workspace/configuration", params: { items: [{ section: "python.analysis" }, { section: "nothing" }] } }) });
    const answer = socket.sent.at(-1) as unknown as { id: number; result: unknown[] };
    expect(answer.id).toBe(9);
    expect(answer.result).toEqual([{ typeCheckingMode: "standard" }, null]);
  });

  it("turns published diagnostics into markers and sends edits with rising versions", async () => {
    const socket = new FakeSocket(initialize);
    const attempt = LspClient.connect("ws://x/lsp", "file:///w", () => socket);
    socket.onopen?.({});
    const client = await attempt;
    const seen: { uri: string; messages: string[] }[] = [];
    client.onDiagnostics((uri, diagnostics) => seen.push({ uri, messages: diagnostics.map((d) => toMarker(d).message) }));
    socket.onmessage?.({
      data: JSON.stringify({
        jsonrpc: "2.0",
        method: "textDocument/publishDiagnostics",
        params: { uri: "file:///w/a.py", diagnostics: [{ range: { start: { line: 26, character: 32 }, end: { line: 26, character: 40 } }, severity: 1, message: 'No parameter named "keep_dim"', code: "reportCallIssue", source: "basedpyright" }] },
      }),
    });
    expect(seen).toEqual([{ uri: "file:///w/a.py", messages: ['No parameter named "keep_dim"'] }]);
    expect(toMarker({ range: { start: { line: 26, character: 32 }, end: { line: 26, character: 40 } }, message: "m", code: "reportCallIssue" })).toMatchObject({ line: 27, column: 33, endColumn: 41, severity: "error", source: "basedpyright", code: "reportCallIssue" });

    client.open("file:///w/a.py", "x = 1");
    client.change("file:///w/a.py", "x = 2");
    client.change("file:///w/a.py", "x = 3");
    const changes = socket.sent.filter((m) => m.method === "textDocument/didChange").map((m) => (m.params["textDocument"] as { version: number }).version);
    expect(changes).toEqual([2, 3]);
  });

  it("answers hover, completion and signature requests, and rejects them when the socket drops", async () => {
    const socket = new FakeSocket((message, reply) => {
      initialize(message, reply);
      if (message.method === "textDocument/hover") reply({ id: message.id, result: { contents: { kind: "markdown", value: "(method) def norm()" } } });
      if (message.method === "textDocument/completion") reply({ id: message.id, result: { items: [{ label: "norm" }] } });
      if (message.method === "textDocument/signatureHelp") reply({ id: message.id, result: null });
    });
    const attempt = LspClient.connect("ws://x/lsp", "file:///w", () => socket);
    socket.onopen?.({});
    const client = await attempt;
    const at = { line: 26, character: 19 };
    expect(markdownOf((await client.hover("file:///w/a.py", at))?.contents)).toBe("(method) def norm()");
    expect((await client.completion("file:///w/a.py", at)).map((i) => i.label)).toEqual(["norm"]);
    expect(await client.signatureHelp("file:///w/a.py", at)).toBeNull();

    const silent = new FakeSocket(initialize);
    const second = LspClient.connect("ws://x/lsp", "file:///w", () => silent);
    silent.onopen?.({});
    const other = await second;
    const waiting = other.hover("file:///w/a.py", at);
    silent.onclose?.({});
    await expect(waiting).rejects.toThrow("connection closed");
    expect(other.closed).toBe(true);
  });

  it("builds the service address and a document uri", () => {
    expect(lspUrl({ protocol: "http:", hostname: "127.0.0.1" }, 8767)).toBe("ws://127.0.0.1:8767/lsp");
    expect(lspUrl({ protocol: "https:", hostname: "lab.example" }, 8767)).toBe("wss://lab.example:8767/lsp");
    expect(lspUrl({ protocol: "http:", hostname: "::1" }, 8767)).toBe("ws://[::1]:8767/lsp");
    expect(documentUri("/nanoscope/workspace/", "models/my lm.py")).toBe("file:///nanoscope/workspace/models/my%20lm.py");
  });
});
