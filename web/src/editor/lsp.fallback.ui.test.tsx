import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import server from "../../test/fixtures/settings.json";
import unlocksFirst from "../../test/fixtures/unlocks-first-run.json";
import data from "../../test/fixtures/data.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { resetLevelCache, setLevel } from "../app/level";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands, setUseLsp } from "../app/settings";
import { Editor } from "../components/Editor";
import { Settings } from "../pages/Settings";
import { resetLsp } from "./lspService";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

const FILE = { path: "models/my_lm.py", content: "x = 1\n", etag: "e1" };
const URI = `file://${server.workspace}/models/my_lm.py`;
const ROUTES = {
  "GET /api/files/models/my_lm.py": { body: FILE },
  "GET /api/settings": { body: server },
  "POST /api/files/models/my_lm.py/lint": { body: [] },
  "GET /api/learn/unlocks": { body: { ...unlocksFirst, first_run: false } },
  "GET /api/data": { body: data },
  "GET /api/jobs": { body: [] },
  "GET /api/workers": { body: [] },
  "GET /api/catalog": { body: [] },
  "GET /api/blocks": { body: { blocks: [], errors: [] } },
};

// The service as the page sees it: either nothing answers, or basedpyright does.
class FakeSocket {
  static made: string[] = [];
  static mode: "refused" | "server" = "refused";
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(url: string) {
    FakeSocket.made.push(url);
    setTimeout(() => (FakeSocket.mode === "refused" ? this.onerror?.() : this.onopen?.()), 0);
  }
  send(data: string) {
    const m = JSON.parse(data) as { id?: number; method: string };
    const reply = (x: object) => this.onmessage?.({ data: JSON.stringify({ jsonrpc: "2.0", ...x }) });
    if (m.method === "initialize") reply({ id: m.id, result: { capabilities: {}, serverInfo: { name: "basedpyright", version: "1.40.2" } } });
    // the real server publishes after didOpen, which the real surface sends; the test surface has none
    if (m.method === "initialized")
      setTimeout(
        () =>
          reply({
            method: "textDocument/publishDiagnostics",
            params: { uri: URI, diagnostics: [{ range: { start: { line: 0, character: 0 }, end: { line: 0, character: 1 } }, severity: 1, message: "bad", source: "basedpyright" }] },
          }),
        50,
      );
  }
  close() {}
}

beforeEach(() => {
  localStorage.clear();
  FakeSocket.made = [];
  FakeSocket.mode = "refused";
  vi.stubGlobal("WebSocket", FakeSocket);
  installEventSource();
  act(() => {
    resetSettingsCache();
    resetLevelCache();
    resetLsp();
    setLevel("Tinker");
    setShowCommands(true);
  });
});
afterEach(() => {
  act(() => resetLsp());
  vi.unstubAllGlobals();
});

function editor() {
  mockApi(ROUTES);
  render(
    <Providers>
      <Editor path="models/my_lm.py" />
    </Providers>,
  );
}

describe("the editor without a language server (lsp.fallback)", () => {
  it("asks the address /api/settings gives and, when nothing answers, checks with ruff only and says so", async () => {
    editor();
    expect(await screen.findByText("no type checks: the language server is not running")).toBeTruthy();
    expect(FakeSocket.made).toEqual(["ws://localhost:8767/lsp"]);
    expect(screen.getByText("ruff: no problems")).toBeTruthy();
  });

  it("does not even try when the setting is off, and says that instead", async () => {
    act(() => setUseLsp(false));
    editor();
    expect(await screen.findByText("no type checks: the language server is turned off in Settings")).toBeTruthy();
    expect(FakeSocket.made).toEqual([]);
  });

  it("shows the server's problems beside ruff's when it answers", async () => {
    FakeSocket.mode = "server";
    editor();
    expect(await screen.findByText("basedpyright: 1 error")).toBeTruthy();
    expect(screen.getByText("ruff: no problems")).toBeTruthy();
    expect(screen.queryByText(/no type checks/)).toBeNull();
  });
});

describe("the setting", () => {
  const settings = () =>
    render(
      <Providers>
        <MemoryRouter>
          <Settings />
        </MemoryRouter>
      </Providers>,
    );

  it("warns and gives the start command when the service is not running", async () => {
    mockApi(ROUTES);
    settings();
    expect(await screen.findByText("The lsp service is not running, so the editor checks with ruff only.")).toBeTruthy();
    expect(screen.getByText("docker compose --profile editor-lsp up -d lsp")).toBeTruthy();
  });

  it("shows the service's name, version and read-only workspace when connected, and keeps the choice", async () => {
    FakeSocket.mode = "server";
    mockApi(ROUTES);
    settings();
    expect(await screen.findByText(/basedpyright 1\.40\.2/)).toBeTruthy();
    expect(screen.getByText("typeCheckingMode = \"standard\"")).toBeTruthy();
    const box = screen.getByRole("checkbox", { name: /Use the Python language server/ }) as HTMLInputElement;
    expect(box.checked).toBe(true);
    box.click();
    await waitFor(() => expect(localStorage.getItem("nanoscope.editor.lsp")).toBe("0"));
  });
});
