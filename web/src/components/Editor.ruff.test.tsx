import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { Editor } from "./Editor";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

beforeEach(installEventSource);
afterEach(() => vi.unstubAllGlobals());

const UNUSED = { code: "F401", message: "`os` imported but unused", line: 1, column: 8, end_line: 1, end_column: 10, fixable: true };
const SYNTAX = { code: null, message: "SyntaxError: Expected an expression", line: 3, column: 5, end_line: 3, end_column: 6, fixable: false };

describe("Editor and ruff", () => {
  it("shows ruff's findings for the file as markers on their lines", async () => {
    mockApi({
      "GET /api/files/m.py": { body: { path: "m.py", content: "import os\n", etag: "e1" } },
      "POST /api/files/m.py/lint": { body: [UNUSED, SYNTAX] },
    });
    render(
      <Providers>
        <Editor path="m.py" />
      </Providers>,
    );
    const markers = await screen.findByRole("list", { name: "Markers" });
    await waitFor(() => expect(within(markers).getAllByRole("listitem")).toHaveLength(2));
    expect(within(markers).getByText("ruff 1:8 `os` imported but unused")).toBeTruthy();
    expect(within(markers).getByText("ruff 3:5 SyntaxError: Expected an expression")).toBeTruthy();
  });

  it("lints again after a save, not while typing", async () => {
    let etag = "e1";
    const seen = mockApi({
      "GET /api/files/m.py": { body: () => ({ path: "m.py", content: "import os\n", etag }) },
      "PUT /api/files/m.py": { body: (sent: unknown) => ({ path: "m.py", content: (sent as { content: string }).content, etag: (etag = "e2") }) },
      "POST /api/files/m.py/lint": { body: () => (etag === "e1" ? [UNUSED] : []) },
    });
    render(
      <Providers>
        <Editor path="m.py" />
      </Providers>,
    );
    const box = await screen.findByLabelText("Source of m.py");
    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1));
    await userEvent.type(box, "# edit");
    expect(seen.filter((s) => s.path.endsWith("/lint"))).toHaveLength(1); // typing lints nothing
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.queryAllByRole("listitem")).toHaveLength(0));
    expect(seen.filter((s) => s.path.endsWith("/lint"))).toHaveLength(2);
  });

  it("an editor without findings shows no markers", async () => {
    mockApi({
      "GET /api/files/m.py": { body: { path: "m.py", content: "x = 1\n", etag: "e1" } },
      "POST /api/files/m.py/lint": { body: [] },
    });
    render(
      <Providers>
        <Editor path="m.py" markers={[{ line: 1, column: 1, endLine: 1, endColumn: 2, message: "shape error", severity: "error", source: "nanoscope" }]} />
      </Providers>,
    );
    expect(await screen.findByText("nanoscope 1:1 shape error")).toBeTruthy(); // the page's own markers join ruff's
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });
});
