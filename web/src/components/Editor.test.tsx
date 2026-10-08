import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { Editor } from "./Editor";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

const FILE = { path: "models/my_lm.py", content: "print('hi')\n", etag: "e1" };

afterEach(() => vi.unstubAllGlobals());

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({ "GET /api/files/models/my_lm.py": { body: FILE }, ...routes });
  render(
    <Providers>
      <Editor path="models/my_lm.py" />
    </Providers>,
  );
  return seen;
}

describe("Editor", () => {
  it("opens the file and shows it saved", async () => {
    open();
    expect(((await screen.findByLabelText("Source of models/my_lm.py")) as HTMLTextAreaElement).value).toBe("print('hi')\n");
    expect(screen.getByText("saved")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Save" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("marks the buffer unsaved as soon as it differs from the file", async () => {
    open();
    const box = await screen.findByLabelText("Source of models/my_lm.py");
    await userEvent.type(box, "x");
    expect(screen.getByText("unsaved")).toBeTruthy();
    await userEvent.clear(box);
    await userEvent.type(box, "print('hi')\n");
    expect(screen.getByText("saved")).toBeTruthy(); // back to what is on disk
  });

  it("saves with the ETag it read, on the button and on Ctrl-S", async () => {
    const puts: { content: string }[] = [];
    let etag = "e1";
    const seen = open({
      "PUT /api/files/models/my_lm.py": {
        body: (sent: unknown) => {
          puts.push(sent as { content: string });
          etag = "e" + (puts.length + 1);
          return { ...FILE, content: (sent as { content: string }).content, etag };
        },
      },
    });
    const box = await screen.findByLabelText("Source of models/my_lm.py");
    await userEvent.type(box, "# a");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.getByText("saved")).toBeTruthy());
    expect(puts).toEqual([{ content: "print('hi')\n# a" }]);
    expect(seen.filter((s) => s.method === "PUT")).toHaveLength(1);
    // Ctrl-S sends the new ETag
    await userEvent.type(box, "b");
    await userEvent.keyboard("{Control>}s{/Control}");
    await waitFor(() => expect(puts).toHaveLength(2));
    expect(puts[1]).toEqual({ content: "print('hi')\n# ab" });
  });

  it("shows a refused save word for word and keeps the text", async () => {
    open({
      "PUT /api/files/models/my_lm.py": {
        status: 422,
        problem: true,
        body: { title: "Unprocessable", status: 422, detail: "the disk is full" },
      },
    });
    const box = await screen.findByLabelText("Source of models/my_lm.py");
    await userEvent.type(box, "z");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("the disk is full")).toBeTruthy();
    expect((box as HTMLTextAreaElement).value).toContain("z"); // the text is still there
    expect(screen.getByText("unsaved")).toBeTruthy();
  });

  it("says so when the file is missing", async () => {
    mockApi({});
    render(
      <Providers>
        <Editor path="models/my_lm.py" />
      </Providers>,
    );
    expect(await screen.findByRole("alert")).toBeTruthy();
  });
});
