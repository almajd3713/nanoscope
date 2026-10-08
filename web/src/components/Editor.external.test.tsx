import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installEventSource, lastSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { Editor } from "./Editor";

vi.mock("../editor/CodeSurface", async () => await import("../../test/fakeSurface"));

let disk = { path: "m.py", content: "one\n", etag: "e1" };

beforeEach(() => {
  disk = { path: "m.py", content: "one\n", etag: "e1" };
  installEventSource();
});
afterEach(() => vi.unstubAllGlobals());

async function open() {
  const seen = mockApi({
    "GET /api/files/m.py": { body: () => disk },
    "PUT /api/files/m.py": { body: (sent: unknown) => (disk = { ...disk, content: (sent as { content: string }).content, etag: "e2" }) },
  });
  render(
    <Providers>
      <Editor path="m.py" />
    </Providers>,
  );
  const box = (await screen.findByLabelText("Source of m.py")) as HTMLTextAreaElement;
  act(() => lastSource().open());
  return { seen, box };
}

const edit = (change: Partial<{ path: string; kind: string; etag: string | null }>) =>
  act(() => lastSource().emit("change", { path: "m.py", kind: "modified", etag: "e7", ...change }));

describe("Editor and the workspace watcher", () => {
  it("listens to the file events stream", async () => {
    await open();
    expect(lastSource().url).toBe("/api/files/events");
  });

  it("reloads a clean buffer when the file changes on disk", async () => {
    const { box } = await open();
    disk = { path: "m.py", content: "two\n", etag: "e7" };
    edit({});
    await waitFor(() => expect(box.value).toBe("two\n"));
    expect(screen.queryByText("changed on disk")).toBeNull();
  });

  it("asks before touching a buffer with unsaved edits", async () => {
    const { box } = await open();
    await userEvent.type(box, "x");
    disk = { path: "m.py", content: "two\n", etag: "e7" };
    edit({});
    expect(await screen.findByText("changed on disk")).toBeTruthy();
    expect(box.value).toBe("one\nx"); // untouched
    await userEvent.click(screen.getByRole("button", { name: "Reload from disk" }));
    await waitFor(() => expect(box.value).toBe("two\n"));
    expect(screen.queryByText("changed on disk")).toBeNull();
    expect(screen.getByText("saved")).toBeTruthy();
  });

  it("keeps my edits when I say so, and the save then goes through the conflict check", async () => {
    const { box } = await open();
    await userEvent.type(box, "x");
    edit({ kind: "deleted", etag: null });
    expect(await screen.findByText("deleted on disk")).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Keep my edits" }));
    expect(screen.queryByText("deleted on disk")).toBeNull();
    expect(box.value).toBe("one\nx");
  });

  it("ignores its own save and other files", async () => {
    const { box, seen } = await open();
    edit({ path: "other.py" });
    edit({ etag: "e1" }); // the etag we already hold
    await userEvent.type(box, "y");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.getByText("saved")).toBeTruthy());
    edit({ etag: "e2" }); // the watcher reporting our own write
    expect(screen.queryByText("changed on disk")).toBeNull();
    expect(seen.filter((s) => s.method === "GET" && s.path === "/api/files/m.py")).toHaveLength(1); // never re-read
  });
});
