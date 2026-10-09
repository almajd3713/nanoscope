import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import blocks from "../../test/fixtures/blocks-guided.json";
import authoring from "../../test/fixtures/authoring-list.json";
import file from "../../test/fixtures/workspace-file.json";
import files from "../../test/fixtures/workspace-files.json";
import models from "../../test/fixtures/workspace-models.json";
import studies from "../../test/fixtures/workspace-studies.json";
import { installEventSource } from "../../test/eventsource";
import { mockApi } from "../../test/fetch";
import { Providers } from "../app/providers";
import { resetSettingsCache, setShowCommands } from "../app/settings";
import { Workspace } from "./Workspace";

const git = { repo: true, root: "/w", branch: "master", head: "a7662fd1234", clean: false, changed: ["models/my_lm.py"], identity: true, path: null, path_committed: null };

function open(routes: Parameters<typeof mockApi>[0] = {}) {
  const seen = mockApi({
    "GET /api/files": { body: files },
    "GET /api/models": { body: models },
    "GET /api/blocks": { body: blocks },
    "GET /api/studies": { body: studies },
    "GET /api/authoring": { body: authoring },
    "GET /api/git/status": { body: git },
    "GET /api/files/models/my_lm.py": { body: file },
    ...routes,
  });
  render(
    <Providers>
      <MemoryRouter initialEntries={["/workspace"]}>
        <Routes>
          <Route path="/workspace" element={<Workspace />} />
        </Routes>
      </MemoryRouter>
    </Providers>,
  );
  return seen;
}

beforeEach(() => {
  localStorage.clear();
  installEventSource();
  act(() => {
    resetSettingsCache();
    setShowCommands(true);
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Workspace", () => {
  it("lists the files with what nanoscope finds in them and the git state", async () => {
    open();
    await screen.findByRole("heading", { name: "Workspace" });
    const tree = await screen.findByRole("region", { name: "Files" });
    expect(await within(tree).findByRole("link", { name: "MyLM" })).toBeTruthy();
    expect(within(tree).getByRole("link", { name: "my-course/01-thing" }).getAttribute("href")).toBe("/authoring/curricula/my-course/01-thing");
    expect(within(tree).getAllByText("changed").length).toBe(1);
    expect(screen.getByText("1 file changed")).toBeTruthy();
    expect(screen.getByText("a7662fd")).toBeTruthy();
  });

  it("folds a folder and remembers it", async () => {
    open();
    const tree = await screen.findByRole("region", { name: "Files" });
    await within(tree).findByText("my_lm.py");
    await userEvent.click(within(tree).getByRole("button", { name: "models" }));
    expect(within(tree).queryByText("my_lm.py")).toBeNull();
    expect(JSON.parse(localStorage.getItem("nanoscope.workspace.closed") ?? "[]")).toEqual(["models"]);
  });

  it("shows a file read only, with a way to open it on the model page", async () => {
    open();
    const tree = await screen.findByRole("region", { name: "Files" });
    await userEvent.click(await within(tree).findByRole("button", { name: "my_lm.py" }));
    const panel = await screen.findByRole("complementary", { name: "Selected file" });
    expect(within(panel).getByText("models/my_lm.py")).toBeTruthy();
    expect(await within(panel).findByLabelText("models/my_lm.py (first lines)")).toBeTruthy();
    expect(within(panel).getByRole("link", { name: "Open on the model page" }).getAttribute("href")).toBe("/model/models/my_lm.py?class=MyLM");
  });

  it("says what is missing when the workspace is empty", async () => {
    open({ "GET /api/files": { body: [] } });
    expect(await screen.findByText("The workspace is empty")).toBeTruthy();
  });
});
