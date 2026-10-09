import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GitStatus, recordReady, type Git } from "./GitStatus";

const base: Git = { repo: true, root: "/ws", branch: "main", head: "a99fe9aa0e70b58a5d011120093e0935157b4f35", clean: true, changed: [], identity: true, path: "studies/m1.toml", path_committed: true };
const PATH = "studies/m1.toml";

describe("GitStatus", () => {
  it("is clean and committed: record may run", () => {
    render(<GitStatus git={base} path={PATH} />);
    expect(screen.getByText("clean")).toBeTruthy();
    expect(screen.getByText("a99fe9a")).toBeTruthy();
    expect(screen.getByText(/is committed and unchanged/)).toBeTruthy();
    expect(recordReady(base)).toBe(true);
  });

  it("lists every changed file, and says when the spec itself is the only one", () => {
    const dirty = { ...base, clean: false, changed: ["studies/m1.toml", "notes/ideas.md"], path_committed: false };
    render(<GitStatus git={dirty} path={PATH} />);
    expect(screen.getByText("2 uncommitted changes")).toBeTruthy();
    expect(screen.getByText("notes/ideas.md")).toBeTruthy();
    expect(screen.getByText(/read-only; nanoscope never stages/)).toBeTruthy();
    expect(recordReady(dirty)).toBe(false);
  });

  it("points at the spec when it is the only change", () => {
    render(<GitStatus git={{ ...base, clean: false, changed: [PATH], path_committed: false }} path={PATH} />);
    expect(screen.getByText("1 uncommitted change")).toBeTruthy();
    expect(screen.getByText(/the spec itself: commit it to run in record mode/)).toBeTruthy();
  });

  it("is not ready outside a repository or when the spec was never committed", () => {
    render(<GitStatus git={{ ...base, repo: false }} path={PATH} />);
    expect(screen.getByText("not a git repository")).toBeTruthy();
    expect(recordReady({ ...base, repo: false })).toBe(false);
    expect(recordReady({ ...base, path_committed: false })).toBe(false);
    expect(recordReady(undefined)).toBe(false);
  });
});
