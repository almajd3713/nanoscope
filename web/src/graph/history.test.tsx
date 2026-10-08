import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "../../test/fetch";
import { EMPTY, observe, useHistory } from "./history";

afterEach(() => vi.unstubAllGlobals());

describe("observe", () => {
  const seen = (...texts: string[]) => texts.reduce(observe, EMPTY);

  it("keeps each new version of the file in order", () => {
    expect(seen("a", "b", "c")).toEqual({ texts: ["a", "b", "c"], at: 2 });
  });
  it("sees the same text twice as one version", () => {
    expect(seen("a", "a")).toEqual({ texts: ["a"], at: 0 });
  });
  it("a step back or forward moves along the list", () => {
    const back = observe(seen("a", "b", "c"), "b");
    expect(back).toEqual({ texts: ["a", "b", "c"], at: 1 });
    expect(observe(back, "c")).toEqual({ texts: ["a", "b", "c"], at: 2 });
  });
  it("an edit after stepping back ends the redo", () => {
    expect(observe(observe(seen("a", "b", "c"), "b"), "x")).toEqual({ texts: ["a", "b", "x"], at: 2 });
  });
});

describe("useHistory", () => {
  it("undo and redo send an earlier version through PUT with the ETag the file has now", async () => {
    let disk = { path: "m.py", content: "v1\n", etag: "e1" };
    const puts: { etag: string | null; content: string }[] = [];
    const seen = mockApi({
      "GET /api/files/m.py": { body: () => disk },
      "PUT /api/files/m.py": {
        body: (sent: unknown) => {
          puts.push({ etag: disk.etag, content: (sent as { content: string }).content });
          return (disk = { ...disk, content: (sent as { content: string }).content, etag: `e${puts.length + 1}` });
        },
      },
      "POST /api/files/m.py/graph": { body: { classes: [] } },
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
    const { result } = renderHook(() => useHistory("m.py"), { wrapper });
    await waitFor(() => expect(result.current.versions).toBe(1));
    expect(result.current.canUndo).toBe(false); // one version: nothing to undo

    // the file changes twice (a graph edit, then a save): two more versions
    for (const [n, content, etag] of [[2, "v2\n", "e5"], [3, "v3\n", "e6"]] as const) {
      disk = { path: "m.py", content, etag };
      await act(async () => {
        await client.refetchQueries({ queryKey: ["file", "m.py"] });
      });
      await waitFor(() => expect(result.current.versions).toBe(n));
    }
    await waitFor(() => expect(result.current.canUndo).toBe(true));
    expect(result.current.canRedo).toBe(false);

    act(() => result.current.undo());
    await waitFor(() => expect(puts).toEqual([{ etag: "e6", content: "v2\n" }]));
    await waitFor(() => expect(result.current.canRedo).toBe(true));
    expect(result.current.canUndo).toBe(true); // v1 is still behind

    act(() => result.current.redo());
    await waitFor(() => expect(puts).toHaveLength(2));
    expect(puts[1]).toEqual({ etag: "e2", content: "v3\n" }); // sent with the ETag the undo returned
    await waitFor(() => expect(result.current.canRedo).toBe(false));
    expect(seen.filter((s) => s.method === "PUT")).toHaveLength(2);
  });

  it("an edit after an undo ends the redo", async () => {
    let disk = { path: "m.py", content: "v1\n", etag: "e1" };
    mockApi({
      "GET /api/files/m.py": { body: () => disk },
      "PUT /api/files/m.py": { body: (sent: unknown) => (disk = { ...disk, content: (sent as { content: string }).content, etag: "e9" }) },
      "POST /api/files/m.py/graph": { body: { classes: [] } },
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
    const { result } = renderHook(() => useHistory("m.py"), { wrapper });
    await waitFor(() => expect(result.current.versions).toBe(1));
    disk = { path: "m.py", content: "v2\n", etag: "e2" };
    await act(async () => {
      await client.refetchQueries({ queryKey: ["file", "m.py"] });
    });
    await waitFor(() => expect(result.current.versions).toBe(2));
    act(() => result.current.undo());
    await waitFor(() => expect(result.current.canRedo).toBe(true));
    disk = { path: "m.py", content: "other\n", etag: "e3" };
    await act(async () => {
      await client.refetchQueries({ queryKey: ["file", "m.py"] });
    });
    await waitFor(() => expect(result.current.canRedo).toBe(false));
    expect(result.current.canUndo).toBe(true);
  });
});
