import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { toProblem, unwrap } from "../api/problem";

export type FileDoc = { path: string; content: string; etag: string };

// The server refused a save because the file changed underneath (409), or the file exists and
// no ETag was sent (428). `diff` is the server's unified diff, on disk against yours.
export type Conflict = { diff: string; currentEtag: string };

export function fileKey(path: string) {
  return ["file", path] as const;
}

// One file open for editing: the server's copy (with its ETag), the unsaved text on top of it,
// and the save that sends the ETag back. The text lives here and on the server, nowhere else.
export function useFileBuffer(path: string) {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<{ path: string; text: string } | null>(null);
  const [conflict, setConflict] = useState<Conflict | null>(null);

  const file = useQuery({
    queryKey: fileKey(path),
    queryFn: () => unwrap(api.GET("/api/files/{path}", { params: { path: { path } } })) as unknown as Promise<FileDoc>,
  });
  const base = file.data;
  const mine = draft && draft.path === path ? draft.text : null;
  const text = mine ?? base?.content ?? "";
  const dirty = base !== undefined && mine !== null && mine !== base.content;

  const write = useMutation({
    mutationFn: async (args: { text: string; etag: string }) => {
      const { data, error, response } = await api.PUT("/api/files/{path}", {
        params: { path: { path } },
        body: { content: args.text },
        headers: { "If-Match": args.etag },
      });
      if (response.status === 409 || response.status === 428) {
        const doc = error as { current_etag?: string; diff?: string };
        return { conflict: { diff: doc.diff ?? "", currentEtag: doc.current_etag ?? "" } as Conflict };
      }
      if (error !== undefined || !response.ok) {
        throw toProblem(error, response);
      }
      return { saved: data as unknown as FileDoc };
    },
    onSuccess: (result) => {
      if ("saved" in result) {
        queryClient.setQueryData(fileKey(path), result.saved);
        setDraft(null);
        setConflict(null);
        // the graph, the lint and the describe all read this file
        void queryClient.invalidateQueries({ queryKey: ["graph", path] });
      } else {
        setConflict(result.conflict);
      }
    },
  });

  return {
    loading: file.isPending,
    error: file.error,
    path,
    etag: base?.etag ?? null,
    text,
    dirty,
    edit: (next: string) => setDraft({ path, text: next }),
    save: () => {
      if (!dirty || !base || write.isPending) return;
      write.mutate({ text, etag: base.etag });
    },
    saving: write.isPending,
    saveError: write.error,
    conflict,
    // keep my text: save it over what is on disk, with the ETag the server just gave
    keepMine: () => {
      if (conflict) write.mutate({ text, etag: conflict.currentEtag });
    },
    // take the server's text: drop mine and read the file again
    takeTheirs: async () => {
      setDraft(null);
      setConflict(null);
      await queryClient.invalidateQueries({ queryKey: fileKey(path) });
    },
  };
}
