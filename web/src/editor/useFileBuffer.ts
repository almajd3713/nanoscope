import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { toProblem, unwrap } from "../api/problem";
import { useEvents } from "../hooks/useEvents";

export type FileDoc = { path: string; content: string; etag: string };

// The server refused a save because the file changed underneath (409), or the file exists and
// no ETag was sent (428). `diff` is the server's unified diff, on disk against yours.
export type Conflict = { diff: string; currentEtag: string };

// Something else changed the file (another editor, `git checkout`, a graph edit) while there are
// unsaved edits here. A clean buffer just reloads; this is only for a dirty one.
export type External = { kind: "added" | "modified" | "deleted"; etag: string | null };

export function fileKey(path: string) {
  return ["file", path] as const;
}

// The server's copy of a file with its ETag.
export function useFileDoc(path: string) {
  return useQuery({
    queryKey: fileKey(path),
    queryFn: () => unwrap(api.GET("/api/files/{path}", { params: { path: { path } } })) as unknown as Promise<FileDoc>,
  });
}

// One file open for editing: the server's copy (with its ETag), the unsaved text on top of it,
// and the save that sends the ETag back. The text lives here and on the server, nowhere else.
export function useFileBuffer(path: string) {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<{ path: string; text: string } | null>(null);
  const [conflict, setConflict] = useState<Conflict | null>(null);
  const [external, setExternal] = useState<{ path: string; change: External } | null>(null);
  const known = useRef<{ etag: string | null; dirty: boolean; saving: boolean }>({ etag: null, dirty: false, saving: false });

  const file = useFileDoc(path);
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

  const etag = base?.etag ?? null;
  const saving = write.isPending;
  useEffect(() => {
    known.current = { etag, dirty, saving };
  });

  // The workspace watcher: a change to this file that is not our own save.
  useEvents("/api/files/events", {
    events: ["change"],
    onEvent: (_type, data) => {
      const change = data as { path: string; kind: External["kind"]; etag: string | null };
      const now = known.current;
      if (change.path !== path || change.etag === now.etag || now.saving) return;
      if (now.dirty) setExternal({ path, change: { kind: change.kind, etag: change.etag } });
      else void queryClient.invalidateQueries({ queryKey: fileKey(path) });
    },
  });

  return {
    external: external && external.path === path && dirty ? external.change : null,
    // reload from disk: my unsaved text goes
    reload: async () => {
      setDraft(null);
      setExternal(null);
      setConflict(null);
      await queryClient.invalidateQueries({ queryKey: fileKey(path) });
    },
    // keep editing: the next save is checked against the disk and may conflict (and show the diff)
    keepEditing: () => setExternal(null),
    loading: file.isPending,
    error: file.error,
    path,
    etag,
    text,
    dirty,
    edit: (next: string) => setDraft({ path, text: next }),
    save: () => {
      if (!dirty || !base || write.isPending) return;
      write.mutate({ text, etag: base.etag });
    },
    saving,
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
