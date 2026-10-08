import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useReducer } from "react";
import { api } from "../api/client";
import { toProblem } from "../api/problem";
import { fileKey, useFileDoc } from "../editor/useFileBuffer";

// Undo and redo are earlier versions of the file, sent back to the server (PUT with the ETag it
// holds now). The list below is only a memory of what the file has been; the file on the server
// is always the state, so a reload shows the same model.
export type Versions = { texts: string[]; at: number };

export const EMPTY: Versions = { texts: [], at: -1 };

// A new text for the file: the one we asked for (a step back or forward), or a new version
// (a graph edit, a save, someone else's edit), which ends any redo.
export function observe(state: Versions, text: string): Versions {
  if (state.at >= 0 && state.texts[state.at] === text) return state;
  if (state.at > 0 && state.texts[state.at - 1] === text) return { ...state, at: state.at - 1 };
  if (state.at >= 0 && state.at < state.texts.length - 1 && state.texts[state.at + 1] === text) return { ...state, at: state.at + 1 };
  const texts = [...state.texts.slice(0, state.at + 1), text];
  return { texts, at: texts.length - 1 };
}

export const canUndo = (v: Versions) => v.at > 0;
export const canRedo = (v: Versions) => v.at >= 0 && v.at < v.texts.length - 1;

type Action = { type: "seen"; text: string } | { type: "reset" };

function reduce(state: Versions, action: Action): Versions {
  return action.type === "reset" ? EMPTY : observe(state, action.text);
}

export function useHistory(path: string) {
  const queryClient = useQueryClient();
  const file = useFileDoc(path);
  const [versions, dispatch] = useReducer(reduce, EMPTY);

  const content = file.data?.content;
  useEffect(() => {
    if (content !== undefined) dispatch({ type: "seen", text: content });
  }, [content]);
  useEffect(() => () => dispatch({ type: "reset" }), [path]);

  const send = useMutation({
    mutationFn: async (text: string) => {
      const etag = file.data?.etag;
      if (!etag) throw new Error("the file has not been read yet");
      const { data, error, response } = await api.PUT("/api/files/{path}", {
        params: { path: { path } },
        body: { content: text },
        headers: { "If-Match": etag },
      });
      if (error !== undefined || !response.ok) throw toProblem(error, response);
      return data as unknown as { path: string; content: string; etag: string };
    },
    onSuccess: (doc) => {
      queryClient.setQueryData(fileKey(path), doc);
      void queryClient.invalidateQueries({ queryKey: ["graph", path] });
    },
  });

  return {
    // how many versions of the file this page has seen, and which one is on disk now
    versions: versions.texts.length,
    position: versions.at + 1,
    canUndo: canUndo(versions) && !send.isPending,
    canRedo: canRedo(versions) && !send.isPending,
    undo: () => {
      const text = versions.texts[versions.at - 1];
      if (canUndo(versions) && text !== undefined) send.mutate(text);
    },
    redo: () => {
      const text = versions.texts[versions.at + 1];
      if (canRedo(versions) && text !== undefined) send.mutate(text);
    },
    error: send.error,
  };
}
