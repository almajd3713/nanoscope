import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { toProblem, unwrap } from "../api/problem";
import { useEvents } from "../hooks/useEvents";
import type { GClass } from "./flow";
import type { PaletteBlock } from "./Palette";

export type GraphDoc = { path: string; classes: GClass[]; etag: string };

// One edit as nanoscope/blocks/graph.py applies it (set_arg, replace_block, remove_arg,
// add_layer, remove_layer, set_pattern, fill_slot).
export type Edit = { op: string; class: string; [key: string]: unknown };

export function graphKey(path: string) {
  return ["graph", path] as const;
}

// The parsed graph of a workspace file. The file is read with `ast`; nothing is imported.
export function useGraph(path: string) {
  return useQuery({
    queryKey: graphKey(path),
    queryFn: () => unwrap(api.POST("/api/files/{path}/graph", { params: { path: { path } } })) as unknown as Promise<GraphDoc>,
  });
}

// Send edits through the API: it patches only the edited arguments of the file, with the ETag
// the graph was read at. A refusal (a locked block, a stale ETag) is the mutation's error,
// word for word, and nothing is written.
export function usePatchGraph(path: string, etag: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (edits: Edit[]) => {
      const { data, error, response } = await api.POST("/api/files/{path}/graph/patch", {
        params: { path: { path } },
        body: { edits },
        headers: { "If-Match": etag ?? "" },
      });
      if (error !== undefined || !response.ok) throw toProblem(error, response);
      return data as unknown as GraphDoc;
    },
    onSuccess: (doc) => {
      queryClient.setQueryData(graphKey(path), doc);
      // the file changed on disk: an open editor with a clean buffer reads it again
      void queryClient.invalidateQueries({ queryKey: ["file", path] });
    },
  });
}

// The palette's catalog (every block with its lock state and, for your own, its certification),
// kept current without a restart: a lesson check that unlocks a block, `nanoscope learn unlock`
// in a terminal, a block you register in a file, a certification that finished.
export function useBlocks() {
  const queryClient = useQueryClient();
  const refresh = () => void queryClient.invalidateQueries({ queryKey: ["blocks"] });
  useEvents("/api/learn/events", { events: ["unlocks", "progress"], onEvent: refresh });
  useEvents("/api/files/events", {
    events: ["change"],
    onEvent: (_type, data) => {
      if ((data as { path?: string }).path?.endsWith(".py")) refresh();
    },
  });
  return useQuery({
    queryKey: ["blocks"],
    queryFn: () => unwrap(api.GET("/api/blocks")) as unknown as Promise<{ blocks: PaletteBlock[]; policy: string }>,
  });
}
