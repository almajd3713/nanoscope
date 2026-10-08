import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
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
  const queryClient = useQueryClient();
  // anything that changes the file (another editor, git, a save) parses it again
  useEvents("/api/files/events", {
    events: ["change"],
    onEvent: (_type, data) => {
      if ((data as { path?: string }).path === path) void queryClient.invalidateQueries({ queryKey: graphKey(path) });
    },
  });
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
  // A certification is a job: when one finishes, the block's badge changed on disk.
  const certify = useQuery({
    queryKey: ["jobs", "certify"],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { kind: "certify", limit: 50 } } })) as unknown as Promise<{ id: number; state: string }[]>,
    refetchInterval: (q) => ((q.state.data ?? []).some((j) => ["queued", "running"].includes(j.state)) ? 1000 : false),
  });
  const finished = (certify.data ?? []).filter((j) => !["queued", "running"].includes(j.state)).map((j) => j.id).join(",");
  const seen = useRef<string | null>(null);
  useEffect(() => {
    if (seen.current !== null && seen.current !== finished) void queryClient.invalidateQueries({ queryKey: ["blocks"] });
    seen.current = finished;
  }, [finished, queryClient]);
  return useQuery({
    queryKey: ["blocks"],
    queryFn: () => unwrap(api.GET("/api/blocks")) as unknown as Promise<{ blocks: PaletteBlock[]; policy: string }>,
  });
}

export type LayerStat = { name: string; grad_norm: number };
type StatsLine = { step: number; blocks: LayerStat[] };

// The newest block statistics of the newest run of this model class, kept live while that run
// writes them (blockstats events). Null when no run of it recorded any (run(..., block_stats=True)).
export function useLayerStats(className: string | null) {
  const [live, setLive] = useState<{ ref: string; line: StatsLine } | null>(null);
  const runs = useQuery({
    queryKey: ["runs", "of-model", className],
    enabled: className !== null,
    queryFn: () => unwrap(api.GET("/api/runs")) as unknown as Promise<{ ref: string; model?: string | null; state: string }[]>,
  });
  const mine = (runs.data ?? []).filter((r) => r.model === className);
  const run = mine.at(-1)?.ref ?? null; // the library lists oldest first
  const stats = useQuery({
    queryKey: ["blockstats", run],
    enabled: run !== null,
    queryFn: () => unwrap(api.GET("/api/runs/{ref}/blockstats", { params: { path: { ref: run! } } })) as unknown as Promise<StatsLine[]>,
  });
  const live_ = run ? `/api/runs/${run}/events` : null;
  useEvents(live_, {
    events: ["blockstats"],
    onEvent: (_type, data) => run && setLive({ ref: run, line: data as StatsLine }),
  });
  const newest = live && live.ref === run ? live.line : (stats.data ?? []).at(-1);
  return { run, step: newest?.step ?? null, blocks: newest?.blocks ?? null };
}

type DescribeJob = { state: string; payload: Record<string, unknown>; result: { modules?: import("./trace").TraceRow[] } | null };

// The modules of the newest finished describe job for this model class (the trace its shapes,
// parameters and FLOPs come from), or null when it has not been traced.
export function useTrace(path: string, className: string | null) {
  const jobs = useQuery({
    queryKey: ["jobs", "describe"],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { kind: "describe", limit: 200 } } })) as unknown as Promise<DescribeJob[]>,
    refetchInterval: (q) => ((q.state.data ?? []).some((j) => ["queued", "running"].includes(j.state)) ? 1000 : false),
  });
  if (!className) return null;
  const ref = `${path}:${className}`;
  const mine = (jobs.data ?? []).filter((j) => j.state === "done" && typeof j.payload["model"] === "string" && (j.payload["model"] as string).endsWith(ref));
  return mine.at(-1)?.result?.modules ?? null;
}
