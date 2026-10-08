// nanoscope's own findings, on the source lines they belong to (ruff's are in useLint).
// Four sources, each a plain function of what the server said: validation problems and locked
// uses, a describe job's shape error, and a failed equivalence check.
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import type { Marker } from "./types";

const SOURCE = "nanoscope";

function whole(line: number, message: string, code?: string): Marker {
  // the column range is the line's start: Monaco widens a marker with no extent to the word
  return { line, column: 1, endLine: line, endColumn: 1, message, severity: "error", source: SOURCE, code };
}

export type ValidationProblem = { code: string; field?: string | null; message: string; hint?: string | null };

// A locked use comes back as "line 12: <block> is locked until ...": the library's own text.
// Other problems (an unknown keyword, a wrong type) are about the run form, not a line.
export function fromValidation(problems: ValidationProblem[]): Marker[] {
  const out: Marker[] = [];
  for (const p of problems) {
    const at = /^line (\d+): ([\s\S]*)$/.exec(p.message);
    if (at) out.push(whole(Number(at[1]), p.hint ? `${at[2]} (${p.hint})` : at[2]!, p.code));
  }
  return out;
}

// `ShapeError: Block.attn: <reason> (/path/file.py:14)` from a failed describe job.
export function fromDescribeError(error: string | null | undefined, path: string): Marker[] {
  if (!error) return [];
  const m = /^ShapeError: ([\s\S]*) \((.+):(\d+)\)\s*$/.exec(error);
  if (!m) return [];
  const [, message, file, line] = m;
  const name = (p: string) => p.split("/").pop();
  if (name(file!) !== name(path)) return []; // the error is in another file
  return [whole(Number(line), message!, "shape")];
}

export type LessonCheck = { id: string; kind: string; class?: string };
export type CheckLine = { id: string; passed: boolean; reason: string };

// A failed `equivalent` check goes on the line of the class the lesson asks for.
export function fromChecks(results: CheckLine[], checks: LessonCheck[], source: string): Marker[] {
  const lines = source.split("\n");
  const out: Marker[] = [];
  for (const r of results) {
    if (r.passed) continue;
    const spec = checks.find((c) => c.id === r.id);
    if (!spec || spec.kind !== "equivalent" || !spec.class) continue;
    const at = lines.findIndex((l) => new RegExp(`^class\\s+${spec.class}\\b`).test(l));
    if (at >= 0) out.push(whole(at + 1, r.reason, "equivalent"));
  }
  return out;
}

type Job = { id: number; state: string; error?: string | null };

// After each save, ask a worker to trace the model (a describe job) and show a shape error on
// its line. The API never builds the model; the worker does.
export function useDescribeMarkers(path: string, etag: string | null, model: string | null): Marker[] {
  const start = useMutation({
    mutationFn: (ref: string) => unwrap(api.POST("/api/models/{ref}/describe", { params: { path: { ref } } })) as unknown as Promise<Job>,
  });
  const startedFor = useRef<string | null>(null);
  useEffect(() => {
    const key = `${model}|${etag}`;
    if (!model || !etag || startedFor.current === key) return;
    startedFor.current = key;
    start.mutate(model);
    // `start` is stable enough: one job per save
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, etag]);
  const id = start.data?.id;
  const job = useQuery({
    queryKey: ["jobs", "describe", id],
    enabled: id !== undefined,
    queryFn: () => unwrap(api.GET("/api/jobs/{job_id}", { params: { path: { job_id: id! } } })) as unknown as Promise<Job>,
    refetchInterval: (q) => (q.state.data && ["queued", "running", "cancelling"].includes(q.state.data.state) ? 1000 : false),
  });
  return job.data?.state === "failed" ? fromDescribeError(job.data.error, path) : [];
}
