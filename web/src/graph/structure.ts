// The structure of a model class the graph cannot draw (plain PyTorch): what its code builds,
// from a structure job a worker runs after each save (structure.v1). The API never builds it.
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";

export type StructureRow = {
  path: string;
  type: string;
  block: string | null;
  depth: number;
  params: number;
  input_shapes: number[][] | null;
  output_shapes: number[][] | null;
  line: number | null;
};

export type StructureReport = {
  model: string;
  line: number | null;
  args: Record<string, unknown>;
  inputs: number[][] | null;
  built: boolean;
  traced: boolean;
  error: { stage: "build" | "forward"; type: string; message: string; module: string | null; line: number | null } | null;
  params: number;
  modules: StructureRow[];
};

export type Hints = { args?: Record<string, unknown>; inputs?: unknown[] };
type LessonCheck = { kind: string; class?: string; args?: unknown; inputs?: unknown };

// A lesson's checks already build its class: the same arguments and the first set of inputs.
export function hintsFromChecks(checks: LessonCheck[] | undefined, className: string): Hints {
  const mine = (checks ?? []).filter((c) => c.class === className);
  const args = mine.find((c) => c.args && typeof c.args === "object")?.args as Record<string, unknown> | undefined;
  const inputs = mine.find((c) => c.kind === "equivalent" && Array.isArray(c.inputs))?.inputs as unknown[] | undefined;
  return { ...(args ? { args } : {}), ...(inputs ? { inputs } : {}) };
}

type Job = { id: number; state: string; error?: string | null; result?: StructureReport | null };

export type Structure = { state: "idle" | "waiting" | "done" | "failed"; report: StructureReport | null; error: string | null };

// One structure job per saved version of the file (its ETag) and class.
export function useStructure(path: string, className: string | null, etag: string | null, hints: Hints): Structure {
  const start = useMutation({
    mutationFn: (ref: string) => unwrap(api.POST("/api/models/{ref}/structure", { params: { path: { ref } }, body: { args: hints.args ?? {}, inputs: hints.inputs ?? null } })) as unknown as Promise<Job>,
  });
  const startedFor = useRef<string | null>(null);
  const hintKey = JSON.stringify(hints);
  useEffect(() => {
    const key = `${path}:${className}|${etag}|${hintKey}`;
    if (!className || !etag || startedFor.current === key) return;
    startedFor.current = key;
    start.mutate(`${path}:${className}`);
    // `start` is stable enough: one job per save
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, className, etag, hintKey]);
  const id = start.data?.id;
  const job = useQuery({
    queryKey: ["jobs", "structure", id],
    enabled: id !== undefined,
    queryFn: () => unwrap(api.GET("/api/jobs/{job_id}", { params: { path: { job_id: id! } } })) as unknown as Promise<Job>,
    refetchInterval: (q) => (q.state.data && ["queued", "running", "cancelling"].includes(q.state.data.state) ? 1000 : false),
  });
  if (start.error) return { state: "failed", report: null, error: start.error.message };
  if (!className || id === undefined) return { state: className ? "waiting" : "idle", report: null, error: null };
  const data = job.data;
  if (!data || ["queued", "running", "cancelling"].includes(data.state)) return { state: "waiting", report: null, error: null };
  if (data.state === "done" && data.result) return { state: "done", report: data.result, error: null };
  return { state: "failed", report: null, error: data.error ?? `the structure job ended ${data.state}` };
}

export type TreeNode = StructureRow & { name: string; repeat: number; children: TreeNode[] };

const LISTS = new Set(["ModuleList", "Sequential"]);

// The rows (depth-first, as named_modules lists them) as a tree. The items of a ModuleList or
// Sequential that repeat the one before them fold into it ("x4"), as the graph folds a stack.
export function structureTree(rows: StructureRow[]): TreeNode | null {
  const byPath = new Map<string, TreeNode>();
  let root: TreeNode | null = null;
  for (const row of rows) {
    const node: TreeNode = { ...row, name: row.path.split(".").pop() || row.path, repeat: 1, children: [] };
    byPath.set(row.path, node);
    if (!row.path) {
      root = node;
      continue;
    }
    const parentPath = row.path.includes(".") ? row.path.slice(0, row.path.lastIndexOf(".")) : "";
    const parent = byPath.get(parentPath);
    if (!parent) continue; // inside a folded item: the first one stands for it
    const previous = parent.children.at(-1);
    if (LISTS.has(parent.type) && previous && previous.type === row.type && previous.params === row.params) {
      previous.repeat += 1;
      byPath.delete(row.path);
      continue;
    }
    parent.children.push(node);
  }
  return root;
}

// "MyBigram(vocab_size=50, d_model=32)"
export function builtAs(report: StructureReport): string {
  const args = Object.entries(report.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`);
  return `${report.model}(${args.join(", ")})`;
}
