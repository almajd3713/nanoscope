import { useQueries, useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { followJob } from "../studies/useStudyChecks";
import type { InspectReport } from "./report";

export type Checkpoint = { name: string; step: number | null; bytes: number; archived: boolean; latest: boolean };

// The steps a run has a checkpoint for: the newest, regular ones not yet pruned, and archived
// ones (checkpoint_steps), which are kept for good.
export function useCheckpoints(ref: string) {
  return useQuery({
    queryKey: ["run", ref, "checkpoints"],
    queryFn: () => unwrap(api.GET("/api/runs/{ref}/checkpoints", { params: { path: { ref } } })) as Promise<Checkpoint[]>,
  });
}

export type Step = { step: number; archived: boolean; latest: boolean };

export function stepsOf(checkpoints: Checkpoint[]): Step[] {
  const byStep = new Map<number, Step>();
  for (const c of checkpoints) {
    if (c.step === null) continue;
    const prev = byStep.get(c.step);
    byStep.set(c.step, {
      step: c.step,
      archived: (prev?.archived ?? false) || c.archived,
      latest: (prev?.latest ?? false) || c.latest,
    });
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step);
}

// One inspect job: the worker loads the run's model, so this queues it and follows it. The
// prompt is checked by the worker (it holds the tokenizer), so its refusal arrives as the job's
// error word for word.
async function runInspect(ref: string, prompt: string, step: number | null): Promise<InspectReport> {
  const job = await unwrap(
    api.POST("/api/runs/{ref}/inspect", { params: { path: { ref } }, body: { prompt, top_k: 5, ...(step === null ? {} : { step }) } }),
  );
  return followJob<InspectReport>((job as unknown as { id: number }).id);
}

const options = (ref: string, prompt: string, step: number | null) => ({
  queryKey: ["inspect", ref, prompt, step],
  queryFn: () => runInspect(ref, prompt, step),
  staleTime: Infinity, // a checkpoint and a prompt give the same report
  retry: false,
});

export function useInspect(ref: string, prompt: string | null, step: number | null) {
  return useQuery({ ...options(ref, prompt ?? "", step), enabled: prompt !== null });
}

// One job per kept step, same prompt.
export function useInspectAcross(ref: string, prompt: string | null, steps: number[], enabled: boolean) {
  return useQueries({
    queries: steps.map((step) => ({ ...options(ref, prompt ?? "", step), enabled: enabled && prompt !== null })),
  });
}
