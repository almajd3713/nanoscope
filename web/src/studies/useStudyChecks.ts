import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";

export type Problem = { code: string; field: string | null; message: string; hint: string | null };
export type Estimate = { text: string; complete: boolean; wall_seconds: number | null; total_seconds: number | null; runs: number };
export type Validation = { ok: boolean; problems: Problem[]; estimate: Estimate | null; toml: string | null };
export type SizeRow = {
  name: string;
  kwargs: Record<string, unknown>;
  non_embedding_params: number;
  flops_per_token: number;
  delta: number;
  within: boolean;
};
export type SizeTable = { reference: string; tolerance: number; variants: SizeRow[] };

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

// The library's verdict on the spec as it is typed (POST /api/validate/study), plus the estimate
// and the file Save would write. Kept while the next answer loads.
export function useStudyValidation(spec: Record<string, unknown>, devices: string[], perDevice: number) {
  const body = useDebounced(JSON.stringify({ spec, devices, workers_per_device: perDevice }), 300);
  return useQuery({
    queryKey: ["validate", "study", body],
    queryFn: () => unwrap(api.POST("/api/validate/study", { body: JSON.parse(body) })) as unknown as Promise<Validation>,
    placeholderData: keepPreviousData,
  });
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Parameter counts and FLOPs per variant. A worker builds the models (the API never runs the
// learner's code), so this queues a sizes job and follows it until it is done.
export function useSizes(spec: Record<string, unknown>, enabled: boolean) {
  const body = useDebounced(JSON.stringify({ spec }), 500);
  return useQuery({
    queryKey: ["sizes", body],
    enabled,
    queryFn: async (): Promise<SizeTable> => {
      const queued = await unwrap(api.POST("/api/studies/sizes", { body: JSON.parse(body) }));
      const id = (queued as unknown as { id: number }).id;
      for (;;) {
        const job = (await unwrap(api.GET("/api/jobs/{job_id}", { params: { path: { job_id: id } } }))) as unknown as {
          state: string;
          result: SizeTable | null;
          error: string | null;
        };
        if (job.state === "done" && job.result) return job.result;
        if (["failed", "cancelled"].includes(job.state)) throw new Error(job.error ?? `the sizes job ${job.state}`);
        await sleep(400);
      }
    },
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: false,
  });
}
