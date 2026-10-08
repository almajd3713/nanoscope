import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api } from "./api/client";
import { unwrap } from "./api/problem";

type CheckJobResult = {
  passed: boolean;
  check_id: string;
  lesson: string;
  checks: { id: string; passed: boolean; reason: string }[];
};

// The latest check job for a lesson, kept fresh while it is queued or running. The job list is
// the history, so the last result is there after a reload too.
export function useLessonCheck(lessonId: string) {
  const queryClient = useQueryClient();
  const jobs = useQuery({
    queryKey: ["jobs", "check"],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { kind: "check", limit: 50 } } })),
    refetchInterval: (q) => {
      const mine = (q.state.data ?? []).filter((j) => j.payload["lesson"] === lessonId).at(-1);
      return mine && (mine.state === "queued" || mine.state === "running") ? 1000 : false;
    },
  });
  const job = (jobs.data ?? []).filter((j) => j.payload["lesson"] === lessonId).at(-1);

  // When a check finishes, progress and unlocks have changed on disk: refetch them.
  const lastState = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (job && lastState.current && lastState.current !== job.state && ["done", "failed"].includes(job.state)) {
      void queryClient.invalidateQueries({ queryKey: ["curricula"] });
      void queryClient.invalidateQueries({ queryKey: ["learn"] });
    }
    lastState.current = job?.state;
  }, [job, queryClient]);

  const run = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/curricula/{path}/{lesson}/check", {
        params: { path: { path: lessonId.split("/")[0] ?? "", lesson: lessonId.split("/")[1] ?? "" } },
        body: { variant: "cpu" },
      })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs", "check"] }),
  });

  return { job, result: job?.result as CheckJobResult | null | undefined, run, error: jobs.error };
}
