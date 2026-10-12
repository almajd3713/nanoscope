import { useQuery } from "@tanstack/react-query";
import { api } from "./api/client";
import { unwrap } from "./api/problem";

type Targeted = { ref: string | null; payload: Record<string, unknown> };

// The workspace folder on the server, so a job's absolute file can be shown the way you wrote it.
export function useWorkspaceRoot(): string | undefined {
  const facts = useQuery({
    queryKey: ["settings"],
    queryFn: () => unwrap(api.GET("/api/settings")) as unknown as Promise<{ workspace?: string }>,
    staleTime: 60_000,
    retry: false,
  });
  return facts.data?.workspace;
}

// What a job works on: its run ref, else its lesson, model or file. A worker's payload holds the
// absolute path of a workspace file; shown relative to the workspace (models/my_lm.py:MyLM).
export function jobTarget(job: Targeted, workspace?: string): string {
  const p = job.payload;
  const raw = job.ref ?? (p["lesson"] as string | undefined) ?? (p["model"] as string | undefined) ?? (p["file"] as string | undefined) ?? "";
  return workspace && raw.startsWith(workspace + "/") ? raw.slice(workspace.length + 1) : raw;
}
