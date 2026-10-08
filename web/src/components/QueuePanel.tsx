import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useLevel } from "../app/level";
import { shows } from "../levels";
import { CaretDown, CaretRight, Cpu, GraphicsCard, X } from "../icons";
import { ProblemFromError } from "./ProblemView";
import { StateTag } from "./StateTag";
import styles from "./QueuePanel.module.css";

const ACTIVE = ["queued", "running", "cancelling"];
const FINISHED_SHOWN = 5;
const WORKER_GONE_SECONDS = 60;

type Job = {
  id: number;
  kind: string;
  state: string;
  ref: string | null;
  payload: Record<string, unknown>;
  created_at: number;
};
type Worker = { worker_id: string; device: string; slots: number; jobs?: unknown[]; age?: number };

function target(job: Job): string {
  const p = job.payload;
  return job.ref ?? (p["lesson"] as string | undefined) ?? (p["model"] as string | undefined) ?? (p["file"] as string | undefined) ?? "";
}

// The shell footer: one line when collapsed, the jobs when opened. Cancelling a run saves a
// checkpoint first, so cancelling asks nothing.
export function QueuePanel() {
  const [open, setOpen] = useState(false);
  const level = useLevel();
  const queryClient = useQueryClient();
  const jobs = useQuery({
    queryKey: ["jobs", "all"],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { limit: 50 } } })) as Promise<Job[]>,
    refetchInterval: 3000,
  });
  const workers = useQuery({
    queryKey: ["workers"],
    queryFn: () => unwrap(api.GET("/api/workers")) as unknown as Promise<Worker[]>,
    refetchInterval: 5000,
  });
  const cancel = useMutation({
    mutationFn: (id: number) => unwrap(api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: id } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const all = jobs.data ?? [];
  const count = (s: string) => all.filter((j) => j.state === s).length;
  const live = (workers.data ?? []).filter((w) => (w.age ?? 0) < WORKER_GONE_SECONDS);
  const active = all.filter((j) => ACTIVE.includes(j.state));
  const recent = all.filter((j) => !ACTIVE.includes(j.state)).slice(-FINISHED_SHOWN);
  const shown = [...active, ...recent.reverse()];
  const busy = live.some((w) => (w.jobs?.length ?? 0) > 0);

  const summary = [count("running") && `${count("running")} running`, count("queued") && `${count("queued")} queued`, count("cancelling") && `${count("cancelling")} cancelling`].filter(Boolean);

  return (
    <footer className={styles.footer} aria-label="Queue">
      {open && (
        <div className={styles.panel}>
          {jobs.error ? (
            <div className={styles.note}>
              <ProblemFromError error={jobs.error} />
            </div>
          ) : shown.length === 0 ? (
            <p className={`small ${styles.muted} ${styles.note}`} style={{ margin: 0 }}>
              No jobs yet. A run, a lesson check or a model trace shows up here while it works.
            </p>
          ) : (
            <ul className={styles.jobs}>
              {shown.map((job) => (
                <li key={job.id} className={styles.job}>
                  <span>
                    <StateTag state={job.state} />
                  </span>
                  <span className={`value ${styles.muted}`}>{job.kind}</span>
                  <span className={`value ${styles.target}`} title={target(job)}>
                    {job.kind === "run" && job.ref ? <Link to={`/runs/${job.ref}`}>{job.ref}</Link> : target(job)}
                  </span>
                  <span />
                  {job.state === "queued" || job.state === "running" ? (
                    <button
                      type="button"
                      className={styles.cancel}
                      aria-label={`Cancel job ${job.id}`}
                      title="Cancel"
                      onClick={() => cancel.mutate(job.id)}
                    >
                      <X size={16} aria-hidden="true" />
                    </button>
                  ) : (
                    <span />
                  )}
                </li>
              ))}
            </ul>
          )}
          {cancel.error && (
            <div className={styles.note}>
              <ProblemFromError error={cancel.error} />
            </div>
          )}
          <p className={`small ${styles.muted} ${styles.note}`} style={{ margin: 0 }}>
            Cancelling a run saves a checkpoint first; train it again and it resumes.
          </p>
        </div>
      )}
      <div className={`${styles.bar} small`}>
        <button type="button" className={`${styles.toggle} small`} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? <CaretDown size={16} aria-hidden="true" /> : <CaretRight size={16} aria-hidden="true" />}
          Queue
        </button>
        <span>{summary.length ? summary.join(" · ") : "nothing running"}</span>
        <span className={styles.muted}>
          {live.length === 0 ? "· no worker" : !shows("queue.devices", level) ? `· ${live.length === 1 ? live[0]!.device : `${live.length} workers`} ${busy ? "busy" : "idle"}` : ""}
        </span>
        <span className={styles.spacer} />
        {shows("queue.devices", level) &&
          live.map((w) => (
            <span key={w.worker_id} className={`small ${styles.worker}`}>
              {w.device.startsWith("cuda") ? <GraphicsCard size={16} aria-hidden="true" /> : <Cpu size={16} aria-hidden="true" />}
              {w.worker_id} · {w.device} · {w.slots} {w.slots === 1 ? "slot" : "slots"} · {(w.jobs?.length ?? 0) > 0 ? "busy" : "idle"}
            </span>
          ))}
        {shows("queue.devices", level) && (
          <Link to="/hardware" className={`small ${styles.muted}`}>
            Hardware
          </Link>
        )}
      </div>
    </footer>
  );
}
