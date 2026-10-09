import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import { StateTag } from "../components/StateTag";
import { stampText } from "../format";
import { Cpu, GraphicsCard, Play, Warning, X } from "../icons";
import styles from "./Hardware.module.css";

type Device = { name: string; kind: string; memory_total: number | null; memory_free: number | null; label: string | null };
type Hw = { devices: Device[]; cpu_count: number; torch: string };
type Worker = {
  worker_id: string;
  device: string;
  slots: number;
  jobs: number[];
  started_at: string;
  age: number;
  secrets?: Record<string, boolean>;
};
type Job = {
  id: number;
  kind: string;
  state: string;
  ref: string | null;
  device: string | null;
  payload: Record<string, unknown>;
  created_at: number;
};
type Bench = { at: string; model: string; device: string; step_ms: number; tokens_per_sec: number; tflops: number; verdict: string };

const ACTIVE = ["queued", "running", "cancelling"];
const WORKER_GONE_SECONDS = 60;
const SHOWN = 12;
const FILTERS = ["All", "Running", "Queued", "Finished"] as const;
type Filter = (typeof FILTERS)[number];

const gb = (n: number) => (n / 1024 ** 3).toFixed(1);

function target(job: Job): string {
  const p = job.payload;
  return job.ref ?? (p["lesson"] as string | undefined) ?? (p["model"] as string | undefined) ?? (p["file"] as string | undefined) ?? "";
}

// What this machine can train on, who is working, what is queued, and how fast each device is.
export function Hardware() {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<Filter>("All");
  const [showAll, setShowAll] = useState(false);
  const [benchModel, setBenchModel] = useState("");
  const [benchDevice, setBenchDevice] = useState("");
  const [benchPreset, setBenchPreset] = useState("tinystories-5min");

  const hardware = useQuery({ queryKey: ["hardware"], queryFn: () => unwrap(api.GET("/api/hardware")) as unknown as Promise<Hw>, refetchInterval: 5000 });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => unwrap(api.GET("/api/workers")) as unknown as Promise<Worker[]>, refetchInterval: 3000 });
  const jobs = useQuery({
    queryKey: ["jobs", "all"],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { limit: 200 } } })) as unknown as Promise<Job[]>,
    refetchInterval: 2000,
  });
  const history = useQuery({
    queryKey: ["bench-history"],
    queryFn: () => unwrap(api.GET("/api/hardware/bench")) as unknown as Promise<Bench[]>,
    refetchInterval: 5000,
  });
  const models = useQuery({ queryKey: ["models"], queryFn: () => unwrap(api.GET("/api/models")) as unknown as Promise<{ name: string; ref: string }[]> });
  const presets = useQuery({ queryKey: ["presets"], queryFn: () => unwrap(api.GET("/api/presets")) as unknown as Promise<{ name: string }[]> });

  const cancel = useMutation({
    mutationFn: (id: number) => unwrap(api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: id } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const stopStudy = useMutation({
    mutationFn: (name: string) => unwrap(api.POST("/api/studies/{name}/stop", { params: { path: { name } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const bench = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/bench", { body: { model: model, preset: benchPreset, steps: 60, device: device || null } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const error = hardware.error ?? workers.error ?? jobs.error ?? history.error ?? models.error ?? presets.error;
  if (error) return <ProblemFromError error={error} />;
  if (!hardware.data || !workers.data || !jobs.data || !models.data || !presets.data) return <p className="body">Loading hardware…</p>;

  const live = workers.data.filter((w) => w.age < WORKER_GONE_SECONDS);
  const all = jobs.data;
  const count = (s: string) => all.filter((j) => j.state === s).length;
  const matches = (j: Job) =>
    filter === "All" ? true : filter === "Running" ? j.state === "running" || j.state === "cancelling" : filter === "Queued" ? j.state === "queued" : !ACTIVE.includes(j.state);
  // active jobs first (newest last, the order they will start), then the latest finished
  const ordered = [...all.filter((j) => ACTIVE.includes(j.state)), ...all.filter((j) => !ACTIVE.includes(j.state)).reverse()].filter(matches);
  const rows = showAll ? ordered : ordered.slice(0, SHOWN);
  const studies = [...new Set(all.filter((j) => ACTIVE.includes(j.state)).map((j) => /^studies\/([^/]+)\//.exec(j.ref ?? "")?.[1]).filter((s): s is string => !!s))];

  const model = benchModel || models.data[0]?.ref || "";
  const device = benchDevice || hardware.data.devices.at(-1)?.name || "cpu";
  const busyOn = (d: string) => all.filter((j) => j.state === "running" && j.device === d && j.kind === "run").length;

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className="title">Hardware</h1>
        <span className={`small ${styles.muted}`}>
          this machine · torch <span className="value">{hardware.data.torch}</span> · live
        </span>
      </header>

      <section className={styles.row}>
        <section className={`${styles.panel} ${styles.devices}`} aria-label="Devices">
          <h2 className="heading">Devices</h2>
          <div className={styles.wrap}>
            <table className={`${styles.table} small`}>
              <thead>
                <tr>
                  <th className="label" scope="col">Device</th>
                  <th className="label" scope="col">Kind</th>
                  <th className={`label ${styles.right}`} scope="col">Memory free</th>
                  <th className="label" scope="col">Worker</th>
                </tr>
              </thead>
              <tbody>
                {hardware.data.devices.map((d) => {
                  const ws = live.filter((w) => w.device === d.name);
                  const slots = ws.reduce((n, w) => n + w.slots, 0);
                  const used = ws.reduce((n, w) => n + w.jobs.length, 0);
                  return (
                    <tr key={d.name}>
                      <td>
                        <span className={styles.cell}>
                          {d.kind === "cuda" ? <GraphicsCard size={16} aria-hidden="true" /> : <Cpu size={16} aria-hidden="true" />}
                          <code className="value-strong">{d.name}</code>
                          {d.label && <span className={styles.muted}>{d.label}</span>}
                        </span>
                      </td>
                      <td>{d.kind === "cpu" ? `cpu · ${hardware.data.cpu_count} cores` : d.kind}</td>
                      <td className={`value ${styles.right} ${d.memory_free === null ? styles.muted : ""}`}>
                        {d.memory_free === null || d.memory_total === null ? "–" : `${gb(d.memory_free)} of ${gb(d.memory_total)} GB`}
                      </td>
                      <td>
                        {ws.length === 0
                          ? "no worker"
                          : `${ws.length} ${ws.length === 1 ? "worker" : "workers"} · ${slots} ${slots === 1 ? "slot" : "slots"} · ${used === 0 ? "idle" : used >= slots ? "all busy" : `${used} busy`}`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className={`${styles.panel} ${styles.workers}`} aria-label="Workers">
          <h2 className="heading">Workers</h2>
          {live.length === 0 ? (
            <p className={`body ${styles.muted}`}>No worker is running. Start one with <span className="value">nanoscope worker --device cpu</span>; queued jobs wait for it.</p>
          ) : (
            <div className={styles.wrap}>
              <table className={`${styles.table} small`}>
                <thead>
                  <tr>
                    <th className="label" scope="col">Worker</th>
                    <th className="label" scope="col">Device</th>
                    <th className={`label ${styles.right}`} scope="col">Slots</th>
                    <th className="label" scope="col">Jobs now</th>
                    <th className="label" scope="col">Heartbeat</th>
                    <th className="label" scope="col">Secrets it has</th>
                  </tr>
                </thead>
                <tbody>
                  {live.map((w) => (
                    <tr key={w.worker_id}>
                      <td><code className="value">{w.worker_id}</code></td>
                      <td><code className="value">{w.device}</code></td>
                      <td className={`value ${styles.right}`}>{w.slots}</td>
                      <td className={w.jobs.length ? "value" : `small ${styles.muted}`}>{w.jobs.length ? w.jobs.join(", ") : "idle"}</td>
                      <td>{Math.round(w.age)} s ago</td>
                      <td>
                        <span className={styles.secrets}>
                          {Object.entries(w.secrets ?? {}).filter(([, has]) => has).map(([k]) => (
                            <code key={k} className="value">{k}</code>
                          ))}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <span className={`caption ${styles.muted}`}>
            A worker that misses heartbeats for its lease is dropped and its jobs go back to the queue. Secrets are shown by name only; their
            values never leave the worker.
          </span>
        </section>
      </section>

      <section className={`${styles.panel} ${styles.flush}`} aria-label="Jobs">
        <div className={styles.jobsHead}>
          <h2 className="heading">Jobs</h2>
          <span className="small">{count("running")} running</span>
          <span className={`small ${styles.muted}`}>· {count("queued")} queued · {count("done")} done</span>
          <span className={styles.spacer} />
          <div className={`${styles.seg} small`} role="radiogroup" aria-label="Show jobs">
            {FILTERS.map((f) => (
              <button key={f} type="button" role="radio" aria-checked={filter === f} onClick={() => setFilter(f)}>
                {f}
              </button>
            ))}
          </div>
        </div>
        {ordered.length === 0 ? (
          <p className={`small ${styles.muted}`} style={{ margin: 0, padding: "0 16px 16px" }}>No jobs {filter === "All" ? "yet" : "here"}.</p>
        ) : (
          <ul className={styles.jobs}>
            {rows.map((j) => (
              <li key={j.id} className={styles.job}>
                <StateTag state={j.state} />
                <span className={`value ${styles.muted}`}>{j.kind}</span>
                <span className={`value ${styles.target}`} title={target(j)}>
                  {j.kind === "run" && j.ref ? <Link to={`/runs/${j.ref}`} className={styles.link}>{j.ref}</Link> : target(j)}
                </span>
                <span className={`value ${styles.muted}`}>{j.device ?? ""}</span>
                {j.state === "queued" || j.state === "running" ? (
                  <button type="button" className={styles.cancel} aria-label={`Cancel job ${j.id}`} title="Cancel" onClick={() => cancel.mutate(j.id)}>
                    <X size={16} aria-hidden="true" />
                  </button>
                ) : (
                  <span />
                )}
              </li>
            ))}
          </ul>
        )}
        {(ordered.length > SHOWN || studies.length > 0) && (
          <div className={`${styles.jobsFoot} small`}>
            {ordered.length > SHOWN && (
              <>
                <span className={styles.muted}>{showAll ? `All ${ordered.length} shown` : `${ordered.length - SHOWN} more`}</span>
                <Button variant="quiet" size="sm" onClick={() => setShowAll((v) => !v)}>
                  {showAll ? "Show fewer" : `Show all ${ordered.length}`}
                </Button>
              </>
            )}
            <span className={styles.spacer} />
            {studies.map((s) => (
              <Button key={s} variant="danger" size="sm" disabled={stopStudy.isPending} onClick={() => stopStudy.mutate(s)}>
                Stop study {s}
              </Button>
            ))}
          </div>
        )}
        {(cancel.error || stopStudy.error) && <ProblemFromError error={cancel.error ?? stopStudy.error} />}
      </section>

      <section className={styles.panel} aria-label="Bench">
        <div className={styles.benchBar}>
          <div className={styles.benchTitle}>
            <h2 className="heading">Bench</h2>
            <span className={`small ${styles.muted}`}>
              Training speed measured on each device, 60 steps of the preset. Study estimates use these numbers.
            </span>
          </div>
          <label className={styles.field}>
            <span className="label">Model</span>
            <select className={`${styles.select} value`} value={model} onChange={(e) => setBenchModel(e.target.value)}>
              {models.data.map((m) => (
                <option key={m.ref} value={m.ref}>{m.name}</option>
              ))}
            </select>
          </label>
          <label className={styles.field}>
            <span className="label">Device</span>
            <select className={`${styles.select} value`} value={device} onChange={(e) => setBenchDevice(e.target.value)}>
              {hardware.data.devices.map((d) => (
                <option key={d.name} value={d.name}>{d.name}</option>
              ))}
            </select>
          </label>
          <label className={styles.field}>
            <span className="label">Preset</span>
            <select className={`${styles.select} value`} value={benchPreset} onChange={(e) => setBenchPreset(e.target.value)}>
              {presets.data.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </select>
          </label>
          <Button variant="primary" onClick={() => bench.mutate()} disabled={bench.isPending || !model}>
            <Play size={16} aria-hidden="true" />
            Run bench
          </Button>
        </div>
        {busyOn(device) > 0 && (
          <span className={`small ${styles.warn}`}>
            <Warning size={14} aria-hidden="true" />
            {device} is training {busyOn(device)} {busyOn(device) === 1 ? "run" : "runs"}; a bench there now measures a shared device.
          </span>
        )}
        {live.every((w) => w.device !== device) && (
          <span className={`small ${styles.muted}`}>No worker is running on {device}: the bench waits in the queue until one starts.</span>
        )}
        {bench.error && <ProblemFromError error={bench.error} />}
        {history.data && history.data.length === 0 ? (
          <p className={`small ${styles.muted}`} style={{ margin: 0 }}>No bench results yet. Run one to give studies an estimate.</p>
        ) : (
          <div className={styles.wrap}>
            <table className={`${styles.table} small`}>
              <thead>
                <tr>
                  <th className="label" scope="col">Measured</th>
                  <th className="label" scope="col">Model</th>
                  <th className="label" scope="col">Device</th>
                  <th className={`label ${styles.right}`} scope="col">Step</th>
                  <th className={`label ${styles.right}`} scope="col">Tokens/s</th>
                  <th className={`label ${styles.right}`} scope="col">TFLOPs</th>
                  <th className="label" scope="col">Reading</th>
                </tr>
              </thead>
              <tbody>
                {[...(history.data ?? [])].reverse().map((b) => (
                  <tr key={`${b.at}${b.model}${b.device}`}>
                    <td className="value">{stampText(b.at)}</td>
                    <td><code className="value">{b.model}</code></td>
                    <td><code className="value">{b.device}</code></td>
                    <td className={`value ${styles.right}`}>{b.step_ms.toFixed(1)} ms</td>
                    <td className={`value ${styles.right}`}>{Math.round(b.tokens_per_sec).toLocaleString("en-US")}</td>
                    <td className={`value ${styles.right}`}>{b.tflops.toFixed(2)}</td>
                    <td>{b.verdict}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <span className={`caption ${styles.muted}`}>Newest first.</span>
      </section>

      <EquivalentCommand cli={`nanoscope status --workers\nnanoscope bench ${(models.data.find((m) => m.ref === model)?.name ?? "modern").toLowerCase()} --device ${device} --save`} />
    </div>
  );
}
