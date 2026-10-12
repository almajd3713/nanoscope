import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useLevel } from "../app/level";
import { useState } from "react";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { NewModelDialog } from "../components/NewModelDialog";
import { ProblemFromError } from "../components/ProblemView";
import { StateTag } from "../components/StateTag";
import { Tag } from "../components/Tag";
import { countText } from "../format";
import { useBlocks } from "../graph/api";
import { Check, Circle, Plus, SealCheck, Warning, X } from "../icons";
import { shows } from "../levels";
import styles from "./Models.module.css";

type Model = { name: string; ref: string; doc: string; shipped: boolean; source?: string | null };
type Job = {
  id: number;
  kind: string;
  state: string;
  payload: Record<string, unknown>;
  result: { params?: { total: number; non_embedding: number }; flops_per_token?: number } | null;
  error: string | null;
};

const ACTIVE = ["queued", "running", "cancelling"];

// The newest describe job whose model is this ref (a worker's payload has the absolute path).
function lastTrace(jobs: Job[], ref: string): Job | undefined {
  return jobs.filter((j) => typeof j.payload["model"] === "string" && (j.payload["model"] as string).endsWith(ref)).at(-1);
}

function Certification({ state }: { state: string }) {
  switch (state) {
    case "certified":
      return <Tag tone="good" icon={SealCheck}>certified</Tag>;
    case "failed":
      return <Tag tone="bad" icon={X}>failed its check</Tag>;
    case "stale":
      return <Tag tone="warn" icon={Warning}>changed since</Tag>;
    default:
      return <Tag tone="muted" icon={Circle}>not certified</Tag>;
  }
}

// Your models and your blocks. A model is read from its file (never imported); its numbers come
// from the last trace, a describe job a worker ran. Each opens the model page.
export function Models() {
  const level = useLevel();
  const queryClient = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: () => unwrap(api.GET("/api/models")) as unknown as Promise<Model[]> });
  const jobs = useQuery({
    queryKey: ["jobs", "describe"],
    queryFn: () => unwrap(api.GET("/api/jobs", { params: { query: { kind: "describe", limit: 200 } } })) as unknown as Promise<Job[]>,
    refetchInterval: (q) => ((q.state.data ?? []).some((j) => ACTIVE.includes(j.state)) ? 1000 : false),
  });
  const blocks = useBlocks();
  const [creating, setCreating] = useState(false);
  const files = useQuery({ queryKey: ["files", "models"], queryFn: () => unwrap(api.GET("/api/files", { params: { query: { glob: "models/**/*.py" } } })) as unknown as Promise<{ path: string; modified: number }[]> });
  const edited = (file: string) => {
    const f = files.data?.find((x) => x.path === file);
    return f ? new Date(f.modified * 1000).toISOString().replace("T", " ").slice(0, 16) : "–";
  };
  const trace = useMutation({
    mutationFn: (ref: string) => unwrap(api.POST("/api/models/{ref}/describe", { params: { path: { ref } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs", "describe"] }),
  });
  const certify = useMutation({
    mutationFn: (name: string) => unwrap(api.POST("/api/blocks/{name}/certify", { params: { path: { name } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
  });

  const error = models.error ?? jobs.error ?? blocks.error;
  if (error) return <ProblemFromError error={error} />;
  const mine = (models.data ?? []).filter((m) => !m.shipped);
  const myBlocks = (blocks.data?.blocks ?? []).filter((b) => b.user);

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className="title">Models</h1>
        <span className={`small ${styles.muted}`}>model classes found in <span className="value">workspace/</span>, read without running them</span>
        <span className={styles.spacer} />
        <Button variant="primary" onClick={() => setCreating(true)}>
          <Plus size={16} aria-hidden="true" />
          New model
        </Button>
      </header>
      <NewModelDialog open={creating} onOpenChange={setCreating} />

      <section className={styles.section} aria-label="Your models">
        <h2 className="heading">Your models</h2>
        {!models.data ? (
          <p className={`body ${styles.muted}`}>Reading the workspace…</p>
        ) : mine.length === 0 ? (
          <p className={`body ${styles.muted}`}>
            No model files yet. <Link to="/learn" className={styles.link}>Start a lesson</Link> to get one.
          </p>
        ) : (
          <div className={styles.wrap}>
            <table className={`${styles.table} small`}>
              <thead>
                <tr>
                  <th className="label" scope="col">File</th>
                  <th className="label" scope="col">Class</th>
                  <th className={`label ${styles.right}`} scope="col">Params</th>
                  <th className={`label ${styles.right}`} scope="col">FLOPs/token</th>
                  <th className="label" scope="col">Last trace</th>
                  <th className="label" scope="col">Edited</th>
                </tr>
              </thead>
              <tbody>
                {mine.map((m) => {
                  const file = m.ref.slice(0, m.ref.lastIndexOf(":"));
                  const job = lastTrace(jobs.data ?? [], m.ref);
                  const done = job?.state === "done" ? job.result : null;
                  const busy = job !== undefined && ACTIVE.includes(job.state);
                  return (
                    <tr key={m.ref}>
                      <td>
                        <Link to={`/model/${file}?class=${m.name}`} className={`value ${styles.link}`}>
                          {file}
                        </Link>
                      </td>
                      <td className="value">{m.name}</td>
                      <td className={`value ${styles.right}`}>{done?.params ? countText(done.params.total) : "–"}</td>
                      <td className={`value ${styles.right}`}>{done?.flops_per_token !== undefined ? countText(done.flops_per_token) : "–"}</td>
                      <td>
                        <span className={styles.cell}>
                          {job?.state === "done" && <span className={styles.good}><Check size={14} aria-hidden="true" /> no shape errors</span>}
                          {job?.state === "failed" && <span className={styles.bad}><X size={14} aria-hidden="true" /> {job.error ?? "the trace failed"}</span>}
                          {busy && job && <StateTag state={job.state} />}
                          {!job && <span className={styles.muted}>not traced yet</span>}
                          <Button size="sm" variant="quiet" disabled={trace.isPending || busy} onClick={() => trace.mutate(m.ref)}>
                            Trace
                          </Button>
                        </span>
                      </td>
                      <td className={`value ${styles.muted}`}>{edited(file)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {trace.error && <ProblemFromError error={trace.error} />}
      </section>

      {shows("certification", level) && (
        <section className={styles.section} aria-label="Your blocks">
          <h2 className="heading">Your blocks</h2>
          {myBlocks.length === 0 ? (
            <p className={`body ${styles.muted}`}>
              Blocks you register with <span className="value">register_block(reference=…)</span> are listed here, with whether they match their reference.
            </p>
          ) : (
            <div className={styles.wrap}>
              <table className={`${styles.table} small`}>
                <thead>
                  <tr>
                    <th className="label" scope="col">Block</th>
                    <th className="label" scope="col">Family</th>
                    <th className="label" scope="col">Reference</th>
                    <th className="label" scope="col">Certification</th>
                  </tr>
                </thead>
                <tbody>
                  {myBlocks.map((b) => (
                    <tr key={b.name}>
                      <td className="value">{b.name}</td>
                      <td>{b.family}</td>
                      <td className="value">{(b as { reference?: string | null }).reference ?? "–"}</td>
                      <td>
                        <span className={styles.cell}>
                          <Certification state={b.certification?.state ?? "uncertified"} />
                          <Button
                            size="sm"
                            disabled={certify.isPending || !(b as { reference?: string | null }).reference}
                            onClick={() => certify.mutate(b.name)}
                          >
                            Certify
                          </Button>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {certify.error && <ProblemFromError error={certify.error} />}
          {certify.isSuccess && <p className={`small ${styles.muted}`}>The check is queued; the result appears here when a worker finishes it.</p>}
        </section>
      )}
      <EquivalentCommand cli={`nanoscope graph ${mine[0]?.source ?? "models/my_lm.py"}\nnanoscope blocks`} />
    </div>
  );
}
