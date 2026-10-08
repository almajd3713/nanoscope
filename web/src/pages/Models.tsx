import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useLevel } from "../app/level";
import { Button } from "../components/Button";
import { ProblemFromError } from "../components/ProblemView";
import { StateTag } from "../components/StateTag";
import { Tag } from "../components/Tag";
import { countText } from "../format";
import { useBlocks } from "../graph/api";
import { Check, Circle, SealCheck, Warning, X } from "../icons";
import { shows } from "../levels";
import styles from "./Models.module.css";

type Model = { name: string; ref: string; doc: string; shipped: boolean };
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
        <span className={`small ${styles.muted}`}>from your workspace, read without running them</span>
      </header>

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
                  <th className="label" scope="col">Model</th>
                  <th className="label" scope="col">File</th>
                  <th className={`label ${styles.right}`} scope="col">Params</th>
                  <th className={`label ${styles.right}`} scope="col">FLOPs/token</th>
                  <th className="label" scope="col">Trace</th>
                </tr>
              </thead>
              <tbody>
                {mine.map((m) => {
                  const file = m.ref.slice(0, m.ref.lastIndexOf(":"));
                  const job = lastTrace(jobs.data ?? [], m.ref);
                  const done = job?.state === "done" ? job.result : null;
                  return (
                    <tr key={m.ref}>
                      <td>
                        <Link to={`/model/${file}?class=${m.name}`} className={`value ${styles.link}`}>
                          {m.name}
                        </Link>
                      </td>
                      <td className="value">{file}</td>
                      <td className={`value ${styles.right}`}>{done?.params ? countText(done.params.total) : "–"}</td>
                      <td className={`value ${styles.right}`}>{done?.flops_per_token !== undefined ? countText(done.flops_per_token) : "–"}</td>
                      <td>
                        <span className={styles.cell}>
                          {!job && <Tag tone="muted" icon={Circle}>not traced</Tag>}
                          {job && job.state !== "done" && <StateTag state={job.state} />}
                          {job?.state === "done" && <Tag tone="good" icon={Check}>traced</Tag>}
                          <Button size="sm" disabled={trace.isPending || (job !== undefined && ACTIVE.includes(job.state))} onClick={() => trace.mutate(m.ref)}>
                            Trace
                          </Button>
                          {job?.state === "failed" && job.error && <span className={styles.muted}>{job.error}</span>}
                        </span>
                      </td>
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
    </div>
  );
}
