import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { ApiProblem, unwrap } from "../api/problem";
import { Button, ButtonLink } from "../components/Button";
import { CardExport } from "../components/CardExport";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ForestPlot } from "../components/ForestPlot";
import { Markdown } from "../components/Markdown";
import { Note } from "../components/Note";
import { ProblemFromError } from "../components/ProblemView";
import { Progress } from "../components/Progress";
import { Readout } from "../components/Readout";
import { StateTag } from "../components/StateTag";
import { Tag } from "../components/Tag";
import { Verdict } from "../components/Verdict";
import { Circle, Stop } from "../icons";
import { useEvents } from "../hooks/useEvents";
import { parseRef, studyCounts, studyState, type RunLine } from "../studies/studyState";
import styles from "./Study.module.css";

type Delta = { mean: number; ci95_low: number | null; ci95_high: number | null; n: number; paired: boolean };
type Row = {
  label: string;
  seeds: number[];
  verdict: string;
  delta: Delta | null;
  text: { params: string; tokens: string; value: string; delta: string; verdict: string; statement: string | null };
};
type Comparison = {
  metric: string;
  baseline: string;
  title: string;
  rows: Row[];
  notes: string[];
  precision_plan: { text?: string; note?: string };
};
type Prediction = {
  variant: string;
  metric: string;
  predicted: number;
  actual: { mean: number; ci95_low: number | null; ci95_high: number | null };
  error: number;
};
type Manifest = { commit: string; study_file_commit: string; committed_via?: string; started_at: string };
type Report = { mode: string; manifest: Manifest | null; predictions: Prediction[]; comparison: Comparison };
type Patch = { state?: string; step?: number; max_steps?: number; val_bpb?: number };

const UNIT: Record<string, string> = { val_bpb: "bpb", val_loss: "nats per token" };

// One study: its runs as a variants × seeds grid that fills in over SSE, the comparison and forest
// plot from finished runs, the report and the evidence bundle.
export function Study() {
  const { name = "" } = useParams();
  const queryClient = useQueryClient();
  const [live, setLive] = useState<Record<string, Patch>>({});
  const [cardOpen, setCardOpen] = useState(false);

  const spec = useQuery({
    queryKey: ["study-spec", name],
    queryFn: () => unwrap(api.GET("/api/studies/{name}/spec", { params: { path: { name } } })),
  });
  const runs = useQuery({
    queryKey: ["runs", `studies/${name}`],
    queryFn: () => unwrap(api.GET("/api/runs", { params: { query: { prefix: `studies/${name}` } } })) as unknown as Promise<RunLine[]>,
  });
  const report = useQuery({
    queryKey: ["study-report", name],
    // 422 until two variants each have a finished run: that is "not yet", not an error
    queryFn: async () => {
      try {
        return (await unwrap(api.GET("/api/studies/{name}/report", { params: { path: { name } } }))) as unknown as Report;
      } catch (e) {
        if (e instanceof ApiProblem && e.status === 422) return null;
        throw e;
      }
    },
  });
  const markdown = useQuery({
    queryKey: ["study-report-md", name, report.data ? report.data.comparison.rows.map((r) => r.seeds.length).join(",") : ""],
    enabled: !!report.data,
    queryFn: () => unwrap(api.GET("/api/studies/{name}/report.md", { params: { path: { name } }, parseAs: "text" })) as unknown as Promise<string>,
  });

  const onEvent = useCallback(
    (type: string, data: unknown) => {
      const d = data as Record<string, unknown> & { ref?: string };
      if (!d.ref) return;
      const ref = d.ref;
      setLive((prev) => {
        const patch: Patch = { ...prev[ref] };
        if (type === "state") {
          patch.state = d["state"] as string;
          patch.step = d["step"] as number;
          patch.max_steps = d["max_steps"] as number;
        } else if (type === "step") patch.step = d["step"] as number;
        else if (type === "eval" && typeof d["val_bpb"] === "number") patch.val_bpb = d["val_bpb"];
        return { ...prev, [ref]: patch };
      });
      // a run that finished changes the comparison and the forest plot
      if (type === "state" && d["state"] === "done") void queryClient.invalidateQueries({ queryKey: ["study-report", name] });
    },
    [name, queryClient],
  );
  const stream = useEvents(`/api/events?prefix=${encodeURIComponent(`studies/${name}`)}`, {
    events: ["state", "step", "eval"],
    onEvent,
    onReset: () => setLive({}),
  });

  const stop = useMutation({
    mutationFn: () => unwrap(api.POST("/api/studies/{name}/stop", { params: { path: { name } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });

  const error = spec.error ?? runs.error ?? report.error;
  if (error) return <ProblemFromError error={error} />;
  if (!spec.data || !runs.data) return <p className="body">Loading the study…</p>;

  const s = spec.data.spec as {
    preset: string;
    seeds: number[];
    baseline?: string;
    mode: string;
    budget?: { tokens?: number };
    variants: { name: string; model: string }[];
  };
  const lines = runs.data.map((r) => ({ ...r, ...live[r.ref] }));
  const total = s.variants.length * s.seeds.length;
  const counts = studyCounts(lines, total);
  const state = studyState(counts, lines);
  const byCell = new Map(lines.flatMap((r) => { const p = parseRef(r.ref, name); return p ? [[`${p.variant}/${p.seed}`, r] as const] : []; }));
  const rep = report.data ?? null;
  const rows = new Map((rep?.comparison.rows ?? []).map((r) => [r.label, r]));
  const manifest = rep?.manifest ?? null;
  const unit = rep ? (UNIT[rep.comparison.metric] ?? rep.comparison.metric) : "bpb";

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <span className={`small ${styles.muted}`}>
          <Link to="/studies" className={styles.link}>Studies</Link> / <span className="value">{name}</span>
        </span>
        <div className={styles.bar}>
          <h1 className={`title ${styles.name}`}>{name}</h1>
          <StateTag state={state} />
          <Tag tone="neutral" icon={Circle}>{s.mode}</Tag>
          <span className={styles.spacer} />
          {state === "running" && (
            <Button variant="danger" onClick={() => stop.mutate()} disabled={stop.isPending}>
              <Stop size={16} aria-hidden="true" />
              Stop study
            </Button>
          )}
          {s.mode === "record" && state === "done" && <Button onClick={() => setCardOpen(true)}>Export ablation card…</Button>}
          <ButtonLink to={`/studies/${name}/edit`}>Open in builder</ButtonLink>
          {rep && (
            <a className={`${styles.link} body-strong`} href={`/api/studies/${encodeURIComponent(name)}/bundle.zip`} download>
              Download bundle
            </a>
          )}
        </div>
        {stop.error && <ProblemFromError error={stop.error} />}
        <span className={`small ${styles.muted}`}>
          {stream === "paused" ? "Live updates paused. Reconnecting…" : state === "running" ? "Live" : `${counts.done} of ${total} runs finished`}
          {manifest ? ` · started ${manifest.started_at.replace("T", " ").slice(0, 16)}` : ""}
        </span>
        <span className={`small ${styles.muted}`}>
          <span className="value">{s.preset}</span>
          {s.budget?.tokens ? ` · ${(s.budget.tokens / 1e6).toFixed(1)}M tokens a run` : ""} · seeds {s.seeds.join(", ")}
          {s.baseline ? <> · baseline <span className="value">{s.baseline}</span></> : null}
          {manifest?.committed_via ? <> · preregistration <span className="value">{manifest.study_file_commit.slice(0, 7)}</span>, committed via {manifest.committed_via}</> : null}
        </span>
      </header>

      {state !== "done" && (
        <section className={styles.progress} aria-label="Progress">
          <Readout label="Runs finished" value={String(counts.done)} unit={`of ${total}`} />
          <div className={styles.bar2}>
            <Progress
              step={counts.done}
              total={total}
              unit="run"
              extra={`${counts.running} running · ${counts.waiting} queued`}
              label="Study progress"
            />
          </div>
        </section>
      )}

      <section className={styles.panel} aria-label="Variants and seeds">
        <div className={styles.title}>
          <h2 className="heading">Variants × seeds</h2>
          <span className={`small ${styles.muted}`}>
            Each cell is one run; its value is the final val_bpb, or the step it has reached. Open a cell for its run page.
          </span>
        </div>
        <div className={styles.wrap}>
          <table className={`${styles.table} small`}>
            <thead>
              <tr>
                <th className="label" scope="col">Variant</th>
                <th className={`label ${styles.right}`} scope="col">Non-emb params</th>
                {s.seeds.map((seed) => (
                  <th key={seed} className="label" scope="col">Seed {seed}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {s.variants.map((v) => (
                <tr key={v.name}>
                  <td>
                    <code className="value-strong">{v.name}</code>
                    {v.name === s.baseline && <span className={`caption ${styles.muted}`}> · baseline</span>}
                  </td>
                  <td className={`value ${styles.right}`}>{rows.get(v.name)?.text.params ?? "–"}</td>
                  {s.seeds.map((seed) => {
                    const run = byCell.get(`${v.name}/${seed}`);
                    if (!run) {
                      return (
                        <td key={seed}>
                          <StateTag state="queued" />
                        </td>
                      );
                    }
                    return (
                      <td key={seed}>
                        <Link to={`/runs/${run.ref}`} className={styles.cell} aria-label={run.ref}>
                          <StateTag state={run.state} />
                          <code className={`value ${run.state === "done" ? "" : styles.muted}`}>
                            {run.state === "done" && run.val_bpb != null
                              ? run.val_bpb.toFixed(3)
                              : run.state === "running"
                                ? `${run.step} / ${run.max_steps}`
                                : ""}
                          </code>
                        </Link>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {rep && (
        <section className={styles.split} aria-label="Comparison">
          <div className={styles.tableCol}>
            <p className="body">{rep.comparison.title}</p>
            <div className={styles.wrap}>
              <table className={`${styles.table} small`}>
                <thead>
                  <tr>
                    <th className="label" scope="col">Variant</th>
                    <th className={`label ${styles.right}`} scope="col">Seeds</th>
                    <th className={`label ${styles.right}`} scope="col">{rep.comparison.metric}</th>
                    <th className={`label ${styles.right}`} scope="col">Δ vs {rep.comparison.baseline}, 95% CI</th>
                    <th className="label" scope="col">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {rep.comparison.rows.map((r) => {
                    const suffix = r.text.verdict.startsWith(r.verdict) ? r.text.verdict.slice(r.verdict.length).trim() : "";
                    return (
                      <tr key={r.label}>
                        <td><code className="value-strong">{r.label}</code></td>
                        <td className={`value ${styles.right}`}>{r.seeds.length}</td>
                        <td className={`value ${styles.right}`}>{r.text.value}</td>
                        <td className={`value-strong ${styles.right} ${r.delta ? "" : styles.muted}`}>{r.text.delta}</td>
                        <td>
                          <Verdict verdict={r.verdict}>{suffix ? <span className={styles.muted}>{suffix}</span> : null}</Verdict>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <span className={`small ${styles.muted}`}>{rep.comparison.precision_plan.text ?? rep.comparison.precision_plan.note}</span>
            {rep.comparison.notes.map((n) => (
              <Note key={n} tone="warn">{n}</Note>
            ))}
          </div>
          <div className={`${styles.panel} ${styles.plotCol}`}>
            <ForestPlot
              caption={`Δ ${rep.comparison.metric} vs ${rep.comparison.baseline}, 95% CI`}
              unit={unit}
              rows={rep.comparison.rows.map((r) => ({ label: r.label, delta: r.delta }))}
            />
            <span className={`caption ${styles.muted}`}>
              From finished runs only, so a half-trained run never moves a point. A variant gets an interval once it and the baseline both
              have 3 finished seeds.
            </span>
          </div>
        </section>
      )}

      {rep && rep.predictions.length > 0 && (
        <section className={styles.panel} aria-label="Predictions">
          <div className={styles.title}>
            <h2 className="heading">Predictions</h2>
            {manifest?.committed_via && (
              <span className={`small ${styles.muted}`}>committed in {manifest.study_file_commit.slice(0, 7)} before the first run</span>
            )}
          </div>
          <div className={styles.wrap}>
            <table className={`${styles.table} small`}>
              <thead>
                <tr>
                  <th className="label" scope="col">Variant</th>
                  <th className="label" scope="col">Metric</th>
                  <th className={`label ${styles.right}`} scope="col">Predicted</th>
                  <th className={`label ${styles.right}`} scope="col">Actual (95% CI)</th>
                  <th className={`label ${styles.right}`} scope="col">Error</th>
                </tr>
              </thead>
              <tbody>
                {rep.predictions.map((p) => (
                  <tr key={`${p.variant}${p.metric}`}>
                    <td><code className="value-strong">{p.variant}</code></td>
                    <td className="value">{p.metric}</td>
                    <td className={`value ${styles.right}`}>{p.predicted.toFixed(3)}</td>
                    <td className={`value ${styles.right}`}>
                      {p.actual.mean.toFixed(3)}
                      {p.actual.ci95_low != null && p.actual.ci95_high != null ? ` [${p.actual.ci95_low.toFixed(3)}, ${p.actual.ci95_high.toFixed(3)}]` : ""}
                    </td>
                    <td className={`value ${styles.right}`}>{`${p.error >= 0 ? "+" : "−"}${(Math.abs(p.error) * 100).toFixed(1)}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {rep && (
        <section className={styles.panel} aria-label="Report">
          <h2 className="heading">Report</h2>
          <div className={styles.report}>{markdown.data ? <Markdown>{markdown.data}</Markdown> : <p className={`small ${styles.muted}`}>Loading the report…</p>}</div>
        </section>
      )}

      {rep && (
        <section className={styles.panel} aria-label="Bundle">
          <h2 className="heading">Bundle</h2>
          <p className={`small ${styles.muted}`}>One zip with everything needed to check or re-run the study.</p>
          <a className={`${styles.link} body-strong`} href={`/api/studies/${encodeURIComponent(name)}/bundle.zip`} download>
            Download {name}-bundle.zip
          </a>
        </section>
      )}

      <CardExport name={name} open={cardOpen} onOpenChange={setCardOpen} />
      <EquivalentCommand cli={`nanoscope status runs/studies/${name}\nnanoscope stop ${name}`} />
    </div>
  );
}
