import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { ButtonLink } from "../components/Button";
import { CompareCurves, type SetCurves } from "../components/CompareCurves";
import { EmptyState } from "../components/EmptyState";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ForestPlot } from "../components/ForestPlot";
import { Note } from "../components/Note";
import { ProblemFromError } from "../components/ProblemView";
import { Verdict } from "../components/Verdict";
import styles from "./Compare.module.css";

type Delta = { mean: number; ci95_low: number | null; ci95_high: number | null; n: number; paired: boolean };
type Row = {
  label: string;
  source: string;
  seeds: number[];
  verdict: string;
  delta: Delta | null;
  params_kind?: string;
  text: { params: string; tokens: string; value: string; delta: string; verdict: string; statement: string | null };
};
type Doc = {
  metric: string;
  baseline: string;
  title: string;
  params_header: string;
  rows: Row[];
  notes: string[];
  curves: { label: string; seeds: { seed: number; points: [number, number][] }[] }[];
  precision_plan: { text?: string; note?: string };
  noise_floor?: { text?: string; note?: string } | null;
};

const UNIT: Record<string, string> = { val_bpb: "bpb", val_loss: "nats per token" };
const SERIES = ["s1", "s2", "s3", "s4", "s5", "s6"] as const;

export function Compare() {
  const [params] = useSearchParams();
  const runs = (params.get("runs") ?? "").split(",").filter(Boolean);
  const baseline = params.get("baseline");
  const preset = params.get("preset");
  const metric = params.get("metric") ?? "val_bpb";

  const compare = useQuery({
    queryKey: ["compare", runs, baseline, preset, metric],
    queryFn: () =>
      unwrap(
        api.POST("/api/compare", {
          body: { sets: runs, baseline: baseline ?? null, preset: preset ?? null, metric },
        }),
      ) as unknown as Promise<Doc>,
    enabled: runs.length >= 2,
  });

  if (runs.length < 2) {
    return (
      <div className={styles.page}>
        <h1 className="title">Compare</h1>
        <EmptyState
          title="Pick at least two runs"
          body="A comparison shows each model's difference from the baseline with its 95% interval, a verdict per model, and every seed's curve."
          action={<ButtonLink to="/runs">Choose runs</ButtonLink>}
          command="nanoscope compare baselines/tinystories-5min/modern baselines/tinystories-5min/gpt2"
        />
      </div>
    );
  }
  if (compare.error) return <ProblemFromError error={compare.error} />;
  if (!compare.data) return <p className={`body ${styles.muted}`}>Loading comparison…</p>;

  const doc = compare.data;
  const baselineRow = doc.rows.find((r) => r.delta === null && r.verdict === "baseline");
  const unit = UNIT[doc.metric] ?? doc.metric;
  // series colours go to the non-baseline rows in order, as on the curves
  let next = 0;
  const colourOf = new Map(doc.rows.map((r) => [r.label, r === baselineRow ? null : (next++ % 6)]));
  const curves: SetCurves[] = doc.curves.map((c) => ({
    label: c.label,
    baseline: c.label === baselineRow?.label,
    seeds: c.seeds,
  }));
  const sentences = doc.rows.filter((r) => r.text.statement);

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <div className={styles.titleRow}>
          <h1 className="title">Compare</h1>
          <span className={styles.spacer} />
          <ButtonLink to="/runs">Change runs</ButtonLink>
        </div>
        <p className="body">{doc.title}</p>
        <div className={`small ${styles.muted} ${styles.list}`}>
          <span>Comparing</span>
          {doc.rows.filter((r) => r !== baselineRow).map((r) => (
            <span key={r.label} className="value">
              {r.source.replace(/^.*\/(baselines\/)/, "$1")}
            </span>
          ))}
          <span>against</span>
          <span className="value">{baselineRow ? baselineRow.source.replace(/^.*\/(baselines\/)/, "$1") : doc.baseline}</span>
        </div>
      </header>

      <section className={styles.panel} aria-label="Verdicts">
        <div className={styles.verdicts}>
          {sentences.map((r) => (
            <Verdict key={r.label} verdict={r.verdict}>
              {r.text.statement}
            </Verdict>
          ))}
        </div>
      </section>

      <section className={styles.split}>
        <div className={styles.tableCol}>
          <div className={styles.wrap}>
            <table className={`${styles.table} small`}>
              <thead>
                <tr>
                  <th className="label" scope="col">Model</th>
                  <th className={`label ${styles.right}`} scope="col">Seeds</th>
                  <th className={`label ${styles.right}`} scope="col">{doc.params_header}</th>
                  <th className={`label ${styles.right}`} scope="col">Tokens</th>
                  <th className={`label ${styles.right}`} scope="col">{doc.metric}</th>
                  <th className={`label ${styles.right}`} scope="col">Δ vs {doc.baseline}, 95% CI</th>
                  <th className="label" scope="col">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {doc.rows.map((r) => {
                  const colour = colourOf.get(r.label);
                  const suffix = r.text.verdict.startsWith(r.verdict) ? r.text.verdict.slice(r.verdict.length) : "";
                  return (
                    <tr key={r.label}>
                      <td>
                        <span className={`${styles.swatch} ${colour === null || colour === undefined ? styles.dashed : styles[SERIES[colour]!]}`} />
                        {r.label}
                      </td>
                      <td className={`value ${styles.right}`}>{r.seeds.length}</td>
                      <td className={`value ${styles.right}`}>{r.text.params}</td>
                      <td className={`value ${styles.right}`}>{r.text.tokens}</td>
                      <td className={`value ${styles.right}`}>{r.text.value}</td>
                      <td className={`value-strong ${styles.right} ${r.delta ? "" : styles.muted}`}>{r.text.delta}</td>
                      <td>
                        <Verdict verdict={r.verdict}>{suffix.trim() ? <span className={styles.muted}>{suffix}</span> : null}</Verdict>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {doc.notes.map((n) => (
            <Note key={n} tone="warn">
              {n}
            </Note>
          ))}
          {(doc.noise_floor?.text ?? doc.noise_floor?.note) && (
            <span className={`small ${styles.muted}`}>{doc.noise_floor?.text ?? doc.noise_floor?.note}</span>
          )}
          <span className={`small ${styles.muted}`}>{doc.precision_plan.text ?? doc.precision_plan.note}</span>
        </div>
        <div className={`${styles.panel} ${styles.plotCol}`}>
          <ForestPlot
            caption={`Δ ${doc.metric} vs ${doc.baseline}, 95% CI`}
            unit={unit}
            rows={doc.rows.map((r) => ({ label: r.label, delta: r.delta }))}
          />
        </div>
      </section>

      <section className={styles.panel} aria-label="Curves">
        <div className={styles.titleRow}>
          <h2 className="heading">Curves, every seed</h2>
          <span className={`small ${styles.muted}`}>seed 0 labelled; the other seeds are drawn thin in the same colour</span>
        </div>
        <CompareCurves sets={curves} xLabel="tokens" yLabel={`${doc.metric} (lower is better)`} />
      </section>

      <EquivalentCommand
        cli={`nanoscope compare ${runs.join(" ")}${preset ? ` --preset ${preset}` : ""}`}
        python={`from nanoscope import compare\n\nprint(compare(${runs.map((r) => `"${r}"`).join(", ")}${preset ? `, preset="${preset}"` : ""}))`}
      />
      <Link to="/runs" className={`small ${styles.link}`}>
        Back to runs
      </Link>
    </div>
  );
}
