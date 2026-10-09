import { useQuery } from "@tanstack/react-query";
import { memo, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { ApiProblem, unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { Field } from "../components/Field";
import { Note } from "../components/Note";
import { ProblemFromError, ProblemView } from "../components/ProblemView";
import { Segmented } from "../components/Segmented";
import { StateTag } from "../components/StateTag";
import { Check, Play } from "../icons";
import { cell, heat, inspectCommand, LEGEND, MAX_PROMPT_TOKENS, probText, quote, topWeights, weightText, type Attention, type InspectReport } from "../inspect/report";
import { stepsOf, useCheckpoints, useInspect, useInspectAcross } from "../inspect/useInspect";
import styles from "./Inspect.module.css";

const DEFAULT_PROMPT = "Once upon a time";

type Selection = { module: string | null; head: number; row: number | null };

const HeatMap = memo(function HeatMap({ weights, tokens, size, outlineRow }: { weights: number[][]; tokens: number; size: number; outlineRow?: number | null }) {
  const cells = [];
  for (let t = 0; t < tokens; t++) {
    for (let j = 0; j < tokens; j++) {
      cells.push(
        <span
          key={`${t}-${j}`}
          style={{
            background: heat(cell(weights, t, j)),
            boxShadow: t === outlineRow ? "inset 0 1px 0 var(--accent), inset 0 -1px 0 var(--accent)" : undefined,
          }}
        />,
      );
    }
  }
  return (
    <span className={styles.map} style={{ gridTemplateColumns: `repeat(${tokens}, ${size}px)`, gridAutoRows: `${size}px` }}>
      {cells}
    </span>
  );
});

function Maps({ report, sel, onPick }: { report: InspectReport; sel: { attn: Attention; head: number }; onPick: (module: string, head: number) => void }) {
  const n = report.tokens.length;
  const size = Math.max(2, Math.min(5, Math.floor(100 / Math.max(n, 1))));
  const maxHeads = Math.max(...report.attention.map((a) => a.heads));
  return (
    <div className={styles.maps} style={{ gridTemplateColumns: `40px repeat(${maxHeads}, max-content)` }}>
      <span />
      {Array.from({ length: maxHeads }, (_, h) => (
        <span key={h} className={`caption ${styles.muted} ${styles.center}`}>
          head {h}
        </span>
      ))}
      {report.attention.map((a) => (
        <Row key={a.module} attn={a} n={n} size={size} sel={sel} onPick={onPick} />
      ))}
    </div>
  );
}

function Row({ attn, n, size, sel, onPick }: { attn: Attention; n: number; size: number; sel: { attn: Attention; head: number }; onPick: (module: string, head: number) => void }) {
  return (
    <>
      <span className={`caption ${styles.muted}`}>L{attn.layer}</span>
      {attn.weights.map((w, h) => {
        const on = sel.attn.module === attn.module && sel.head === h;
        return (
          <button
            key={h}
            type="button"
            className={`${styles.mapButton} ${on ? styles.mapOn : ""}`}
            aria-label={`${attn.module} head ${h}`}
            aria-pressed={on}
            onClick={() => onPick(attn.module, h)}
          >
            <HeatMap weights={w} tokens={n} size={size} />
          </button>
        );
      })}
    </>
  );
}

function BigMap({ report, attn, head, row, onRow }: { report: InspectReport; attn: Attention; head: number; row: number; onRow: (row: number) => void }) {
  const n = report.tokens.length;
  const size = Math.max(5, Math.min(17, Math.floor(320 / Math.max(n, 1))));
  const weights = attn.weights[head] ?? [];
  const col = `70px repeat(${n}, ${size}px)`;
  return (
    <figure className={styles.figure}>
      <figcaption className="small">
        <code className="value-strong">{attn.module}</code> · head {head}{" "}
        <span className={styles.muted}>
          · the row of {quote(report.tokens[row]?.text ?? "")} (position {row}) is selected
        </span>
      </figcaption>
      <div className={styles.big} style={{ gridTemplateColumns: col, gridAutoRows: `${size}px` }}>
        {report.tokens.map((tok, t) => (
          <BigRow key={t} t={t} label={quote(tok.text)} n={n} weights={weights} on={t === row} onRow={onRow} />
        ))}
      </div>
      <div className={styles.cols} style={{ gridTemplateColumns: col }}>
        <span />
        {report.tokens.map((tok, j) => (
          <span key={j} className={`caption ${styles.muted} ${styles.colLabel}`} style={{ lineHeight: `${size}px` }}>
            {quote(tok.text)}
          </span>
        ))}
      </div>
    </figure>
  );
}

function BigRow({ t, label, n, weights, on, onRow }: { t: number; label: string; n: number; weights: number[][]; on: boolean; onRow: (row: number) => void }) {
  return (
    <>
      <button type="button" className={`caption ${styles.rowLabel} ${on ? styles.rowOn : ""}`} aria-pressed={on} onClick={() => onRow(t)}>
        {label}
      </button>
      {Array.from({ length: n }, (_, j) => (
        <span
          key={j}
          title={j <= t ? `${t} reads ${j}: ${weightText(weights[t]?.[j] ?? 0)}` : undefined}
          style={{ background: heat(cell(weights, t, j)), boxShadow: on ? "inset 0 2px 0 var(--accent), inset 0 -2px 0 var(--accent)" : undefined }}
          onClick={() => onRow(t)}
        />
      ))}
    </>
  );
}

function Lens({ report, row, onRow }: { report: InspectReport; row: number; onRow: (row: number) => void }) {
  const layers = report.lens.layers;
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table} aria-label="Logit lens by layer">
        <thead>
          <tr>
            <th className="label" scope="col" style={{ textAlign: "right" }}>
              Pos
            </th>
            <th className="label" scope="col">
              Token
            </th>
            {layers.map((l, i) => (
              <th key={l.name} className="label" scope="col">
                {i === layers.length - 1 && layers.length > 1 ? `${l.name} = output` : l.name}
              </th>
            ))}
            <th className="label" scope="col">
              Actual next (p, rank)
            </th>
          </tr>
        </thead>
        <tbody>
          {report.tokens.map((tok, t) => {
            const next = layers[0]?.positions[t]?.next ?? null;
            return (
              <tr key={t} className={t === row ? styles.selectedRow : undefined} onClick={() => onRow(t)}>
                <td className="value" style={{ textAlign: "right", color: "var(--ink-muted)" }}>
                  {t}
                </td>
                <td>
                  <code className="value-strong pre">{quote(tok.text)}</code>
                </td>
                {layers.map((l) => {
                  const guess = l.positions[t]?.top[0];
                  const hit = guess !== undefined && next !== null && guess.id === next.id;
                  return (
                    <td key={l.name} className={styles.nowrap}>
                      {guess && (
                        <span className={styles.guess}>
                          <code className="value pre">{quote(guess.text)}</code>
                          <span className={`value ${styles.muted}`}>{probText(guess.p)}</span>
                          {hit && <Check size={14} aria-label="the actual next token" className={styles.good} />}
                        </span>
                      )}
                    </td>
                  );
                })}
                <td className={styles.nowrap}>
                  {next ? (
                    <>
                      <code className="value pre">{quote(next.text)}</code>{" "}
                      <span className={`value ${styles.muted}`}>
                        {probText(next.p)}, #{next.rank}
                      </span>
                    </>
                  ) : (
                    <span className={styles.muted}>–</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AcrossSteps({ ref_, prompt, steps, sel, row }: { ref_: string; prompt: string; steps: number[]; sel: { module: string; head: number }; row: number }) {
  const results = useInspectAcross(ref_, prompt, steps, true);
  const done = results.filter((r) => r.data).length;
  return (
    <>
      <span className="small muted">
        {done} of {steps.length} jobs done · rows are the token attending
      </span>
      <div className={styles.across}>
        {results.map((r, i) => {
          const step = steps[i]!;
          const report = r.data;
          if (!report) {
            return (
              <figure key={step} className={styles.stepFigure}>
                <figcaption className="small">
                  <span className={styles.muted}>step</span> <code className="value-strong">{step}</code>
                </figcaption>
                {r.error ? <ProblemFromError error={r.error} /> : <span className={`caption ${styles.muted}`}>waiting for the job</span>}
              </figure>
            );
          }
          const attn = report.attention.find((a) => a.module === sel.module);
          const weights = attn?.weights[sel.head];
          const n = report.tokens.length;
          const size = Math.max(2, Math.min(8, Math.floor(150 / Math.max(n, 1))));
          const reads = weights?.[row] ? topWeights(weights[row], report.tokens, 1)[0] : undefined;
          const out = report.lens.layers.at(-1)?.positions[row];
          const guess = out?.top[0];
          return (
            <figure key={step} className={styles.stepFigure}>
              <figcaption className="small">
                <span className={styles.muted}>step</span> <code className="value-strong">{report.step}</code>
              </figcaption>
              {weights && <HeatMap weights={weights} tokens={n} size={size} outlineRow={row} />}
              {reads && (
                <span className="caption">
                  {quote(report.tokens[row]?.text ?? "")} reads <code className="value">{quote(reads.text)}</code> ({reads.pos}):{" "}
                  <span className="value-strong">{weightText(reads.w)}</span>
                </span>
              )}
              {guess && (
                <span className={`caption ${styles.muted}`}>
                  output guesses <code className="value">{quote(guess.text)}</code> {guess.p.toFixed(4)}
                  {out?.next && (
                    <>
                      ; <code className="value">{quote(out.next.text)}</code> is {out.next.p.toFixed(4)}, #{out.next.rank}
                    </>
                  )}
                </span>
              )}
            </figure>
          );
        })}
      </div>
    </>
  );
}

export function Inspect() {
  const ref = useParams()["*"] ?? "";
  const run = useQuery({
    queryKey: ["run", ref],
    queryFn: () => unwrap(api.GET("/api/runs/{ref}", { params: { path: { ref } } })),
  });
  const checkpoints = useCheckpoints(ref);
  const kept = useMemo(() => stepsOf(checkpoints.data ?? []), [checkpoints.data]);

  const [text, setText] = useState(DEFAULT_PROMPT);
  const [prompt, setPrompt] = useState<string | null>(DEFAULT_PROMPT);
  const [step, setStep] = useState<number | null>(null);
  const [view, setView] = useState<"one" | "across">("one");
  const [sel, setSel] = useState<Selection>({ module: null, head: 0, row: null });

  const query = useInspect(ref, prompt, step);
  const [last, setLast] = useState<InspectReport | null>(null);
  if (query.data && query.data !== last) setLast(query.data); // keep the last good result on screen
  const report = query.data ?? last;

  // After 3 seconds without a result the page says what it waits on.
  const waitKey = `${prompt}|${step}`;
  const [slowKey, setSlowKey] = useState<string | null>(null);
  useEffect(() => {
    if (!query.isFetching) return;
    const timer = setTimeout(() => setSlowKey(waitKey), 3000);
    return () => clearTimeout(timer);
  }, [query.isFetching, waitKey]);
  const slow = slowKey === waitKey;

  const attn = report?.attention.find((a) => a.module === sel.module) ?? report?.attention.at(-1);
  const head = attn ? Math.min(sel.head, attn.heads - 1) : 0;
  const row = report ? Math.min(sel.row ?? report.tokens.length - 1, report.tokens.length - 1) : 0;
  const archived = kept.filter((s) => s.archived).map((s) => s.step);
  const command = inspectCommand(ref, prompt ?? text, step);
  const state = run.data?.status?.state;
  const shownStep = step ?? report?.step ?? kept.at(-1)?.step ?? null;

  const submit = () => {
    if (text.trim() !== "") setPrompt(text);
  };
  const topRow = attn && report ? topWeights(attn.weights[head]?.[row] ?? [], report.tokens) : [];
  const layers = report?.attention.length ?? 0;

  return (
    <div className={styles.page}>
      <div className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            <Link to="/runs" className={styles.link}>
              Runs
            </Link>{" "}
            /{" "}
            <Link to={`/runs/${ref}`} className={`value ${styles.link}`}>
              {ref}
            </Link>
          </span>
          <div className={styles.titleRow}>
            <h1 className="title">Inspect</h1>
            <code className={`value ${styles.muted}`}>{ref}</code>
            {state && <StateTag state={state} />}
          </div>
          <span className={`small ${styles.muted}`}>
            What the model attends to and what it would predict after each layer, for your prompt, at any checkpoint the run kept.
            {report && (
              <>
                {" "}
                <code className="value">{report.model}</code>
                {layers > 0 && attn && (
                  <>
                    {" "}
                    · {layers} {layers === 1 ? "layer" : "layers"} × {attn.heads} {attn.heads === 1 ? "head" : "heads"}
                  </>
                )}
              </>
            )}
          </span>
        </header>

        <section className={styles.panel} aria-label="Prompt and checkpoint">
          <div className={styles.promptRow}>
            <div className={styles.promptField}>
              <Field
                label="Prompt"
                value={text}
                onChange={setText}
                onCommit={submit}
                help={report ? `${report.tokens.length} tokens; at most ${MAX_PROMPT_TOKENS}` : `at most ${MAX_PROMPT_TOKENS} tokens`}
              />
            </div>
            <Button variant="primary" disabled={text.trim() === "" || query.isFetching} onClick={submit}>
              <Play size={16} aria-hidden="true" />
              Inspect
            </Button>
          </div>
          <div className={styles.group}>
            <span className="label">Checkpoint</span>
            {kept.length > 0 ? (
              <Segmented
                label="Checkpoint"
                items={kept.map((s) => ({ id: String(s.step), label: <span className="value">{s.step}</span>, title: s.archived ? "archived" : "regular checkpoint" }))}
                value={String(shownStep)}
                onChange={(id) => setStep(Number(id))}
              />
            ) : (
              <span className={`small ${styles.muted}`}>no checkpoint yet</span>
            )}
            {kept.length > 0 && kept.length === 1 && (
              <>
                <Note tone="info">
                  This run kept only its latest checkpoint, step {kept[0]!.step}. To look back over training, train again with archived steps; they are not part of the run&apos;s identity, so the run lands in the same folder.
                </Note>
                <EquivalentCommand python={`from nanoscope import run\n\nrun(Model, "<preset>", checkpoint_steps=[100, 500])`} />
              </>
            )}
            {kept.length > 1 && (
              <span className={`caption ${styles.muted}`}>
                {archived.length > 0 ? (
                  <>
                    Archived with <code className="value">checkpoint_steps</code>: {archived.join(", ")}, kept for good. The others are regular checkpoints; they are pruned as a run goes on.
                  </>
                ) : (
                  <>None of these were archived, so older ones are pruned as the run goes on.</>
                )}{" "}
                Changing the step runs the same prompt again.
              </span>
            )}
          </div>
          {archived.length > 1 && (
            <div className={styles.group}>
              <span className="label">View</span>
              <Segmented
                label="View"
                items={[
                  { id: "one", label: "One step" },
                  { id: "across", label: "Across steps" },
                ]}
                value={view}
                onChange={(id) => setView(id as "one" | "across")}
              />
            </div>
          )}
        </section>

        {query.error &&
          (query.error instanceof ApiProblem ? (
            <ProblemFromError error={query.error} />
          ) : (
            <ProblemView title="The inspect job failed" detail={query.error instanceof Error ? query.error.message : String(query.error)} />
          ))}
        {query.error && report && (
          <span className={`small ${styles.muted}`}>
            The maps and lens below still show the previous result: prompt {quote(report.prompt)}, step {report.step}.
          </span>
        )}
        {!report && !query.error && (
          <section className={styles.panel} aria-label="Waiting">
            <span className={`body ${styles.muted}`}>
              {slow
                ? "Waiting for a worker to start the inspect job. It is queued; a worker running a training job takes it when that slot frees."
                : "Loading the checkpoint…"}
            </span>
          </section>
        )}

        {report && (
          <>
            {report.notes?.map((n) => (
              <Note key={n} tone="info">
                {n}
              </Note>
            ))}
            {view === "across" && archived.length > 1 && prompt !== null && attn ? (
              <section className={styles.panel} aria-label="Across steps">
                <h2 className="heading">Across steps: one head over training</h2>
                <span className={`small ${styles.muted}`}>
                  <code className="value">{attn.module}</code> head {head}, the same prompt at every archived step, one inspect job per step.
                </span>
                <AcrossSteps ref_={ref} prompt={prompt} steps={archived} sel={{ module: attn.module, head }} row={row} />
              </section>
            ) : (
              attn && (
                <section className={styles.panel} aria-label="Attention">
                  <div className={styles.sectionHead}>
                    <h2 className="heading">Attention</h2>
                    <span className={`small ${styles.muted}`}>one map per head: rows are the token attending, columns the tokens it reads; the upper right is the future, which a causal model cannot see</span>
                  </div>
                  <div className={styles.mapsRow}>
                    <Maps report={report} sel={{ attn, head }} onPick={(module, h) => setSel((s) => ({ ...s, module, head: h }))} />
                    <BigMap report={report} attn={attn} head={head} row={row} onRow={(r) => setSel((s) => ({ ...s, row: r }))} />
                  </div>
                </section>
              )
            )}
            <section className={styles.panel} aria-label="Logit lens">
              <div className={styles.sectionHead}>
                <h2 className="heading">Logit lens</h2>
                <span className={`small ${styles.muted}`}>the next-token guess read off the residual stream after each layer, through the final norm and head, with its probability</span>
              </div>
              <Lens report={report} row={row} onRow={(r) => setSel((s) => ({ ...s, row: r }))} />
              <span className={`small ${styles.muted}`}>A tick marks a guess that is the prompt&apos;s actual next token.</span>
            </section>
          </>
        )}
      </div>

      <aside className={styles.aside}>
        {report && attn && (
          <section className={styles.panel} aria-label="Selected">
            <h2 className={`label ${styles.muted}`}>Selected</h2>
            <span className="small">
              <code className="value-strong">{attn.module}</code> head {head}, row <code className="value">{quote(report.tokens[row]?.text ?? "")}</code> ({row})
            </span>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className="label" scope="col" style={{ textAlign: "right" }}>
                    Pos
                  </th>
                  <th className="label" scope="col">
                    Reads
                  </th>
                  <th className="label" scope="col" style={{ textAlign: "right" }}>
                    Weight
                  </th>
                </tr>
              </thead>
              <tbody>
                {topRow.map((r) => (
                  <tr key={r.pos}>
                    <td className="value" style={{ textAlign: "right" }}>
                      {r.pos}
                    </td>
                    <td>
                      <code className="value pre">{quote(r.text)}</code>
                    </td>
                    <td className="value" style={{ textAlign: "right" }}>
                      {weightText(r.w)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <span className={`caption ${styles.muted}`}>
              The {topRow.length} largest of the row&apos;s {row + 1} {row === 0 ? "weight" : "weights"}, which sum to 1. Pick another row in the map, or another head on the left.
            </span>
          </section>
        )}
        <section className={styles.panel} aria-label="Attention weight">
          <h2 className={`label ${styles.muted}`}>Attention weight</h2>
          <div className={styles.legend}>
            {LEGEND.map((g) => (
              <span key={g.label} className="small">
                <span className={styles.swatch} style={{ background: g.color }} />
                <span className="value">{g.label}</span>
              </span>
            ))}
          </div>
          <span className={`caption ${styles.muted}`}>Shades blend between the five steps.</span>
        </section>
        <section className={styles.panel} aria-label="This view">
          <h2 className={`label ${styles.muted}`}>This view</h2>
          <span className="small">
            An <code className="value">inspect</code> job
            {report ? (
              <>
                , from the step {report.step} checkpoint
              </>
            ) : null}
            . It loads your model class, so a worker runs it, never the server.
          </span>
        </section>
        <EquivalentCommand cli={command.cli} python={command.python} />
      </aside>
    </div>
  );
}
