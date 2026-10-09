import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { EmptyState } from "../components/EmptyState";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { Markdown } from "../components/Markdown";
import { ProblemFromError } from "../components/ProblemView";
import { Tabs } from "../components/Tabs";
import { Tag } from "../components/Tag";
import { Check, Play, Warning, X } from "../icons";
import { followJob } from "../studies/useStudyChecks";
import { stampText } from "../format";
import styles from "./Authoring.module.css";

type Outcome = { passed: boolean; skipped: boolean; reason: string };
type Result = {
  state: string;
  summary: string;
  at: string;
  variant: string;
  seconds: number;
  fresh: boolean;
  checks: { id: string; kind: string; starter: Outcome; solution: Outcome }[];
};
type Problem = { where: string; message: string; hint?: string | null };
type Lesson = {
  title: string;
  summary: string;
  level: number;
  prerequisites: string[];
  unlocks: string[];
  experiment: Record<string, unknown>;
  compute: Record<string, { preset: string | null; budget: string | null; estimate_minutes: number }>;
  text: { intro: string; surface: string; deep: string; reading: string };
};
type Doc = {
  folder: string;
  id: string;
  files: Record<string, boolean>;
  loads: boolean;
  problems: Problem[];
  lesson: Lesson | null;
  result: Result | null;
};

const TONE: Record<string, "good" | "bad" | "warn"> = {
  ready: "good",
  "does not load": "bad",
  "solution fails": "bad",
  "starter passes": "warn",
  "files missing": "warn",
};

function Outcome({ o }: { o: Outcome }) {
  return (
    <span className={styles.outcome}>
      {o.passed ? <Check size={14} className={styles.good} aria-label="passed" /> : <X size={14} className={styles.bad} aria-label={o.skipped ? "skipped" : "failed"} />}
      <span>{o.reason}</span>
    </span>
  );
}

// A lesson folder an author is writing: whether the loader takes it, what a check of it found
// (starter.py should fail, solution.py should pass), and the lesson as a learner would read it.
export function Authoring() {
  const folder = useParams()["*"] ?? "";
  const queryClient = useQueryClient();
  const doc = useQuery({
    queryKey: ["authoring", folder],
    queryFn: () => unwrap(api.GET("/api/authoring/{folder}", { params: { path: { folder } } })) as unknown as Promise<Doc>,
  });
  const check = useMutation({
    mutationFn: async () => {
      const job = await unwrap(api.POST("/api/authoring/{folder}/check", { params: { path: { folder } }, body: { variant: "cpu" } }));
      return followJob<unknown>((job as unknown as { id: number }).id);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["authoring", folder] }),
  });

  if (doc.error) return <ProblemFromError error={doc.error} />;
  if (!doc.data) return <p className="small">Loading the lesson folder…</p>;
  const d = doc.data;
  const result = d.result;
  const name = d.folder.split("/").slice(0, -1).join("/");
  const command = `nanoscope learn author-check ${d.folder}`;
  const again = d.result !== null;
  const lesson = d.lesson;

  return (
    <div className={styles.page}>
      <div className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            <Link to="/workspace" className={styles.link}>
              Workspace
            </Link>{" "}
            / <code className="value">{name}</code>
          </span>
          <div className={styles.titleRow}>
            {lesson ? (
              <>
                <h1 className="title">{lesson.title}</h1>
                <code className={`value ${styles.muted}`}>{d.id}</code>
              </>
            ) : (
              <h1 className="title">
                <code className="value-strong">{d.folder.split("/").at(-1)}</code>
              </h1>
            )}
            <span className={styles.spacer} />
            <Button variant="primary" disabled={check.isPending} onClick={() => check.mutate()}>
              {again ? null : <Play size={16} aria-hidden="true" />}
              {check.isPending ? "Checking…" : again ? "Check the lesson again" : "Check the lesson"}
            </Button>
          </div>
          <span className={`small ${styles.muted}`}>
            {lesson ? (
              <>
                Lesson authoring preview. A check loads the folder as the curriculum loader does, then runs every check twice on a worker: on <code className="value">starter.py</code>, which must fail, and on{" "}
                <code className="value">solution.py</code>, which must pass.
              </>
            ) : (
              <>
                The title comes from <code className="value">lesson.toml</code>, which does not load yet, so the folder&apos;s name stands in.
              </>
            )}
          </span>
        </header>

        {check.error && <ProblemFromError error={check.error} />}

        {!d.loads && (
          <section className={styles.block} aria-label="Problems">
            <div className={styles.verdict}>
              <Tag tone="bad" icon={X}>
                does not load
              </Tag>
              <span className="body">
                The curriculum loader found {d.problems.length} {d.problems.length === 1 ? "problem" : "problems"}. Every one is listed; fix them and check again.
              </span>
            </div>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className="label" scope="col">
                    Where
                  </th>
                  <th className="label" scope="col">
                    Problem
                  </th>
                </tr>
              </thead>
              <tbody>
                {d.problems.map((p) => (
                  <tr key={`${p.where}-${p.message}`}>
                    <td>
                      <code className="value">{p.where}</code>
                    </td>
                    <td className="small">
                      {p.message}
                      {p.hint && <span className={styles.muted}> {p.hint}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <EmptyState title="No preview and no check runs yet" body="The lesson page and the starter and solution runs need a folder that loads. They appear here after the next check." />
          </section>
        )}

        {d.loads && result === null && !check.isPending && (
          <EmptyState title="Not checked yet" body="A check runs every check on starter.py and on solution.py and says whether the lesson is ready. Nothing runs until you ask." />
        )}

        {d.loads && result && (
          <section className={styles.panel} aria-label="Check result">
            <div className={styles.verdict}>
              <Tag tone={TONE[result.state] ?? "warn"} icon={result.state === "ready" ? Check : result.state === "does not load" || result.state === "solution fails" ? X : Warning}>
                {result.state}
              </Tag>
              <span className="body">{result.summary}</span>
            </div>
            <span className={`small ${styles.muted}`}>
              Checked <code className="value">{stampText(result.at)}</code> on <code className="value">{result.variant}</code> · {result.checks.length * 2} check runs, {result.seconds} s in all ·{" "}
              {result.fresh ? "the files have not changed since" : <span className={styles.warn}>the files changed since: check again</span>}
            </span>
            {result.checks.length > 0 && (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th className="label" scope="col">
                      Check
                    </th>
                    <th className="label" scope="col">
                      Kind
                    </th>
                    <th className="label" scope="col">
                      On starter.py (must fail)
                    </th>
                    <th className="label" scope="col">
                      On solution.py (must pass)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.checks.map((c) => (
                    <tr key={c.id}>
                      <td>
                        <code className="value">{c.id}</code>
                      </td>
                      <td className={`small ${styles.muted}`}>{c.kind}</td>
                      <td className="small">
                        <Outcome o={c.starter} />
                      </td>
                      <td className="small">
                        <Outcome o={c.solution} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <span className={`small ${styles.muted}`}>A failing starter is expected: it is what the learner fills in. The starter&apos;s failure reasons are what a learner will read first, so read them as a learner would.</span>
          </section>
        )}

        {lesson && (
          <section className={styles.panel} aria-label="As a learner sees it">
            <div className={styles.sectionHead}>
              <h2 className="heading">As a learner sees it</h2>
              <span className={`small ${styles.muted}`}>
                rendered from <code className="value">lesson.md</code>; Start and Check are off in a preview
              </span>
            </div>
            <div className={styles.preview}>
              <div className={styles.sectionHead}>
                <span className="heading">{lesson.title}</span>
                <span className={`small ${styles.muted}`}>{lesson.summary}</span>
              </div>
              {lesson.text.intro.trim() && <Markdown>{lesson.text.intro}</Markdown>}
              <Tabs
                items={[
                  { id: "surface", label: "Surface", content: <Markdown>{lesson.text.surface}</Markdown> },
                  { id: "deep", label: "Deep", content: <Markdown>{lesson.text.deep}</Markdown> },
                  { id: "reading", label: "Reading", content: <Markdown>{lesson.text.reading}</Markdown> },
                ]}
              />
              <div className={styles.actions}>
                <Button variant="primary" disabled>
                  Start
                </Button>
                <Button disabled>Run the check</Button>
                {lesson.compute["cpu"] && <span className={`caption ${styles.muted}`}>cpu {lesson.compute["cpu"].estimate_minutes} min</span>}
              </div>
            </div>
          </section>
        )}
      </div>

      <aside className={styles.aside}>
        <section className={styles.panel} aria-label="Lesson folder">
          <h2 className={`label ${styles.muted}`}>Lesson folder</h2>
          <code className={`value ${styles.wrap}`}>{d.folder}/</code>
          <ul className={styles.files}>
            {Object.entries(d.files).map(([file, present]) => (
              <li key={file} className="small">
                {present && file.endsWith(".py") ? (
                  <Link to={`/model/${d.folder}/${file}`} className={styles.link}>
                    <code className="value">{file}</code>
                  </Link>
                ) : (
                  <code className={`value ${styles.muted}`}>{file}</code>
                )}
                <span className={styles.muted}>{FILE_ROLE[file]}{present ? "" : " (absent)"}</span>
              </li>
            ))}
          </ul>
        </section>
        {lesson && (
          <section className={styles.panel} aria-label="From lesson.toml">
            <h2 className={`label ${styles.muted}`}>From lesson.toml</h2>
            <dl className={`small ${styles.facts}`}>
              <dt className={styles.muted}>Level</dt>
              <dd>
                <code className="value">{lesson.level}</code>
              </dd>
              <dt className={styles.muted}>Needs</dt>
              <dd>{lesson.prerequisites.length ? lesson.prerequisites.map((p) => <code key={p} className="value">{p} </code>) : <span className={styles.muted}>nothing</span>}</dd>
              <dt className={styles.muted}>Experiment</dt>
              <dd>
                <code className="value">{String(lesson.experiment["kind"] ?? "")}</code>
              </dd>
              <dt className={styles.muted}>Compute</dt>
              <dd>
                {Object.entries(lesson.compute).map(([variant, c]) => (
                  <span key={variant}>
                    <code className="value">
                      {variant} {c.estimate_minutes} min
                    </code>{" "}
                    <span className={styles.muted}>{c.preset ?? c.budget}</span>
                  </span>
                ))}
              </dd>
              <dt className={styles.muted}>Unlocks</dt>
              <dd>{lesson.unlocks.length ? lesson.unlocks.map((u) => <code key={u} className="value">{u} </code>) : <span className={styles.muted}>nothing</span>}</dd>
            </dl>
          </section>
        )}
        <EquivalentCommand cli={command} />
      </aside>
    </div>
  );
}

const FILE_ROLE: Record<string, string> = {
  "lesson.toml": "checks, compute, unlocks",
  "lesson.md": "Surface, Deep, Reading",
  "starter.py": "copied to the learner",
  "solution.py": "never copied",
  "notebook.py": "optional marimo notebook",
};
