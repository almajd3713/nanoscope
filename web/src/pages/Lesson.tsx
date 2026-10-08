import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button, ButtonLink } from "../components/Button";
import { CheckResult } from "../components/CheckResult";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { LessonState } from "../components/LessonState";
import { Markdown } from "../components/Markdown";
import { Predict } from "../components/Predict";
import { ProblemFromError, ProblemView } from "../components/ProblemView";
import { Tabs } from "../components/Tabs";
import { Check, Circle, CircleHalf, LockSimple, Play, X } from "../icons";
import { computeText } from "../format";
import { RUN_DEFAULTS } from "../runDefaults";
import { useLessonCheck } from "../useLessonCheck";
import styles from "./Lesson.module.css";

function Glyph({ state, locked }: { state: string; locked: boolean }) {
  if (locked) return <LockSimple size={16} aria-label="locked" />;
  if (state === "passed") return <Check size={16} className={styles.good} aria-label="passed" />;
  if (state === "failed") return <X size={16} className={styles.bad} aria-label="failed" />;
  if (state === "started" || state === "checking") return <CircleHalf size={16} aria-label={state} />;
  return <Circle size={16} aria-label={state} />;
}

export function Lesson() {
  const { path = "", lesson = "" } = useParams();
  const id = `${path}/${lesson}`;
  const detail = useQuery({
    queryKey: ["curricula", id],
    queryFn: () => unwrap(api.GET("/api/curricula/{path}/{lesson}", { params: { path: { path, lesson } } })),
  });
  const all = useQuery({ queryKey: ["curricula"], queryFn: () => unwrap(api.GET("/api/curricula")) });

  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const startedState = detail.data?.state ?? "not-started";
  const started = startedState !== "not-started";
  const starterPath = detail.data ? `${detail.data.workspace}/starter.py` : "";
  const hasStarter = detail.data?.files.includes("starter.py") ?? false;

  const start = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/curricula/{path}/{lesson}/start", { params: { path: { path, lesson } }, body: {} })),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["curricula"] });
      await queryClient.invalidateQueries({ queryKey: ["learn"] });
      // the starter file opens on the model page, with the graph and the palette
      if (hasStarter && starterPath) navigate(`/model/${starterPath}`);
    },
  });

  const file = useQuery({
    queryKey: ["file", starterPath],
    queryFn: () => unwrap(api.GET("/api/files/{path}", { params: { path: { path: starterPath } } })),
    enabled: started && hasStarter,
    retry: false,
  });

  const check = useLessonCheck(id);
  const experiment = detail.data?.experiment as { kind?: string; model?: string } | undefined;
  const train = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/runs", {
          body: { ...RUN_DEFAULTS, kwargs: {}, model: `${starterPath}:${experiment?.model ?? ""}` },
        }),
      ) as Promise<{ ref: string }>,
    onSuccess: (run) => navigate(`/runs/${run.ref}`),
  });

  if (detail.error) return <ProblemFromError error={detail.error} />;
  if (!detail.data) return <p className={`body ${styles.muted}`}>Loading lesson…</p>;

  const d = detail.data;
  const pathInfo = all.data?.find((p) => p.id === path);
  const intro = d.text["intro"] ?? "";
  const sections = [
    { id: "surface", label: "Surface", text: d.text["surface"] ?? "" },
    { id: "deep", label: "Deep", text: d.text["deep"] ?? "" },
    { id: "reading", label: "Reading", text: d.text["reading"] ?? "" },
  ].filter((s) => s.text.trim());

  return (
    <div className={styles.page}>
      <aside className={styles.aside}>
        {pathInfo && (
          <nav className={styles.outline} aria-label={`${pathInfo.title} lessons`}>
            <Link to="/learn" className={`label ${styles.outlineTitle}`}>
              {pathInfo.title}
            </Link>
            {pathInfo.lessons.map((l) => (
              <Link
                key={l.id}
                to={`/learn/${l.id}`}
                className={`small ${styles.item}`}
                aria-current={l.id === d.id ? "page" : undefined}
              >
                <Glyph state={l.state} locked={l.state === "not-started" && l.locked_by.length > 0} />
                <span className={`value ${styles.muted}`}>{l.slug.match(/^\d+/)?.[0]}</span>
                {l.title}
              </Link>
            ))}
          </nav>
        )}
        <dl className={styles.facts}>
          <div className={styles.fact}>
            <dt className={`label ${styles.muted}`}>Passing unlocks</dt>
            <dd className="value">{d.unlocks.length ? d.unlocks.join(", ") : "-"}</dd>
          </div>
          <div className={styles.fact}>
            <dt className={`label ${styles.muted}`}>Compute</dt>
            <dd className="value">{computeText(d.compute)}</dd>
          </div>
          <div className={styles.fact}>
            <dt className={`label ${styles.muted}`}>Checks</dt>
            <dd className="value">{d.checks.map((c) => c.kind).join(", ")}</dd>
          </div>
          {d.forbid.length > 0 && (
            <div className={styles.fact}>
              <dt className={`label ${styles.muted}`}>Not allowed</dt>
              <dd className="value">{d.forbid.join(", ")}</dd>
            </div>
          )}
        </dl>
      </aside>

      <div className={styles.main}>
        <header className={styles.head}>
          <span className={`small ${styles.muted}`}>
            <Link to="/learn" className={styles.link}>
              {pathInfo?.title ?? path}
            </Link>{" "}
            / <span className="value">{d.slug}</span>
          </span>
          <div className={styles.titleRow}>
            <h1 className="title">{d.title}</h1>
            <LessonState state={d.state} lockedBy={d.locked_by} />
          </div>
          {d.locked_by.length > 0 && d.state === "not-started" && (
            <p className={`small ${styles.muted}`}>
              Suggested first: <span className="value">{d.locked_by.join(", ")}</span>. The lock is advice; you can
              start this lesson now.
            </p>
          )}
          {intro.trim() && <Markdown>{intro}</Markdown>}
        </header>

        <div className={styles.actions}>
          {!started && (
            <Button variant="primary" size="lg" disabled={start.isPending} onClick={() => start.mutate()}>
              Start lesson
            </Button>
          )}
          {started && (
            <Button variant="primary" size="lg" disabled={check.run.isPending || check.job?.state === "queued" || check.job?.state === "running"} onClick={() => check.run.mutate()}>
              <Play size={16} aria-hidden="true" />
              Run the check
            </Button>
          )}
          {experiment?.kind === "run" && (
            <Button disabled={!started || train.isPending} onClick={() => train.mutate()}>
              <Play size={16} aria-hidden="true" />
              Train
            </Button>
          )}
          {started && hasStarter && (
            <span className={`small ${styles.muted}`}>
              Your file: <span className="value">workspace/{starterPath}</span>
            </span>
          )}
        </div>
        {start.error && <ProblemFromError error={start.error} />}
        {train.error && <ProblemFromError error={train.error} />}
        {check.run.error && <ProblemFromError error={check.run.error} />}
        {started && d.checks.some((c) => c["kind"] === "predicted") && <Predict lessonId={d.id} />}
        {check.job && <CheckPanel lessonTitle={d.title} job={check.job} result={check.result} checkDefs={d.checks.map((c) => ({ id: String(c["id"]), kind: String(c["kind"]) }))} unlocks={d.unlocks} />}

        <Tabs items={sections.map((s) => ({ id: s.id, label: s.label, content: <Markdown>{s.text}</Markdown> }))} />

        {started && hasStarter && (
          <section className={styles.file} aria-label="Your file">
            <header className={styles.fileHead}>
              <h2 className="heading">Your file</h2>
              <span className={`value ${styles.muted}`}>starter.py</span>
              <span className={styles.spacer} />
              <ButtonLink size="sm" to={`/model/${starterPath}`}>
                Open in the model page
              </ButtonLink>
            </header>
            {file.error ? (
              <ProblemFromError error={file.error} />
            ) : (
              <pre className={`${styles.code} code`}>{file.data?.content ?? "Loading file…"}</pre>
            )}
          </section>
        )}

        <EquivalentCommand cli={`nanoscope learn start ${d.id}\nnanoscope learn check ${d.id}`} />
      </div>
    </div>
  );
}

type JobView = NonNullable<ReturnType<typeof useLessonCheck>["job"]>;

function CheckPanel({ lessonTitle, job, result, checkDefs, unlocks }: {
  lessonTitle: string;
  job: JobView;
  result: ReturnType<typeof useLessonCheck>["result"];
  checkDefs: { id: string; kind: string }[];
  unlocks: string[];
}) {
  const kinds = new Map(checkDefs.map((c) => [c.id, c.kind]));
  if (job.state === "queued" || job.state === "running") {
    return (
      <CheckResult
        status="running"
        title={lessonTitle}
        checks={checkDefs.map((c) => ({ id: c.id, kind: c.kind, passed: null, reason: "not run yet" }))}
        progress={job.state === "queued" ? "Waiting for a worker to start the check" : "Running the checks"}
      />
    );
  }
  if (job.state === "done" && result) {
    return (
      <CheckResult
        status={result.passed ? "passed" : "failed"}
        title={lessonTitle}
        checks={result.checks.map((c) => ({ ...c, kind: kinds.get(c.id) ?? "" }))}
      >
        {result.passed
          ? unlocks.length
            ? `Check passed. ${unlocks.map((u) => u.split(":").pop()).join(", ")} ${unlocks.length === 1 ? "is" : "are"} unlocked.`
            : "Check passed."
          : "Read the reasons, edit your file, and run the check again."}
      </CheckResult>
    );
  }
  return (
    <ProblemView
      title={job.state === "cancelled" ? "The check was cancelled" : "The check did not finish"}
      detail={job.error ?? ""}
    />
  );
}
