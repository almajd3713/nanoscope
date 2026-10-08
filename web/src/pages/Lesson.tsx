import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { LessonState } from "../components/LessonState";
import { Markdown } from "../components/Markdown";
import { ProblemFromError } from "../components/ProblemView";
import { Tabs } from "../components/Tabs";
import { Check, Circle, CircleHalf, LockSimple, X } from "../icons";
import { computeText } from "../format";
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

        <Tabs items={sections.map((s) => ({ id: s.id, label: s.label, content: <Markdown>{s.text}</Markdown> }))} />

        <EquivalentCommand cli={`nanoscope learn start ${d.id}\nnanoscope learn check ${d.id}`} />
      </div>
    </div>
  );
}
