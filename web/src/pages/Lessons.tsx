import { useQuery } from "@tanstack/react-query";
import { Link, Navigate } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { LessonRow } from "../components/LessonRow";
import { ProblemFromError } from "../components/ProblemView";
import { computeText } from "../format";
import styles from "./Lessons.module.css";

export function Lessons() {
  const unlocks = useQuery({ queryKey: ["learn", "unlocks"], queryFn: () => unwrap(api.GET("/api/learn/unlocks")) });
  const curricula = useQuery({ queryKey: ["curricula"], queryFn: () => unwrap(api.GET("/api/curricula")) });

  // Nobody has chosen guided or open yet: ask once.
  // (not while a refetch is under way: a cached "first run" may be about to change)
  if (unlocks.data?.first_run && !unlocks.isFetching) return <Navigate to="/welcome" replace />;

  const error = unlocks.error ?? curricula.error;
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.titles}>
          <h1 className="title">Lessons</h1>
          <p className={`body ${styles.muted} ${styles.prose}`}>
            Build a language model from the smallest part up. Passing a lesson's check unlocks the block it teaches, so
            you can use it in your own models.
          </p>
        </div>
        {unlocks.data && (
          <p className={`small ${styles.muted} ${styles.policy}`}>
            Policy <span className="value">{unlocks.data.policy}</span>
            {unlocks.data.policy === "guided"
              ? ": blocks unlock as you pass lessons. "
              : ": every block is available. "}
            <Link to="/components" className={styles.link}>
              See what is unlocked
            </Link>
          </p>
        )}
      </header>

      {error && <ProblemFromError error={error} />}
      {!error && !curricula.data && <p className={`body ${styles.muted}`}>Loading lessons…</p>}

      {curricula.data?.map((path) => {
        const passed = path.lessons.filter((l) => l.state === "passed").length;
        return (
          <section key={path.id} className={styles.path} aria-labelledby={`path-${path.id}`}>
            <div className={styles.pathHead}>
              <h2 className="heading" id={`path-${path.id}`}>
                {path.title}
              </h2>
              <span className={`small ${styles.muted}`}>
                level {path.level} · {passed} of {path.lessons.length} passed ·{" "}
                <span className="value">{computeText(path.compute)}</span>
              </span>
            </div>
            <p className={`small ${styles.muted} ${styles.prose}`}>{path.summary}</p>
            <div className={styles.table}>
              <div className={`label ${styles.columns}`}>
                <span />
                <span>Lesson</span>
                <span>Passing unlocks</span>
                <span>Compute</span>
                <span>State</span>
              </div>
              {path.lessons.map((lesson) => (
                <LessonRow
                  key={lesson.id}
                  to={`/learn/${lesson.id}`}
                  number={lesson.slug.match(/^\d+/)?.[0] ?? ""}
                  title={lesson.title}
                  summary={lesson.summary}
                  unlocks={lesson.unlocks}
                  estimate={computeText(lesson.compute)}
                  state={lesson.state}
                  lockedBy={lesson.locked_by}
                />
              ))}
            </div>
          </section>
        );
      })}

      <EquivalentCommand cli="nanoscope learn list" />
    </div>
  );
}
