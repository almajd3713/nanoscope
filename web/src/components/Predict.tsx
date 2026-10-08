import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "./Button";
import { ProblemFromError } from "./ProblemView";
import styles from "./Predict.module.css";

const VERDICTS = ["better", "worse", "within noise"] as const;

type Progress = {
  lessons: Record<string, { first_checked_at?: string | null }>;
  predictions?: Record<string, { at: string }>;
};

// Commit to what you expect before the experiment runs (nanoscope learn predict). The
// prediction cannot be changed afterwards, and it only counts if it comes before the first check.
export function Predict({ lessonId }: { lessonId: string }) {
  const [path = "", lesson = ""] = lessonId.split("/");
  const queryClient = useQueryClient();
  const progress = useQuery({
    queryKey: ["learn", "progress"],
    queryFn: () => unwrap(api.GET("/api/learn/progress")) as Promise<Progress>,
  });
  const [verdict, setVerdict] = useState<string>("");
  const [low, setLow] = useState("");
  const [high, setHigh] = useState("");
  const [note, setNote] = useState("");

  const record = useMutation({
    mutationFn: () => {
      const body: { verdict?: string; low?: number; high?: number; note?: string } = {};
      if (verdict) body.verdict = verdict;
      if (low !== "") body.low = Number(low);
      if (high !== "") body.high = Number(high);
      if (note.trim()) body.note = note.trim();
      return unwrap(api.POST("/api/curricula/{path}/{lesson}/predict", { params: { path: { path, lesson } }, body }));
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["learn", "progress"] }),
  });

  if (!progress.data) return null;
  const recorded = progress.data.predictions?.[lessonId];
  const checkedAt = progress.data.lessons[lessonId]?.first_checked_at;

  if (recorded) {
    return (
      <section className={styles.predict} aria-label="Prediction">
        <h2 className="heading">Prediction recorded</h2>
        <p className={`small ${styles.muted}`}>
          Recorded at <span className="value">{recorded.at}</span>. It can't be changed now; the check scores it.
        </p>
      </section>
    );
  }
  if (checkedAt) {
    return (
      <section className={styles.predict} aria-label="Prediction">
        <h2 className="heading">Prediction</h2>
        <p className={`small ${styles.muted}`}>
          This lesson was first checked at <span className="value">{checkedAt}</span>. A prediction only counts if it
          is recorded before the experiment runs, so it is too late for this lesson.
        </p>
      </section>
    );
  }

  return (
    <section className={styles.predict} aria-label="Prediction">
      <h2 className="heading">Predict before you run</h2>
      <p className={`small ${styles.muted}`}>
        Say what you expect first. The check compares it with what the experiment finds.
      </p>
      <form
        className={styles.predict}
       
        onSubmit={(e) => {
          e.preventDefault();
          record.mutate();
        }}
      >
        <fieldset className={styles.verdicts} aria-label="Verdict">
          {VERDICTS.map((v) => (
            <label key={v} className="body">
              <input type="radio" name="verdict" value={v} checked={verdict === v} onChange={() => setVerdict(v)} />
              {v}
            </label>
          ))}
        </fieldset>
        <div className={styles.row}>
          <label className={styles.field}>
            <span className="label">Low end of the difference</span>
            <input className={`${styles.input} value`} inputMode="decimal" value={low} onChange={(e) => setLow(e.target.value)} />
          </label>
          <label className={styles.field}>
            <span className="label">High end of the difference</span>
            <input className={`${styles.input} value`} inputMode="decimal" value={high} onChange={(e) => setHigh(e.target.value)} />
          </label>
        </div>
        <label className={styles.field}>
          <span className="label">Why you expect it</span>
          <textarea className={`${styles.input} ${styles.note} body`} rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
        </label>
        {record.error && <ProblemFromError error={record.error} />}
        <div>
          <Button type="submit" disabled={record.isPending}>
            Record prediction
          </Button>
        </div>
      </form>
    </section>
  );
}
