import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button, ButtonLink } from "../components/Button";
import { CheckPanel } from "../components/CheckPanel";
import { ProblemFromError } from "../components/ProblemView";
import { Play } from "../icons";
import { RUN_DEFAULTS } from "../runDefaults";
import { useLessonCheck } from "../useLessonCheck";
import styles from "./LessonBar.module.css";

// A file in a lesson's workspace is `lessons/<path>/<lesson>/<file>`.
export function lessonOfFile(file: string): { path: string; lesson: string } | null {
  const m = /^lessons\/([^/]+)\/([^/]+)\//.exec(file);
  return m ? { path: m[1]!, lesson: m[2]! } : null;
}

type Detail = { id: string; title: string; unlocks: string[]; checks: { id: string; kind: string; class?: string }[]; experiment?: { kind?: string; model?: string } };

// What the model page knows about the lesson its file belongs to: the lesson, its check job and
// the checks as the lesson declares them (so a failed one can be put on its class).
export function useLessonOfFile(file: string) {
  const ids = lessonOfFile(file);
  const id = ids ? `${ids.path}/${ids.lesson}` : "";
  const detail = useQuery({
    queryKey: ["curricula", id],
    enabled: ids !== null,
    queryFn: () => unwrap(api.GET("/api/curricula/{path}/{lesson}", { params: { path: { path: ids!.path, lesson: ids!.lesson } } })) as unknown as Promise<Detail>,
  });
  const check = useLessonCheck(id);
  return { ids, id, detail: detail.data, check };
}

type Props = { file: string; lesson: ReturnType<typeof useLessonOfFile> };

// Train and check the lesson from where you edit its file. Same actions as the lesson page.
export function LessonBar({ file, lesson }: Props) {
  const navigate = useNavigate();
  const { ids, id, detail, check } = lesson;
  const train = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/runs", { body: { ...RUN_DEFAULTS, kwargs: {}, model: `${file}:${detail?.experiment?.model ?? ""}` } })) as Promise<{ ref: string }>,
    onSuccess: (run) => navigate(`/runs/${run.ref}`),
  });
  if (!ids || !detail) return null;
  const busy = check.run.isPending || check.job?.state === "queued" || check.job?.state === "running";
  return (
    <section className={styles.bar} aria-label="Lesson">
      <div className={styles.row}>
        <ButtonLink size="sm" variant="quiet" to={`/learn/${id}`}>
          {detail.title}
        </ButtonLink>
        <span className={styles.spacer} />
        {detail.experiment?.kind === "run" && (
          <Button size="sm" disabled={train.isPending} onClick={() => train.mutate()}>
            <Play size={16} aria-hidden="true" />
            Train
          </Button>
        )}
        <Button size="sm" variant="primary" disabled={busy} onClick={() => check.run.mutate()}>
          <Play size={16} aria-hidden="true" />
          Run the check
        </Button>
      </div>
      {train.error && <ProblemFromError error={train.error} />}
      {check.run.error && <ProblemFromError error={check.run.error} />}
      {check.job && (
        <CheckPanel lessonTitle={detail.title} job={check.job} result={check.result} checkDefs={detail.checks.map((c) => ({ id: String(c.id), kind: String(c.kind) }))} unlocks={detail.unlocks} />
      )}
    </section>
  );
}
