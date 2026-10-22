import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useLevel } from "../app/level";
import { Button, ButtonLink } from "../components/Button";
import { Editor } from "../components/Editor";
import { NewModelDialog } from "../components/NewModelDialog";
import type { Marker } from "../editor/types";
import { useFileDoc } from "../editor/useFileBuffer";
import { hintsFromChecks, useStructure } from "../graph/structure";
import { StructureView } from "../graph/StructureView";
import { shows } from "../levels";
import { LessonBar, type useLessonOfFile } from "./LessonBar";
import styles from "./Model.module.css";

type Props = {
  path: string;
  etag: string | null;
  lesson: ReturnType<typeof useLessonOfFile>;
  markers: Marker[];
};

type ModelRef = { name: string; source?: string | null };

// The class the page opens on: the one asked for, else the lesson's, else the last in the file
// (a file usually defines its parts before the model that uses them).
function chooseClass(names: string[], asked: string | null, lessonClass: string | null): string | null {
  if (asked && names.includes(asked)) return asked;
  if (lessonClass && names.includes(lessonClass)) return lessonClass;
  return names.at(-1) ?? null;
}

// A plain PyTorch file: the code beside a read-only structure of what it builds. The graph you
// edit by dragging needs a model made of blocks; this page says how to get one.
export function PlainModel({ path, etag, lesson, markers }: Props) {
  const level = useLevel();
  const queryClient = useQueryClient();
  const [search, setSearch] = useSearchParams();
  const [creating, setCreating] = useState(false);
  const [reveal, setReveal] = useState<number | null>(null);
  const models = useQuery({ queryKey: ["models"], queryFn: () => unwrap(api.GET("/api/models")) as unknown as Promise<ModelRef[]> });
  // a save can add or rename a class: read the list again for each version of the file
  useEffect(() => {
    if (etag) void queryClient.invalidateQueries({ queryKey: ["models"] });
  }, [etag, queryClient]);
  const names = (models.data ?? []).filter((m) => m.source === path).map((m) => m.name);
  const lessonClass = lesson.detail?.checks.find((c) => c.class)?.class ?? null;
  const cls = chooseClass(names, search.get("class"), lessonClass);
  const structure = useStructure(path, cls, etag, hintsFromChecks(lesson.detail?.checks, cls ?? ""));
  const source = useFileDoc(path).data?.content ?? "";
  const editor = shows("editor", level) || Boolean(lesson.ids);

  const editInCode = (line: number) => {
    setReveal(null);
    queueMicrotask(() => setReveal(line));
  };

  const footer = lesson.ids ? (
    <>
      <p className={`small ${styles.muted}`}>This lesson is written in plain PyTorch, so there is no graph to edit: you change the structure in the code.</p>
      {lesson.detail?.template && (
        <div>
          <ButtonLink size="sm" to={`/learn/${lesson.id}`}>
            Build it from blocks on the lesson page
          </ButtonLink>
        </div>
      )}
    </>
  ) : (
    <>
      <p className={`small ${styles.muted}`}>A model built from blocks also gets a graph, where you swap blocks by dragging.</p>
      <div>
        <Button size="sm" onClick={() => setCreating(true)}>
          Start a model from blocks
        </Button>
      </div>
    </>
  );

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className="title">{cls ?? path}</h1>
        {names.length > 1 && (
          <select className={`${styles.select} body`} aria-label="Class" value={cls ?? undefined} onChange={(e) => setSearch({ class: e.target.value })}>
            {names.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        )}
        <span className={`value ${styles.muted}`}>{path}</span>
      </header>
      {lesson.ids ? <LessonBar file={path} lesson={lesson} /> : <div />}
      <div className={styles.plainBody}>
        {editor ? (
          <Editor path={path} markers={markers} revealLine={reveal} />
        ) : (
          <pre className={`code ${styles.plainCode}`} aria-label="Your file">
            {source}
          </pre>
        )}
        <StructureView structure={structure} className={cls} onEditInCode={editor ? editInCode : undefined} footer={footer} />
      </div>
      <NewModelDialog open={creating} onOpenChange={setCreating} />
    </div>
  );
}
