import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button, ButtonLink } from "../components/Button";
import { EmptyState } from "../components/EmptyState";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { useGitStatus } from "../components/GitStatus";
import { ProblemFromError } from "../components/ProblemView";
import { stampText } from "../format";
import { useEvents } from "../hooks/useEvents";
import { CaretDown, CaretRight, CopySimple, PencilSimple } from "../icons";
import { rows, sizeText, type FileEntry, type LessonFolder, type ModelRef, type Row, type StudyRef, type UserBlock } from "../workspace/tree";
import styles from "./Workspace.module.css";

const KEY = "nanoscope.workspace.closed";

function readClosed(): Set<string> {
  try {
    const v = JSON.parse(window.localStorage.getItem(KEY) ?? "[]");
    return new Set(Array.isArray(v) ? (v as string[]) : []);
  } catch {
    return new Set();
  }
}

function writeClosed(closed: Set<string>) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify([...closed]));
  } catch {
    /* folders then reopen on reload */
  }
}

function Preview({ path }: { path: string }) {
  const file = useQuery({
    queryKey: ["file", path],
    queryFn: () => unwrap(api.GET("/api/files/{path}", { params: { path: { path } } })) as unknown as Promise<{ content: string }>,
    retry: false,
  });
  if (file.error) return <ProblemFromError error={file.error} />;
  const lines = (file.data?.content ?? "").split("\n").slice(0, 40);
  return (
    <pre className={`code-small ${styles.code}`} aria-label={`${path} (first lines)`}>
      {lines.map((line, i) => (
        <span key={i} className={styles.line}>
          <span className={styles.lineNo}>{i + 1}</span>
          {line}
          {"\n"}
        </span>
      ))}
    </pre>
  );
}

// Every file the server can read and save, one row each, with what nanoscope finds in it and the
// page that works on it. Read only: the model page and the editor change files.
export function Workspace() {
  const queryClient = useQueryClient();
  const files = useQuery({ queryKey: ["files"], queryFn: () => unwrap(api.GET("/api/files")) as unknown as Promise<FileEntry[]> });
  const models = useQuery({ queryKey: ["models"], queryFn: () => unwrap(api.GET("/api/models")) as unknown as Promise<ModelRef[]> });
  const blocks = useQuery({ queryKey: ["blocks"], queryFn: () => unwrap(api.GET("/api/blocks")) as unknown as Promise<{ blocks: UserBlock[] }> });
  const studies = useQuery({ queryKey: ["studies"], queryFn: () => unwrap(api.GET("/api/studies")) as unknown as Promise<StudyRef[]> });
  const lessons = useQuery({ queryKey: ["authoring"], queryFn: () => unwrap(api.GET("/api/authoring")) as unknown as Promise<LessonFolder[]> });
  const git = useGitStatus(null);
  const [closed, setClosed] = useState(readClosed);
  const [selected, setSelected] = useState<string | null>(null);

  const refetch = useCallback(() => {
    for (const key of ["files", "models", "blocks", "studies", "authoring", "git"]) void queryClient.invalidateQueries({ queryKey: [key] });
  }, [queryClient]);
  useEvents("/api/files/events", { events: ["change"], onEvent: refetch });

  const list = useMemo(
    () =>
      rows(
        {
          files: files.data ?? [],
          models: models.data ?? [],
          blocks: blocks.data?.blocks ?? [],
          studies: studies.data ?? [],
          lessons: lessons.data ?? [],
          changed: git.data?.changed ?? [],
        },
        closed,
      ),
    [files.data, models.data, blocks.data, studies.data, lessons.data, git.data, closed],
  );
  const current: Row | undefined = list.find((r) => r.path === selected && !r.folder);

  const toggle = (path: string) =>
    setClosed((prev) => {
      const next = new Set(prev);
      if (!next.delete(path)) next.add(path);
      writeClosed(next);
      return next;
    });

  const g = git.data;
  const changedCount = g?.changed.length ?? 0;

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <div className={styles.titleRow}>
          <h1 className="title">Workspace</h1>
          <span className={styles.spacer} />
          {g?.repo && (
            <span className="small">
              <span className={styles.muted}>git</span> <code className="value">{g.branch}</code> <span className={styles.muted}>at</span> <code className="value">{g.head?.slice(0, 7)}</code>{" "}
              {changedCount > 0 && (
                <span className={styles.warn}>
                  <PencilSimple size={14} aria-hidden="true" /> {changedCount} {changedCount === 1 ? "file" : "files"} changed
                </span>
              )}
            </span>
          )}
        </div>
        <span className={`small ${styles.muted}`}>Every file the server can read and save. Each row says what nanoscope finds in it and links to the page that works on it. Live: an edit in your own editor shows here at once.</span>
      </header>

      {files.error && <ProblemFromError error={files.error} />}
      {files.data && files.data.length === 0 && (
        <EmptyState title="The workspace is empty" body="Models, blocks, studies and lessons you write appear here. Start a lesson, or create a model on the Models page." action={<ButtonLink to="/models">Models</ButtonLink>} />
      )}

      {list.length > 0 && (
        <div className={styles.cols}>
          <section aria-label="Files" className={styles.tree}>
            <div className={`label ${styles.headRow}`}>
              <span>Name</span>
              <span>What nanoscope sees</span>
              <span className={styles.right}>Size</span>
              <span>Modified</span>
            </div>
            {list.map((row) => (
              <div key={row.key} className={`${styles.row} ${row.path === selected && !row.folder ? styles.rowOn : ""}`}>
                <span className={styles.name} style={{ paddingLeft: row.folder ? row.depth * 18 : row.depth * 18 + 20 }}>
                  {row.folder ? (
                    <button type="button" className={styles.fold} aria-expanded={!closed.has(row.path)} aria-label={row.path} onClick={() => toggle(row.path)}>
                      {closed.has(row.path) ? <CaretRight size={14} aria-hidden="true" /> : <CaretDown size={14} aria-hidden="true" />}
                      <span className="body-strong">{row.name}</span>
                    </button>
                  ) : (
                    <button type="button" className={styles.pick} onClick={() => setSelected(row.path)}>
                      <code className="value">{row.name}</code>
                    </button>
                  )}
                  {row.changed && (
                    <span className={`caption ${styles.warn}`}>
                      <PencilSimple size={14} aria-hidden="true" /> changed
                    </span>
                  )}
                </span>
                <span className={`small ${styles.sees}`}>
                  <span className={styles.muted}>{row.seen.kind}</span>
                  {row.seen.link &&
                    (row.seen.to ? (
                      <Link to={row.seen.to} className={styles.link}>
                        {row.seen.link}
                      </Link>
                    ) : (
                      <span>{row.seen.link}</span>
                    ))}
                  {row.seen.note && <span className={styles.muted}>{row.seen.note}</span>}
                </span>
                <span className={`value ${styles.muted} ${styles.right}`}>{row.file ? sizeText(row.file.size) : ""}</span>
                <span className={`value ${styles.muted}`}>{row.file ? stampText(new Date(row.file.modified * 1000).toISOString()).slice(5) : ""}</span>
              </div>
            ))}
            <span className={`caption ${styles.muted} ${styles.foot}`}>
              Hidden: <code className="value">.git</code>, <code className="value">__pycache__</code>, <code className="value">.venv</code>, <code className="value">node_modules</code>, <code className="value">.ipynb_checkpoints</code>. Folders stay open as you leave them.
            </span>
          </section>

          <aside aria-label="Selected file" className={styles.aside}>
            {current ? (
              <section className={styles.panel}>
                <div className={styles.titleRow}>
                  <code className="value-strong">{current.path}</code>
                  {current.changed && g?.head && (
                    <span className={`caption ${styles.warn}`}>
                      <PencilSimple size={14} aria-hidden="true" /> changed since {g.head.slice(0, 7)}
                    </span>
                  )}
                </div>
                <dl className={`small ${styles.facts}`}>
                  <dt className={styles.muted}>Holds</dt>
                  <dd>
                    {current.seen.kind ? (
                      <>
                        {current.seen.kind}
                        {current.seen.link && <> <code className="value">{current.seen.link}</code></>}
                      </>
                    ) : (
                      <span className={styles.muted}>nothing nanoscope reads</span>
                    )}
                  </dd>
                  <dt className={styles.muted}>Size</dt>
                  <dd>
                    <code className="value">{sizeText(current.file!.size)}</code> · modified <code className="value">{stampText(new Date(current.file!.modified * 1000).toISOString())}</code>
                  </dd>
                </dl>
                <Preview path={current.path} />
                <div className={styles.actions}>
                  {current.seen.to && (
                    <ButtonLink variant="primary" to={current.seen.to}>
                      <PencilSimple size={16} aria-hidden="true" />
                      Open on the model page
                    </ButtonLink>
                  )}
                  <Button variant="quiet" onClick={() => void navigator.clipboard?.writeText(current.path)}>
                    <CopySimple size={16} aria-hidden="true" />
                    Copy path
                  </Button>
                </div>
                <span className={`caption ${styles.muted}`}>Read only here. The model page edits it (graph and code), with the same ETag check as every save.</span>
              </section>
            ) : (
              <section className={styles.panel}>
                <span className={`small ${styles.muted}`}>Pick a file to see what nanoscope finds in it.</span>
              </section>
            )}
            <EquivalentCommand cli={"git -C <workspace> status --short\nnanoscope blocks"} />
          </aside>
        </div>
      )}
    </div>
  );
}
