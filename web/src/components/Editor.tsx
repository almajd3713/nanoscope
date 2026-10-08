import { Button } from "./Button";
import { ProblemFromError } from "./ProblemView";
import { Tag } from "./Tag";
import { CodeSurface } from "../editor/CodeSurface";
import type { Marker } from "../editor/types";
import { useFileBuffer } from "../editor/useFileBuffer";
import { Check, Circle } from "../icons";
import styles from "./Editor.module.css";

type Props = {
  path: string;
  markers?: Marker[];
  revealLine?: number | null;
};

// A workspace file in Monaco. Opens and saves through /api/files with the ETag; Ctrl-S saves.
// A save the server refuses is shown, never retried silently.
export function Editor({ path, markers = [], revealLine }: Props) {
  const buffer = useFileBuffer(path);
  if (buffer.error) return <div className={styles.problem}><ProblemFromError error={buffer.error} /></div>;
  return (
    <section className={styles.editor} aria-label={`Editor for ${path}`}>
      <header className={styles.bar}>
        <span className={`${styles.path} code`}>{path}</span>
        {buffer.dirty ? <Tag tone="warn" icon={Circle}>unsaved</Tag> : <Tag tone="muted" icon={Check}>saved</Tag>}
        <Button variant="primary" size="sm" disabled={!buffer.dirty || buffer.saving} onClick={buffer.save}>
          Save
        </Button>
      </header>
      <div className={styles.surface}>
        {buffer.loading ? (
          <p className="small">Reading {path}</p>
        ) : (
          <CodeSurface
            path={path}
            value={buffer.text}
            onChange={buffer.edit}
            onSave={buffer.save}
            markers={markers}
            revealLine={revealLine}
          />
        )}
      </div>
      {buffer.saveError ? <div className={styles.problem}><ProblemFromError error={buffer.saveError} /></div> : null}
    </section>
  );
}
