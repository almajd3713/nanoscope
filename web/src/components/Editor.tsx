import { useState } from "react";
import { Button } from "./Button";
import { ConflictDialog } from "./ConflictDialog";
import { ProblemFromError } from "./ProblemView";
import { Tag } from "./Tag";
import { CodeSurface } from "../editor/CodeSurface";
import type { Marker } from "../editor/types";
import { useFileBuffer } from "../editor/useFileBuffer";
import { useCatalog } from "../editor/useCatalog";
import { documentUri } from "../editor/lsp";
import { useLspService } from "../editor/lspService";
import { useLspMarkers } from "../editor/useLspMarkers";
import { useLint } from "../editor/useLint";
import { useUseLsp } from "../app/settings";
import { Check, Circle, Minus, Warning, X } from "../icons";
import styles from "./Editor.module.css";

const problems = (found: Marker[]): string => {
  const errors = found.filter((m) => m.severity === "error").length;
  if (found.length === 0) return "no problems";
  return errors === found.length ? `${errors} ${errors === 1 ? "error" : "errors"}` : `${found.length} ${found.length === 1 ? "problem" : "problems"}`;
};

type Props = {
  path: string;
  markers?: Marker[];
  revealLine?: number | null;
};

// A workspace file in Monaco. Opens and saves through /api/files with the ETag; Ctrl-S saves.
// A save the server refuses is shown, never retried silently.
export function Editor({ path, markers = [], revealLine }: Props) {
  const buffer = useFileBuffer(path);
  // "Decide later" hides the dialog for this conflict; the banner brings it back
  const [putOff, setPutOff] = useState<string | null>(null);
  const complete = useCatalog();
  const lint = useLint(path, buffer.etag);
  const service = useLspService();
  const wantsLsp = useUseLsp();
  const typed = useLspMarkers(path, service);
  const shown = [...lint, ...typed, ...markers];
  const lsp = service.status === "connected" ? { client: service.client, uri: documentUri(service.root, path) } : null;
  const conflictId = buffer.conflict?.currentEtag ?? null;
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
            markers={shown}
            revealLine={revealLine}
            complete={complete}
            lsp={lsp}
          />
        )}
      </div>
      <footer className={`caption ${styles.checkers}`} aria-label="Checkers">
        <span className={lint.length ? styles.found : styles.clean}>
          {lint.length ? <X size={14} aria-hidden="true" /> : <Check size={14} aria-hidden="true" />}
          ruff: {problems(lint)}
        </span>
        {service.status === "connected" ? (
          <span className={typed.length ? styles.found : styles.clean}>
            {typed.length ? <X size={14} aria-hidden="true" /> : <Check size={14} aria-hidden="true" />}
            {service.server?.name ?? "language server"}: {problems(typed)}
          </span>
        ) : (
          <span className={styles.muted}>
            <Minus size={14} aria-hidden="true" />{" "}
            {!wantsLsp
              ? "no type checks: the language server is turned off in Settings"
              : service.status === "unavailable"
                ? "no type checks: the language server is not running"
                : "connecting to the language server"}
          </span>
        )}
      </footer>
      {buffer.external && (
        <div className={styles.problem} role="status">
          <Tag tone="warn" icon={Warning}>{buffer.external.kind === "deleted" ? "deleted on disk" : "changed on disk"}</Tag>
          <p className="small">
            {buffer.external.kind === "deleted"
              ? "This file was deleted while you have unsaved edits. Saving writes it again."
              : "This file changed while you have unsaved edits."}
          </p>
          <Button size="sm" onClick={() => void buffer.reload()}>Reload from disk</Button>{" "}
          <Button size="sm" variant="quiet" onClick={buffer.keepEditing}>Keep my edits</Button>
        </div>
      )}
      {buffer.conflict && (
        <>
          <div className={styles.problem} role="status">
            <Tag tone="warn" icon={Warning}>changed on disk</Tag>{" "}
            <Button size="sm" onClick={() => setPutOff(null)}>Resolve</Button>
          </div>
          <ConflictDialog
            open={putOff !== conflictId}
            onOpenChange={(open) => setPutOff(open ? null : conflictId)}
            path={path}
            diff={buffer.conflict.diff}
            busy={buffer.saving}
            onKeepMine={buffer.keepMine}
            onTakeTheirs={() => void buffer.takeTheirs()}
          />
        </>
      )}
      {buffer.saveError ? <div className={styles.problem}><ProblemFromError error={buffer.saveError} /></div> : null}
    </section>
  );
}
