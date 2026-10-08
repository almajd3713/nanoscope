import { useId, useState } from "react";
import { Link } from "react-router-dom";
import { Field } from "../components/Field";
import { ProblemFromError } from "../components/ProblemView";
import type { CatalogBlock } from "../editor/completions";
import type { Edit } from "./api";
import { type Box, type GClass, nodeAt, valueText } from "./flow";
import styles from "./Inspector.module.css";

type Props = {
  cls: GClass;
  box: Box;
  blocks: CatalogBlock[];
  onEdit: (edits: Edit[]) => void;
  busy?: boolean;
  // the last refusal from the server (a locked block, a stale file), shown word for word
  error?: unknown;
};

// What a typed value means: numbers, True/False/None, anything else a string.
export function parseLiteral(text: string): unknown {
  const t = text.trim();
  if (t === "True") return true;
  if (t === "False") return false;
  if (t === "None") return null;
  if (/^[-+]?\d+$/.test(t)) return Number.parseInt(t, 10);
  if (/^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$/.test(t)) return Number.parseFloat(t);
  const quoted = /^(["'])(.*)\1$/.exec(t);
  return quoted ? quoted[2] : t;
}

const BLOCK_TYPE = /BlockSpec|nn\.Module|Module/;

// The selected box: its arguments (each edit is a set_arg that changes one line of the file) and
// the other blocks of its family it can be swapped for (replace_block). Locked blocks are listed
// but cannot be picked: they say which lesson unlocks them.
export function Inspector({ cls, box, blocks, onEdit, busy, error }: Props) {
  const swapId = useId();
  const node = box.path ? nodeAt(cls, box.path) : null;
  const info = blocks.find((b) => b.name === box.name);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  if (!box.path || !node) {
    return (
      <section className={styles.inspector} aria-label={`Inspector for ${box.name}`}>
        <div className={styles.title}>
          <h2 className="heading">{box.name}</h2>
          <span className={`caption ${styles.muted}`}>{box.slot} · added by nanoscope, not written in the file</span>
        </div>
      </section>
    );
  }
  const written = node.kind === "block" ? node.args : {};
  const options = blocks.filter((b) => b.family === box.family);
  const current = (name: string): string => (written[name] ? valueText(written[name]) : "");

  const commit = (name: string) => {
    const text = drafts[name];
    if (text === undefined || box.path === null) return;
    setDrafts(({ [name]: _gone, ...rest }) => rest);
    if (text.trim() === current(name)) return;
    if (text.trim() === "") {
      if (written[name]) onEdit([{ op: "remove_arg", class: cls.name, path: box.path, arg: name }]);
      return;
    }
    onEdit([{ op: "set_arg", class: cls.name, path: box.path, arg: name, value: { kind: "literal", value: parseLiteral(text), span: null } }]);
  };

  const swap = (name: string) => {
    if (name === box.name || box.path === null) return;
    onEdit([{ op: "replace_block", class: cls.name, path: box.path, node: { kind: "block", block: name, args: {}, span: null } }]);
  };

  const args = (info?.args ?? []).filter((a) => !(a.type && BLOCK_TYPE.test(a.type)) && written[a.name]?.kind !== "block");
  const lockedPick = options.find((o) => o.lock?.locked && o.name !== box.name);

  return (
    <section className={styles.inspector} aria-label={`Inspector for ${box.name}`}>
      <div className={styles.title}>
        <h2 className="heading">{box.name}</h2>
        <span className={`caption ${styles.muted}`}>{[box.slot, box.family, box.line ? `line ${box.line}` : null].filter(Boolean).join(" · ")}</span>
      </div>

      {box.kind === "block" && options.length > 0 && (
        <div className={styles.fields}>
          <label className="label" htmlFor={swapId}>
            Swap for
          </label>
          <select id={swapId} className={`${styles.select} body`} value={box.name} disabled={busy} onChange={(e) => swap(e.target.value)}>
            {options.map((o) => (
              <option key={o.name} value={o.name} disabled={o.lock?.locked && o.name !== box.name}>
                {o.name}
                {o.lock?.locked ? ` (locked, needs ${o.lock.lesson})` : ""}
              </option>
            ))}
          </select>
          {lockedPick?.lock?.lesson && (
            <span className={`caption ${styles.locked}`}>
              {lockedPick.name} is locked. Pass <Link to={`/learn/${lockedPick.lock.lesson}`}>{lockedPick.lock.lesson}</Link> to use it.
            </span>
          )}
        </div>
      )}

      {box.kind === "opaque" && (
        <p className="small">
          This call is not one the graph can edit{box.line ? `, line ${box.line}` : ""}. Change it in the code.
        </p>
      )}

      {box.kind === "block" && (
        <div className={styles.fields}>
          {args.map((a) => (
            <Field
              key={a.name}
              label={a.name}
              value={drafts[a.name] ?? current(a.name)}
              disabled={busy}
              help={`${a.type ?? ""}${a.required ? ", required" : a.default !== undefined ? `, default ${String(a.default)}` : ""}`.replace(/^, /, "")}
              onChange={(v) => setDrafts((d) => ({ ...d, [a.name]: v }))}
              onCommit={() => commit(a.name)}
            />
          ))}
        </div>
      )}
      {error ? <ProblemFromError error={error} /> : null}
    </section>
  );
}
