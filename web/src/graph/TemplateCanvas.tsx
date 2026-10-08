import { useState } from "react";
import { Button } from "../components/Button";
import { ProblemView } from "../components/ProblemView";
import type { CatalogBlock } from "../editor/completions";
import type { Edit } from "./api";
import { carriesBlock, DRAG_TYPE, emptySlot, planFill } from "./dnd";
import { type GClass, type GNode, type Path, valueText } from "./flow";
import styles from "./TemplateCanvas.module.css";

type Props = {
  cls: GClass;
  blocks: CatalogBlock[];
  onEdit: (edits: Edit[]) => void;
};

type Found = { path: Path; node: Extract<GNode, { kind: "block" }> };

// A template is a Composite whose slots a lesson asks the learner to fill. Found by walking the
// class's arguments; a template inside a template (BlockTemplate's attn slot) nests.
function isTemplate(node: GNode): node is Extract<GNode, { kind: "block" }> {
  return node.kind === "block" && (node.family === "template" || node.local === true);
}

function templatesIn(node: GNode, path: Path): Found[] {
  if (node.kind !== "block") return [];
  const here: Found[] = isTemplate(node) ? [{ path, node }] : [];
  return [...here, ...Object.entries(node.args).flatMap(([k, v]) => (isTemplate(node) && isTemplate(v) ? [] : templatesIn(v, [...path, k])))];
}

function Slot({ cls, template, name, blocks, onEdit, setNote }: { cls: GClass; template: Found; name: string; blocks: CatalogBlock[]; onEdit: Props["onEdit"]; setNote: (m: string | null) => void }) {
  const [over, setOver] = useState(false);
  const value = template.node.args[name];
  const inner = value && isTemplate(value) ? value : null;
  const filled = value !== undefined && !(value.kind === "literal" && value.value === null);
  return (
    <div
      className={`${styles.slot} ${filled ? styles.filled : ""} ${over ? styles.over : ""} ${inner ? styles.nested : ""}`}
      role="group"
      aria-label={`${name} slot`}
      onDragOver={(e) => {
        if (!carriesBlock(e)) return;
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        setOver(false);
        const dropped = e.dataTransfer.getData(DRAG_TYPE);
        if (!dropped) return;
        e.preventDefault();
        const plan = planFill(cls.name, template.path, name, dropped, blocks);
        if (plan.ok) {
          setNote(null);
          onEdit([plan.edit]);
        } else {
          setNote(plan.message);
        }
      }}
    >
      <div className={styles.row}>
        <span className="label">{name}</span>
        {filled && !inner && (
          <Button size="sm" variant="quiet" aria-label={`Empty the ${name} slot`} onClick={() => onEdit([emptySlot(cls.name, template.path, name)])}>
            Empty
          </Button>
        )}
      </div>
      {!inner && <span className={`${styles.name} ${filled ? "body-strong" : `${styles.muted} body`}`}>{filled ? valueText(value!) : "drop a block here"}</span>}
      {inner && <TemplateCard cls={cls} template={{ path: [...template.path, name], node: inner }} blocks={blocks} onEdit={onEdit} setNote={setNote} />}
    </div>
  );
}

function TemplateCard({ cls, template, blocks, onEdit, setNote }: { cls: GClass; template: Found; blocks: CatalogBlock[]; onEdit: Props["onEdit"]; setNote: (m: string | null) => void }) {
  const info = blocks.find((b) => b.name === template.node.block);
  const slots = info?.args.map((a) => a.name) ?? Object.keys(template.node.args);
  return (
    <section className={styles.template} aria-label={`${template.node.block} template`}>
      <h3 className="heading">{template.node.block}</h3>
      <div className={styles.slots}>
        {slots.map((s) => (
          <Slot key={s} cls={cls} template={template} name={s} blocks={blocks} onEdit={onEdit} setNote={setNote} />
        ))}
      </div>
    </section>
  );
}

// Whether the class has a template to fill (a lesson's starter does).
export function hasTemplate(cls: GClass): boolean {
  return Object.entries(cls.args ?? {}).some(([k, v]) => templatesIn(v, [k]).length > 0);
}

// The lesson template as slots to fill: each slot is a drop target for a primitive from the
// palette, and an empty slot says so. Every drop is a fill_slot patch.
export function TemplateCanvas({ cls, blocks, onEdit }: Props) {
  const [note, setNote] = useState<string | null>(null);
  const top = Object.entries(cls.args ?? {}).flatMap(([k, v]) => templatesIn(v, [k]));
  // the outermost templates only: nested ones are drawn inside their owner
  const outer = top.filter((t) => !top.some((o) => o !== t && t.path.length > o.path.length && o.path.every((p, i) => t.path[i] === p)));
  if (outer.length === 0) {
    return <p className="body">{cls.name} has no template to fill. Start a lesson to get one.</p>;
  }
  return (
    <div className={styles.canvas} aria-label={`Template canvas for ${cls.name}`}>
      {outer.map((t) => (
        <TemplateCard key={t.path.join("/")} cls={cls} template={t} blocks={blocks} onEdit={onEdit} setNote={setNote} />
      ))}
      {note && <ProblemView title="That block can't go there" detail={note} />}
    </div>
  );
}
