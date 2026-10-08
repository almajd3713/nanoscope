import { useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useLevel } from "../app/level";
import { Button } from "../components/Button";
import { Editor } from "../components/Editor";
import { EmptyState } from "../components/EmptyState";
import { ProblemFromError, ProblemView } from "../components/ProblemView";
import type { CatalogBlock } from "../editor/completions";
import { fromChecks, useDescribeMarkers } from "../editor/diagnostics";
import { useFileDoc } from "../editor/useFileBuffer";
import { type Edit, useBlocks, useGraph, useLayerStats, usePatchGraph, useTrace } from "../graph/api";
import { defaultDepth, DepthDial } from "../graph/DepthDial";
import type { Depth } from "../graph/layout";
import type { NodeInfo } from "../graph/nodes";
import { StackPanel } from "../graph/StackPanel";
import { planDrop } from "../graph/dnd";
import type { Box } from "../graph/flow";
import { GraphView } from "../graph/GraphView";
import { useHistory } from "../graph/history";
import { Inspector } from "../graph/Inspector";
import { Palette } from "../graph/Palette";
import { hasTemplate, TemplateCanvas } from "../graph/TemplateCanvas";
import { shows } from "../levels";
import { LessonBar, useLessonOfFile } from "./LessonBar";
import styles from "./Model.module.css";

// A model file: the palette, its graph, the inspector for the selected box and (from Tinker up)
// the code beside them. All of it reads one file through the API; an edit anywhere is a change
// to that file, and every view follows it.
export function Model() {
  const path = useParams()["*"] ?? "";
  const [search, setSearch] = useSearchParams();
  const level = useLevel();
  const graph = useGraph(path);
  const blocks = useBlocks();
  const [selected, setSelected] = useState<Box | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [chosenDepth, setDepth] = useState<Depth | null>(null);
  const [reveal, setReveal] = useState<number | null>(null);
  const depth = chosenDepth ?? defaultDepth(level);
  const patch = usePatchGraph(path, graph.data?.etag ?? null);
  const history = useHistory(path);

  const classes = graph.data?.classes ?? [];
  const cls = classes.find((c) => c.name === search.get("class")) ?? classes[0];
  const described = useDescribeMarkers(path, graph.data?.etag ?? null, cls ? `${path}:${cls.name}` : null);
  const trace = useTrace(path, cls?.name ?? null);
  const layerStats = useLayerStats(cls?.name ?? null);
  const lesson = useLessonOfFile(path);
  const source = useFileDoc(path).data?.content ?? "";
  // a failed equivalence check goes on its class, as the lesson names it
  const checked = lesson.check.result && lesson.detail ? fromChecks(lesson.check.result.checks, lesson.detail.checks, source) : [];
  const markers = [...described, ...checked];

  const editInCode = (line: number) => {
    setReveal(null);
    queueMicrotask(() => setReveal(line));
  };

  const apply = (edits: Edit[]) => {
    setNote(null);
    patch.mutate(edits);
  };

  if (graph.error) {
    return (
      <div className={styles.page}>
        <h1 className="title">{path}</h1>
        <ProblemFromError error={graph.error} />
      </div>
    );
  }
  if (graph.data && !cls) {
    // a plain PyTorch model (most of the early lessons): nothing to draw, but it is still the
    // lesson's file, so the lesson's actions and the code are here
    return (
      <div className={styles.page}>
        <h1 className="title">{path}</h1>
        {lesson.ids && <LessonBar file={path} lesson={lesson} />}
        <EmptyState
          title="No graph for this file"
          body="It has no Decoder or Composite class, so there is nothing to draw: it is plain PyTorch. Edit it in your editor or below."
        />
        {shows("editor", level) ? (
          <Editor path={path} markers={checked} />
        ) : (
          <pre className="code" aria-label="Your file">{source}</pre>
        )}
      </div>
    );
  }

  const catalog = (blocks.data?.blocks ?? []) as CatalogBlock[];
  const infoOf = (name: string): NodeInfo | undefined => {
    const b = (blocks.data?.blocks ?? []).find((x) => x.name === name);
    if (!b) return undefined;
    const extra = b as { reference?: string | null; user?: boolean; certification?: { state: string } };
    return { reference: extra.reference ?? null, user: extra.user === true, certification: extra.certification?.state, locked: b.lock?.locked === true, lesson: b.lock?.lesson ?? null };
  };
  const editor = shows("editor", level);
  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className="title">{cls?.name ?? path}</h1>
        {classes.length > 1 && (
          <select className={`${styles.select} body`} aria-label="Class" value={cls?.name} onChange={(e) => setSearch({ class: e.target.value })}>
            {classes.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        )}
        <span className={`value ${styles.muted}`}>{path}</span>
        <DepthDial value={depth} onChange={setDepth} />
        <Button size="sm" disabled={!history.canUndo} onClick={history.undo}>
          Undo
        </Button>
        <Button size="sm" disabled={!history.canRedo} onClick={history.redo}>
          Redo
        </Button>
      </header>
      {lesson.ids && <LessonBar file={path} lesson={lesson} />}
      <div className={`${styles.body} ${editor ? "" : styles.bodyNoEditor}`}>
        <aside className={styles.side} aria-label="Palette">
          {shows("palette", level) && <Palette blocks={catalog} />}
        </aside>
        {editor && (
          <Editor
            path={path}
            markers={markers}
            revealLine={reveal ?? selected?.line ?? null}
          />
        )}
        <div className={styles.stage}>
          {cls && (
            <GraphView
              cls={cls}
              depth={depth}
              infoOf={infoOf}
              trace={trace}
              layerStats={layerStats.blocks}
              onEditInCode={editInCode}
              onAddLayer={() => apply([{ op: "add_layer", class: cls.name }])}
              onRemoveLayer={() => apply([{ op: "remove_layer", class: cls.name }])}
              selected={selected?.id ?? null}
              onSelect={setSelected}
              onDropBlock={(box, name) => {
                const plan = planDrop(cls.name, box, name, catalog);
                if (plan.ok) apply([plan.edit]);
                else setNote(plan.message);
              }}
            />
          )}
          <div>
            {layerStats.run && layerStats.blocks && (
              <p className="caption">Layer colours: gradient norm at step {layerStats.step} of <span className="value">{layerStats.run}</span></p>
            )}
            {note && <ProblemView title="That block can't go there" detail={note} />}
            {cls && selected && selected.path && <Inspector cls={cls} box={selected} blocks={catalog} onEdit={apply} busy={patch.isPending} error={patch.error} />}
            {!(selected && selected.path) && patch.error ? <ProblemFromError error={patch.error} /> : null}
            {cls && <StackPanel cls={cls} onEdit={apply} busy={patch.isPending} />}
            {cls && hasTemplate(cls) && <TemplateCanvas cls={cls} blocks={catalog} onEdit={apply} />}
          </div>
        </div>
      </div>
    </div>
  );
}
