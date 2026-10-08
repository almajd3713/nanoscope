import { useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useLevel } from "../app/level";
import { Button } from "../components/Button";
import { Editor } from "../components/Editor";
import { EmptyState } from "../components/EmptyState";
import { ProblemFromError, ProblemView } from "../components/ProblemView";
import type { CatalogBlock } from "../editor/completions";
import { useDescribeMarkers } from "../editor/diagnostics";
import { type Edit, useBlocks, useGraph, usePatchGraph } from "../graph/api";
import { planDrop } from "../graph/dnd";
import type { Box } from "../graph/flow";
import { GraphView } from "../graph/GraphView";
import { useHistory } from "../graph/history";
import { Inspector } from "../graph/Inspector";
import { Palette } from "../graph/Palette";
import { hasTemplate, TemplateCanvas } from "../graph/TemplateCanvas";
import { shows } from "../levels";
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
  const patch = usePatchGraph(path, graph.data?.etag ?? null);
  const history = useHistory(path);

  const classes = graph.data?.classes ?? [];
  const cls = classes.find((c) => c.name === search.get("class")) ?? classes[0];
  const markers = useDescribeMarkers(path, graph.data?.etag ?? null, cls ? `${path}:${cls.name}` : null);

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
    return (
      <div className={styles.page}>
        <h1 className="title">{path}</h1>
        <EmptyState
          title="No model in this file"
          body={`The file has no Decoder or Composite class to draw. ${shows("editor", level) ? "Write one in the editor below, or open another file." : "Start a lesson to get a file with one."}`}
        />
        {shows("editor", level) && <Editor path={path} />}
      </div>
    );
  }

  const catalog = (blocks.data?.blocks ?? []) as CatalogBlock[];
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
        <Button size="sm" disabled={!history.canUndo} onClick={history.undo}>
          Undo
        </Button>
        <Button size="sm" disabled={!history.canRedo} onClick={history.redo}>
          Redo
        </Button>
      </header>
      <div className={`${styles.body} ${editor ? "" : styles.bodyNoEditor}`}>
        <aside className={styles.side} aria-label="Palette">
          {shows("palette", level) && <Palette blocks={catalog} />}
        </aside>
        {editor && (
          <Editor
            path={path}
            markers={markers}
            revealLine={selected?.line ?? null}
          />
        )}
        <div className={styles.stage}>
          {cls && (
            <GraphView
              cls={cls}
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
            {note && <ProblemView title="That block can't go there" detail={note} />}
            {cls && selected && selected.path && <Inspector cls={cls} box={selected} blocks={catalog} onEdit={apply} busy={patch.isPending} error={patch.error} />}
            {!(selected && selected.path) && patch.error ? <ProblemFromError error={patch.error} /> : null}
            {cls && hasTemplate(cls) && <TemplateCanvas cls={cls} blocks={catalog} onEdit={apply} />}
          </div>
        </div>
      </div>
    </div>
  );
}
