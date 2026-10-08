import { useState } from "react";
import { Link } from "react-router-dom";
import type { CatalogBlock } from "../editor/completions";
import { DotsSixVertical, LockSimple, SealCheck } from "../icons";
import styles from "./Palette.module.css";

export const DRAG_TYPE = "application/x-nanoscope-block";

// Blocks nanoscope puts in every model itself; they are not slots a learner fills.
const FIXED = new Set(["Decoder", "Head", "TokenEmbedding"]);

export type PaletteBlock = CatalogBlock & {
  user?: boolean;
  certified?: boolean;
  certification?: { state: string };
  tier?: string;
};

type Props = {
  blocks: PaletteBlock[];
  // the whole-palette state, for a hint when nothing is locked ("open" policy)
  policy?: string;
};

function Item({ block }: { block: PaletteBlock }) {
  const locked = block.lock?.locked === true;
  return (
    <li
      className={`${styles.item} ${locked ? styles.locked : ""}`}
      draggable={!locked}
      aria-disabled={locked || undefined}
      onDragStart={(e) => {
        if (locked) return e.preventDefault();
        e.dataTransfer.setData(DRAG_TYPE, block.name);
        e.dataTransfer.effectAllowed = "copy";
      }}
    >
      <div className={styles.row}>
        {locked ? <LockSimple size={16} aria-label="locked" /> : <DotsSixVertical size={16} aria-hidden="true" />}
        <span className={`${styles.name} body-strong`}>{block.name}</span>
        {block.user && block.certified && <SealCheck size={16} aria-label="certified" />}
      </div>
      {locked && block.lock?.lesson && (
        <span className="caption">
          Pass <Link to={`/learn/${block.lock.lesson}`}>{block.lock.lesson}</Link> to use it
        </span>
      )}
      {block.user && !block.certified && (
        <span className={`caption ${styles.muted}`}>
          {block.certification?.state === "stale" ? "changed since it was certified" : block.certification?.state === "failed" ? "failed its check" : "not certified"}
        </span>
      )}
    </li>
  );
}

// The blocks a model can be built from, by family; your own blocks come last. A locked block is
// shown but cannot be dragged: it names the lesson that unlocks it.
export function Palette({ blocks }: Props) {
  const [query, setQuery] = useState("");
  const shown = blocks.filter((b) => !FIXED.has(b.name) && b.name.toLowerCase().includes(query.trim().toLowerCase()));
  const families = [...new Set(shown.filter((b) => !b.user).map((b) => b.family))].sort();
  const mine = shown.filter((b) => b.user);
  return (
    <nav className={styles.palette} aria-label="Block palette">
      <input className={`${styles.search} body`} type="search" aria-label="Search blocks" placeholder="Search blocks" value={query} onChange={(e) => setQuery(e.target.value)} />
      {families.map((family) => (
        <section key={family} className={styles.group} aria-label={family}>
          <h3 className="label">{family}</h3>
          <ul className={styles.list}>
            {shown
              .filter((b) => !b.user && b.family === family)
              .map((b) => (
                <Item key={b.name} block={b} />
              ))}
          </ul>
        </section>
      ))}
      {(mine.length > 0 || query === "") && (
        <section className={styles.group} aria-label="Your blocks">
          <h3 className="label">Your blocks</h3>
          {mine.length === 0 ? (
            <p className={`caption ${styles.muted}`}>Blocks you register in your files with register_block show up here.</p>
          ) : (
            <ul className={styles.list}>
              {mine.map((b) => (
                <Item key={b.name} block={b} />
              ))}
            </ul>
          )}
        </section>
      )}
    </nav>
  );
}
