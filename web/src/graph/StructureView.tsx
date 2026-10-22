import type { ReactNode } from "react";
import { Note } from "../components/Note";
import { countText } from "../format";
import { builtAs, type Structure, structureTree, type TreeNode } from "./structure";
import styles from "./StructureView.module.css";
import { shapeChange } from "./trace";

type Props = {
  structure: Structure;
  className: string | null;
  // set when the code is on the page: a line number becomes a link into the editor
  onEditInCode?: (line: number) => void;
  // what to say under the view: how to get a graph you can edit
  footer?: ReactNode;
};

// What a plain PyTorch class builds, read-only: one box per module, nested as the code nests
// them, with shapes when the forward pass runs. It is drawn from the code after each save; to
// change it, change the code.
export function StructureView({ structure, className, onEditInCode, footer }: Props) {
  const report = structure.report;
  const tree = report ? structureTree(report.modules) : null;
  const lineLink = (line: number | null) =>
    line === null ? null : onEditInCode ? (
      <button type="button" className={`${styles.line} value`} onClick={() => onEditInCode(line)}>
        line {line}
      </button>
    ) : (
      <span className={`${styles.muted} value`}>line {line}</span>
    );

  return (
    <section className={styles.view} aria-label="Structure">
      <header className={styles.head}>
        <h2 className="heading">Structure</h2>
        <span className={`caption ${styles.muted}`}>read-only, drawn from your code after each save</span>
      </header>
      {structure.state === "waiting" && <p className={`body ${styles.muted}`}>Waiting for a worker to trace {className ?? "the class"}…</p>}
      {structure.state === "failed" && <Note>The structure job failed: {structure.error}</Note>}
      {report && (
        <>
          <p className={`small ${styles.muted}`}>
            Built as <span className="value">{builtAs(report)}</span>
            {report.inputs && (
              <>
                {" "}
                on input <span className="value">{report.inputs.map((s) => s.join("x")).join(", ")}</span>
              </>
            )}
            {report.built && <> · <span className="value">{countText(report.params)}</span> params</>}
          </p>
          {report.error && (
            <Note>
              {report.error.stage === "build" ? `${report.model} could not be built` : "The forward pass stopped"}
              {report.error.line !== null && <> at {lineLink(report.error.line)}</>}: {report.error.type}: {report.error.message}
              {report.error.stage === "forward" && report.built ? ". Shapes appear once it runs." : ""}
            </Note>
          )}
          {tree && tree.children.length === 0 && (
            <p className={`body ${styles.muted}`}>
              {report.model} has no layers yet. The modules you assign in <span className="value">__init__</span> (<span className="value">self.name = …</span>) appear here after you save.
            </p>
          )}
          {tree && tree.children.length > 0 && (
            <ul className={styles.tree} aria-label={`Modules of ${report.model}`}>
              <Item node={tree} lineLink={lineLink} />
            </ul>
          )}
        </>
      )}
      {footer && <div className={styles.footer}>{footer}</div>}
    </section>
  );
}

function Item({ node, lineLink }: { node: TreeNode; lineLink: (line: number | null) => ReactNode }) {
  const shape = shapeChange(node.input_shapes, node.output_shapes);
  return (
    <li className={styles.item}>
      <div className={styles.box}>
        <div className={styles.row}>
          <span className={`${styles.name} value`}>
            {node.name || node.type}
            {node.repeat > 1 && <span className={styles.muted}> ×{node.repeat}</span>}
          </span>
          {lineLink(node.line)}
        </div>
        <div className={`${styles.row} small`}>
          <span className={styles.muted}>
            {node.name ? node.type : "model"}
            {node.block && node.block !== node.type ? ` · block ${node.block}` : node.block ? " · block" : ""}
          </span>
          <span className={`${styles.muted} value`}>
            {shape}
            {shape && node.params ? " · " : ""}
            {node.params ? `${countText(node.params)} params` : ""}
          </span>
        </div>
      </div>
      {node.children.length > 0 && (
        <ul className={styles.children}>
          {node.children.map((child) => (
            <Item key={child.path} node={child} lineLink={lineLink} />
          ))}
        </ul>
      )}
    </li>
  );
}
