import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { ButtonLink } from "../components/Button";
import { EmptyState } from "../components/EmptyState";
import { ProblemFromError } from "../components/ProblemView";
import { Plus } from "../icons";
import styles from "./Study.module.css";

// Studies in the workspace and studies that have run.
export function Studies() {
  const studies = useQuery({
    queryKey: ["studies"],
    queryFn: () => unwrap(api.GET("/api/studies")),
    refetchInterval: 5000,
  });
  if (studies.error) return <ProblemFromError error={studies.error} />;
  if (!studies.data) return <p className="body">Loading studies…</p>;
  return (
    <div className={styles.page}>
      <header className={styles.bar}>
        <h1 className="title">Studies</h1>
        <span className={`small ${styles.muted}`}>spec files in <span className="value">studies/</span> and studies that have run</span>
        <span className={styles.spacer} />
        <ButtonLink to="/studies/new" variant="primary">
          <Plus size={16} aria-hidden="true" />
          New study
        </ButtonLink>
      </header>
      {studies.data.length === 0 ? (
        <EmptyState
          title="No studies yet"
          body="A study trains several model variants on the same seeds and budget, and compares them with intervals."
          action={<ButtonLink to="/studies/new">Build one</ButtonLink>}
          command="nanoscope study studies/m1-ablation.toml --dry-run"
        />
      ) : (
        <div className={styles.wrap}>
          <table className={`${styles.table} small`}>
            <thead>
              <tr>
                <th className="label" scope="col">Study</th>
                <th className="label" scope="col">Mode</th>
                <th className="label" scope="col">Preset</th>
                <th className={`label ${styles.right}`} scope="col">Variants</th>
                <th className={`label ${styles.right}`} scope="col">Runs</th>
                <th className="label" scope="col">Spec</th>
              </tr>
            </thead>
            <tbody>
              {studies.data.map((s) => (
                <tr key={s.name}>
                  <td>
                    <Link to={`/studies/${s.name}`} className={`value ${styles.link}`}>{s.name}</Link>
                  </td>
                  <td className="value">{s.mode ?? "–"}</td>
                  <td className="value">{s.preset ?? "–"}</td>
                  <td className={`value ${styles.right}`}>{s.variants.length}</td>
                  <td className={`value ${styles.right}`}>
                    {s.runs_by_state["done"] ?? 0} / {s.runs_total}
                  </td>
                  <td>
                    {s.spec ? <Link to={`/studies/${s.name}/edit`} className={`value ${styles.link}`}>{s.spec}</Link> : <span className={styles.muted}>no spec file</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
