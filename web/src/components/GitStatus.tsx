import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Check, Warning } from "../icons";
import styles from "./GitStatus.module.css";

export type Git = {
  repo: boolean;
  root: string | null;
  branch: string | null;
  head: string | null;
  clean: boolean;
  changed: string[];
  identity: boolean;
  path: string | null;
  path_committed: boolean | null;
};

// The workspace repository, read-only (GET /api/git/status): nothing here stages, stashes or commits.
export function useGitStatus(path: string | null) {
  return useQuery({
    queryKey: ["git", path],
    queryFn: () =>
      unwrap(api.GET("/api/git/status", { params: { query: path ? { path } : {} } })) as unknown as Promise<Git>,
    refetchInterval: 5000,
  });
}

// Record mode runs from a committed spec in a clean tree.
export function recordReady(git: Git | undefined): boolean {
  return git !== undefined && git.repo && git.clean && git.path_committed === true;
}

const plural = (n: number) => `${n} uncommitted ${n === 1 ? "change" : "changes"}`;

// One line, three states: other files changed, only the spec itself changed, or clean and committed.
export function GitStatus({ git, path }: { git: Git | undefined; path: string }) {
  if (!git) return null;
  if (!git.repo) {
    return (
      <section aria-label="Git status" className={`${styles.line} ${styles.warn} small`}>
        <span className={styles.state}>
          <Warning size={14} aria-hidden="true" />
          not a git repository
        </span>
        <span className={styles.muted}>Record mode needs the workspace inside one.</span>
      </section>
    );
  }
  const where = (
    <span>
      git <code className="value">{git.branch ?? "detached"}</code> at <code className="value">{git.head?.slice(0, 7) ?? "no commits"}</code>
    </span>
  );
  if (git.clean && git.path_committed) {
    return (
      <section aria-label="Git status" className={`${styles.line} ${styles.good} small`}>
        <span className={styles.state}>
          <Check size={14} aria-hidden="true" />
          clean
        </span>
        {where}
        <span>
          <code className="value">{path}</code> is committed and unchanged
        </span>
      </section>
    );
  }
  const others = git.changed.filter((f) => f !== path);
  return (
    <section aria-label="Git status" className={`${styles.line} ${styles.warn} small`}>
      <span className={styles.state}>
        <Warning size={14} aria-hidden="true" />
        {git.clean ? `${path} is not committed` : plural(git.changed.length)}
      </span>
      {where}
      {git.changed.length > 0 && <span className={styles.muted}>changed:</span>}
      {git.changed.map((f) => (
        <code key={f} className="value">
          {f}
        </code>
      ))}
      {git.changed.length > 0 && others.length === 0 && (
        <span className={styles.muted}>(the spec itself: commit it to run in record mode)</span>
      )}
      <span className={styles.spacer} />
      <span className={styles.muted}>read-only; nanoscope never stages, stashes or discards</span>
    </section>
  );
}
