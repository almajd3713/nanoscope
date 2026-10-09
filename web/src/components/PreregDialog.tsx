import * as AlertDialog from "@radix-ui/react-alert-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { followJob } from "../studies/useStudyChecks";
import { Button } from "./Button";
import { EquivalentCommand } from "./EquivalentCommand";
import { Note } from "./Note";
import { ProblemFromError, ProblemView } from "./ProblemView";
import styles from "./PreregDialog.module.css";

export type Preview = {
  root: string;
  files: string[];
  diff: string;
  message: string;
  unrelated: string[];
  identity: boolean;
  nothing_to_commit: boolean;
  head: string | null;
  hash: string;
};

type Row = { kind: "head" | "add" | "del" | "ctx"; sign: string; text: string };

// A unified diff as rows: the file headers dim, additions and deletions with their sign.
export function diffRows(diff: string): Row[] {
  let inHunk = false;
  return diff
    .split("\n")
    .filter((line, i, all) => !(i === all.length - 1 && line === ""))
    .map((line): Row => {
      if (line.startsWith("@@")) {
        inHunk = true;
        return { kind: "head", sign: "", text: line };
      }
      if (!inHunk) return { kind: "head", sign: "", text: line };
      if (line.startsWith("+")) return { kind: "add", sign: "+", text: line.slice(1) };
      if (line.startsWith("-")) return { kind: "del", sign: "−", text: line.slice(1) };
      return { kind: "ctx", sign: "", text: line.slice(1) };
    });
}

type Props = { name: string; open: boolean; onOpenChange: (open: boolean) => void };

// The preregistration commit, step by step: a worker previews exactly what git would commit, you
// read the diff and the message, then confirm. The commit is refused if anything changed since.
export function PreregDialog({ name, open, onOpenChange }: Props) {
  const queryClient = useQueryClient();
  const preview = useQuery({
    queryKey: ["prereg-preview", name],
    enabled: open,
    staleTime: 0,
    gcTime: 0,
    retry: false,
    queryFn: async () => {
      const queued = await unwrap(api.POST("/api/studies/{name}/preregister/preview", { params: { path: { name } } }));
      return followJob<Preview>((queued as unknown as { job: { id: number } }).job.id);
    },
  });
  const commit = useMutation({
    mutationFn: async (hash: string) => {
      const queued = await unwrap(
        api.POST("/api/studies/{name}/preregister/commit", { params: { path: { name } }, body: { preview_hash: hash } }),
      );
      return followJob<{ commit: string }>((queued as unknown as { job: { id: number } }).job.id);
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["git"] });
      void queryClient.invalidateQueries({ queryKey: ["studies"] });
    },
  });

  const p = preview.data;
  const rows = p ? diffRows(p.diff) : [];
  const added = rows.filter((r) => r.kind === "add").length;
  const removed = rows.filter((r) => r.kind === "del").length;
  const blocked = p && (p.unrelated.length > 0 || !p.identity || p.nothing_to_commit);
  const file = `studies/${name}.toml`;

  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className={styles.scrim} />
        <AlertDialog.Content className={styles.dialog}>
          <div className={styles.block}>
            <AlertDialog.Title className="heading">Commit preregistration</AlertDialog.Title>
            <AlertDialog.Description className="body">
              {p ? (
                <>
                  This commits {p.files.length === 1 ? "one file" : `${p.files.length} files`},{" "}
                  {p.files.map((f) => (
                    <code key={f} className="value">{f}</code>
                  ))}
                  , after <code className="value">{p.head?.slice(0, 7) ?? "no commits"}</code>, with your git identity. The commit runs in a
                  worker, so your git hooks run there.
                </>
              ) : (
                <>Reading what a commit of <code className="value">{file}</code> would contain…</>
              )}
            </AlertDialog.Description>
          </div>

          {preview.error && <ProblemFromError error={preview.error} />}
          {p && p.unrelated.length > 0 && (
            <ProblemView
              title="Other files are changed"
              detail={`A preregistration commit holds only the spec. Commit or discard these yourself first:\n${p.unrelated.join("\n")}`}
            />
          )}
          {p && !p.identity && <ProblemView title="Git has no identity" detail="Set user.name and user.email (git config) so the commit has an author." />}
          {p && p.nothing_to_commit && <ProblemView title="Nothing to commit" detail={`${file} is already committed, unchanged.`} />}

          {p && !p.nothing_to_commit && (
            <div className={styles.block}>
              <div className={styles.legend}>
                <span className="label">Diff</span>
                <span className={`caption ${styles.muted}`}>
                  exactly what the commit contains · {added} lines added, {removed} removed
                </span>
              </div>
              <div className={`${styles.diff} code`} tabIndex={0} aria-label="Diff">
                {rows.map((r, i) => (
                  <div key={i} className={`${styles.line} ${styles[r.kind]}`}>
                    <span className={styles.sign} aria-hidden="true">{r.sign}</span>
                    <span>{r.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {p && (
            <div className={styles.block}>
              <span className="label">Commit message</span>
              <pre className={`${styles.message} code`}>{p.message}</pre>
            </div>
          )}

          <Note tone="info">
            A commit made here is recorded as <code className="value">committed_via: nanoscope</code> in <code className="value">study.json</code>,
            in every run's <code className="value">config.json</code> and in the report. A reader cannot tell how carefully this diff was
            read; a commit you make yourself with <code className="value">git commit</code> carries no such line.
          </Note>

          {commit.error && <ProblemFromError error={commit.error} />}
          {commit.data ? (
            <div className={styles.done} role="status">
              <span className="body-strong">Preregistration committed</span>
              <code className="value">{commit.data.commit}</code>
              <AlertDialog.Cancel asChild>
                <Button>Close</Button>
              </AlertDialog.Cancel>
            </div>
          ) : (
            <footer className={styles.foot}>
              {p && (
                <span className={`caption ${styles.muted}`}>
                  preview <code className="value">{p.hash}</code> · the commit is refused if the file or the repository changes before you
                  confirm
                </span>
              )}
              <span className={styles.spacer} />
              <AlertDialog.Cancel asChild>
                <Button>Cancel</Button>
              </AlertDialog.Cancel>
              <Button variant="primary" disabled={!p || !!blocked || commit.isPending} onClick={() => p && commit.mutate(p.hash)}>
                Commit preregistration
              </Button>
            </footer>
          )}
          <EquivalentCommand cli={`nanoscope study preregister ${file}`} />
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
