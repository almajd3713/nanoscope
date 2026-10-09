import * as AlertDialog from "@radix-ui/react-alert-dialog";
import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { ArrowSquareOut } from "../icons";
import { followJob } from "../studies/useStudyChecks";
import { Button } from "./Button";
import { EquivalentCommand } from "./EquivalentCommand";
import { Field } from "./Field";
import { Note } from "./Note";
import { ProblemFromError } from "./ProblemView";
import preregStyles from "./PreregDialog.module.css";
import styles from "./CardExport.module.css";

type Plan = { repo: string; path_in_repo: string; content: string; commit_message: string; exported_at: string };

export const PROJECT_CARDS = "RedhouaneLazib/nanoscope-ablation-cards";
const validRepo = (repo: string) => /^[^/\s]+\/[^/\s]+$/.test(repo.trim());

const kb = (n: number) => (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} kB`);

// The ablation card: one JSON file, shown whole. Pushing it to a public dataset is a separate,
// opt-in step each time, and the dialog says exactly which file, where, and that it is public.
export function CardExport({ name, open, onOpenChange }: { name: string; open: boolean; onOpenChange: (open: boolean) => void }) {
  const [repo, setRepo] = useState(PROJECT_CARDS);
  const [push, setPush] = useState(false);
  // the stamp of the card first shown; every later plan and the push keep it
  const [stamp, setStamp] = useState<string | undefined>(undefined);
  const target = validRepo(repo) ? repo.trim() : PROJECT_CARDS;

  const plan = useQuery({
    queryKey: ["card-plan", name, target, stamp],
    enabled: open,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async () => {
      const got = (await unwrap(
        api.GET("/api/studies/{name}/card/upload", { params: { path: { name }, query: { repo: target, ...(stamp ? { exported_at: stamp } : {}) } } }),
      )) as unknown as Plan;
      if (!stamp) setStamp(got.exported_at);
      return got;
    },
  });
  const pushed = useMutation({
    mutationFn: async () => {
      const queued = await unwrap(
        api.POST("/api/studies/{name}/card/push", { params: { path: { name } }, body: { repo: repo.trim(), exported_at: stamp ?? null } }),
      );
      return followJob<{ repo: string; path_in_repo: string }>((queued as unknown as { job: { id: number } }).job.id);
    },
  });

  const p = plan.data;
  const lines = p ? p.content.split("\n").length : 0;
  const bytes = p ? new TextEncoder().encode(p.content).length : 0;
  const canPush = push && validRepo(repo) && !!p && !pushed.isPending && !pushed.data;

  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className={preregStyles.scrim} />
        <AlertDialog.Content className={preregStyles.dialog}>
          <div className={preregStyles.block}>
            <AlertDialog.Title className="heading">Ablation card for {name}</AlertDialog.Title>
            <AlertDialog.Description className="body">
              One JSON file: the spec, every seed's final metrics and the provenance (the preregistration commit, when the study started). No
              run folders, checkpoints, samples or paths from this machine. Only record-mode studies have cards.
            </AlertDialog.Description>
          </div>

          {plan.error && !p && <ProblemFromError error={plan.error} />}
          {p && (
            <div className={preregStyles.block}>
              <div className={styles.file}>
                <code className="value-strong">card.json</code>
                <span className={`caption ${preregStyles.muted}`}>
                  card.v1 · {lines} lines · {kb(bytes)}
                </span>
                <span className={preregStyles.spacer} />
                <a
                  className="body-strong"
                  href={`data:application/json;charset=utf-8,${encodeURIComponent(p.content)}`}
                  download="card.json"
                >
                  <ArrowSquareOut size={16} aria-hidden="true" /> Download card.json
                </a>
              </div>
              <pre className={`${styles.content} code`} tabIndex={0} aria-label="card.json">{p.content}</pre>
            </div>
          )}

          <div className={styles.push}>
            <div className={styles.toggle}>
              <input id="push-card" type="checkbox" checked={push} onChange={(e) => setPush(e.target.checked)} />
              <label htmlFor="push-card">
                <span className="body-strong">Also push it to a public Hugging Face dataset</span>
                <span className={`small ${preregStyles.muted}`}>Off unless you turn it on, every time. Nothing is pushed on its own.</span>
              </label>
            </div>
            {push && (
              <>
                <div className={styles.repo}>
                  <Field label="Dataset" value={repo} onChange={setRepo} problem={validRepo(repo) ? undefined : "Use user/name"} />
                </div>
                {p && validRepo(repo) && (
                  <div className={preregStyles.block}>
                    <span className="label">What uploads</span>
                    <dl className={`${styles.plan} small`}>
                      <dt className={preregStyles.muted}>To</dt>
                      <dd><code className="value">{repo.trim()}</code>, a public dataset</dd>
                      <dt className={preregStyles.muted}>File</dt>
                      <dd><code className="value">{p.path_in_repo}</code>: the text above, byte for byte, and nothing else</dd>
                      <dt className={preregStyles.muted}>Commit message</dt>
                      <dd><code className="value">{p.commit_message}</code></dd>
                      <dt className={preregStyles.muted}>Runs as</dt>
                      <dd>a <code className="value">card-push</code> job on a worker that has <code className="value">HF_TOKEN</code></dd>
                    </dl>
                  </div>
                )}
                <Note tone="warn">Anyone can read a public dataset, and a pushed card may be copied or indexed even if you delete it later.</Note>
              </>
            )}
          </div>

          {pushed.error && <ProblemFromError error={pushed.error} />}
          {pushed.data && (
            <p className="body-strong" role="status">
              Pushed <code className="value">{pushed.data.path_in_repo}</code> to <code className="value">{pushed.data.repo}</code>
            </p>
          )}
          <footer className={preregStyles.foot}>
            <span className={preregStyles.spacer} />
            <AlertDialog.Cancel asChild>
              <Button>{pushed.data ? "Close" : "Cancel"}</Button>
            </AlertDialog.Cancel>
            {push && (
              <Button variant="primary" disabled={!canPush} onClick={() => pushed.mutate()}>
                Push card
              </Button>
            )}
          </footer>
          <EquivalentCommand
            cli={`nanoscope card export studies/${name}.toml${push ? `\nnanoscope card push experiments/${name}/card.json --repo ${repo.trim()}` : ""}`}
          />
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
