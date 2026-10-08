import * as Dialog from "@radix-ui/react-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { ConfirmDialog } from "../components/ConfirmDialog";
import dialog from "../components/Dialog.module.css";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import { Tag } from "../components/Tag";
import { stampText } from "../format";
import { LockSimple, LockSimpleOpen } from "../icons";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import styles from "./Components.module.css";

type Entry = { lesson: string; state: string; reason: string | null };
type Unlock = { how: string; lesson?: string; at?: string; evidence?: string | null; reason?: string | null };

function StateCell({ state }: { state: string }) {
  switch (state) {
    case "earned":
      return <Tag tone="good" icon={LockSimpleOpen}>earned</Tag>;
    case "skipped":
      return <Tag tone="neutral" icon={LockSimpleOpen}>skipped</Tag>;
    case "open":
      return <Tag tone="muted" icon={LockSimpleOpen}>open</Tag>;
    default:
      return <Tag tone="muted" icon={LockSimple}>locked</Tag>;
  }
}

export function Components() {
  const queryClient = useQueryClient();
  const view = useQuery({ queryKey: ["learn", "unlocks"], queryFn: () => unwrap(api.GET("/api/learn/unlocks")) });
  const [confirmAll, setConfirmAll] = useState(false);
  const [unlockOne, setUnlockOne] = useState<string | null>(null);
  const [reason, setReason] = useState("");

  const done = () => queryClient.invalidateQueries({ queryKey: ["learn"] });
  const all = useMutation({
    mutationFn: () => unwrap(api.POST("/api/learn/unlock", { body: { all: true } })),
    onSuccess: async () => {
      setConfirmAll(false);
      await done();
    },
  });
  const one = useMutation({
    mutationFn: (id: string) => unwrap(api.POST("/api/learn/unlock", { body: { id, reason: reason.trim(), all: false } })),
    onSuccess: async () => {
      setUnlockOne(null);
      setReason("");
      await done();
    },
  });

  if (view.error) return <ProblemFromError error={view.error} />;
  if (!view.data) return <p className={`body ${styles.muted}`}>Loading components…</p>;

  const lockable = Object.entries(view.data.lockable as Record<string, Entry>).sort(([a], [b]) => a.localeCompare(b));
  const unlocks = view.data.unlocks as Record<string, Unlock>;
  const count = (s: string) => lockable.filter(([, e]) => e.state === s).length;
  const locked = count("locked");

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <div className={styles.titles}>
          <h1 className="title">Components</h1>
          <p className={`body ${styles.muted} ${styles.prose}`}>
            Policy <span className="value">{view.data.policy}</span>
            {view.data.policy === "guided"
              ? ": a block unlocks when you pass the lesson that builds it. "
              : ": every block is available. "}
            This is a learning aid, not a permission: the shipped models (<span className="value">GPT2</span>,{" "}
            <span className="value">Modern</span>, <span className="value">Bigram</span>) always run.
          </p>
        </div>
        {locked > 0 && (
          <Button variant="danger" onClick={() => setConfirmAll(true)}>
            Unlock all
          </Button>
        )}
      </header>

      <div className={`small ${styles.counts}`}>
        {(["earned", "skipped", "open", "locked"] as const)
          .filter((s) => count(s) > 0)
          .map((s) => (
            <span key={s}>
              <strong>{count(s)}</strong> {s}
            </span>
          ))}
      </div>

      <div className={styles.wrap}>
        <table className={`${styles.table} small`}>
          <thead>
            <tr>
              <th className="label" scope="col">Block or feature</th>
              <th className="label" scope="col">State</th>
              <th className="label" scope="col">Lesson that unlocks it</th>
              <th className="label" scope="col">How</th>
              <th className="label" scope="col">
                <VisuallyHidden>Action</VisuallyHidden>
              </th>
            </tr>
          </thead>
          <tbody>
            {lockable.map(([id, entry]) => {
              const u = unlocks[id];
              return (
                <tr key={id}>
                  <td className="value-strong">{id}</td>
                  <td>
                    <StateCell state={entry.state} />
                  </td>
                  <td>
                    <Link to={`/learn/${entry.lesson}`} className={`value ${styles.link}`}>
                      {entry.lesson}
                    </Link>
                  </td>
                  <td className={entry.state === "skipped" ? styles.reason : undefined}>
                    {u?.how === "earned" && (
                      <>
                        check passed {u.at && <span className={`value ${styles.muted}`}>{stampText(u.at)}</span>} ·{" "}
                        <Link to={`/learn/${entry.lesson}`} className={styles.link}>
                          evidence
                        </Link>
                      </>
                    )}
                    {u?.how === "skipped" && (
                      <>
                        “{u.reason}” {u.at && <span className={`value ${styles.muted}`}>{stampText(u.at)}</span>}
                      </>
                    )}
                    {!u && <span className={styles.muted}>–</span>}
                  </td>
                  <td>
                    {entry.state === "locked" && (
                      <Button variant="quiet" size="sm" onClick={() => setUnlockOne(id)}>
                        Unlock one…
                      </Button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <span className={`small ${styles.muted}`}>
        Unlocking one asks for a reason, which is kept with it. <span className="value">nanoscope learn lock --reset</span> turns gating
        back on and keeps what you earned or skipped. Primitives (Linear, the embeddings, LayerNorm, GELU MLP, the templates) are never
        locked.
      </span>
      <EquivalentCommand cli={'nanoscope learn status\nnanoscope learn unlock block:Block --reason "…"\nnanoscope learn unlock --all'} />

      <ConfirmDialog
        open={confirmAll}
        onOpenChange={setConfirmAll}
        title="Unlock every block?"
        confirm="Unlock all"
        busy={all.isPending}
        onConfirm={() => all.mutate()}
      >
        <p>
          Every lesson-gated block and feature becomes available, and the policy changes from <span className="value">guided</span> to{" "}
          <span className="value">open</span>. Your lesson progress and the {Object.keys(unlocks).length} unlocks already recorded stay as
          they are.
        </p>
        <p className={`small ${styles.muted}`}>
          Writes <span className="value">learn/unlocks.json</span>. Turn gating back on later with{" "}
          <span className="value">nanoscope learn lock --reset</span>.
        </p>
        {all.error && <ProblemFromError error={all.error} />}
      </ConfirmDialog>

      <Dialog.Root
        open={unlockOne !== null}
        onOpenChange={(o) => {
          if (!o) {
            setUnlockOne(null);
            setReason("");
          }
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={dialog.scrim} />
          <Dialog.Content className={dialog.dialog}>
            <Dialog.Title className="heading">Unlock {unlockOne}?</Dialog.Title>
            <Dialog.Description className={`small ${styles.muted}`}>
              Skipping a lesson needs a reason, which is kept with the unlock.
            </Dialog.Description>
            <label className={styles.field}>
              <span className="label">Reason</span>
              <textarea className={`${styles.input} body`} rows={3} value={reason} onChange={(e) => setReason(e.target.value)} />
            </label>
            {one.error && <ProblemFromError error={one.error} />}
            <div className={dialog.foot}>
              <Dialog.Close asChild>
                <Button>Cancel</Button>
              </Dialog.Close>
              <Button
                variant="primary"
                disabled={!reason.trim() || one.isPending}
                onClick={() => unlockOne && one.mutate(unlockOne)}
              >
                Unlock
              </Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
