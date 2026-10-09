import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import type { Level } from "./level";

const KEY = "nanoscope.unlockOffered";
// Kept in memory too, so the offer still comes only once when storage is blocked.
let asked = false;

export function wasOffered(): boolean {
  if (asked) return true;
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

function markOffered(): void {
  asked = true;
  try {
    window.localStorage.setItem(KEY, "1");
  } catch {
    /* the offer is still not repeated in this page view */
  }
}

// For tests: forget that the offer was made.
export function resetOffer(): void {
  asked = false;
}

type View = { policy: string; unlocks: Record<string, unknown>; lockable: Record<string, { state: string }> };

// Switching to Research or Extend offers "Unlock all" once, with the same words as the Components
// page. It is never forced: the safe choice has the focus, and nothing changes if it is taken.
export function useUnlockOffer() {
  const queryClient = useQueryClient();
  const [view, setView] = useState<View | null>(null);
  const unlock = useMutation({
    mutationFn: () => unwrap(api.POST("/api/learn/unlock", { body: { all: true } })),
    onSuccess: async () => {
      setView(null);
      await queryClient.invalidateQueries({ queryKey: ["learn"] });
    },
  });

  const onSwitch = async (level: Level) => {
    if ((level !== "Research" && level !== "Extend") || wasOffered()) return;
    try {
      const got = (await queryClient.fetchQuery({
        queryKey: ["learn", "unlocks"],
        queryFn: () => unwrap(api.GET("/api/learn/unlocks")),
        staleTime: 0,
      })) as unknown as View;
      if (got.policy === "guided" && Object.values(got.lockable).some((e) => e.state === "locked")) {
        markOffered();
        setView(got);
      }
    } catch {
      /* no offer if the state cannot be read; the Components page still has Unlock all */
    }
  };

  const locked = view ? Object.values(view.lockable).filter((e) => e.state === "locked").length : 0;
  const earned = view ? Object.values(view.lockable).filter((e) => e.state === "earned").length : 0;
  const skipped = view ? Object.values(view.lockable).filter((e) => e.state === "skipped").length : 0;
  const dialog = (
    <ConfirmDialog
      open={view !== null}
      onOpenChange={(open) => !open && setView(null)}
      title="You switched to Research. Unlock every block too?"
      confirm="Unlock all"
      cancel="Keep the locks"
      busy={unlock.isPending}
      onConfirm={() => unlock.mutate()}
    >
      <p>
        Lessons still lock {locked} blocks and features, so your own models can only use what you have earned (the shipped models always
        run). Unlock all changes the policy from <span className="value">guided</span> to <span className="value">open</span>; the {earned}{" "}
        you earned and the {skipped} you skipped stay recorded as they are.
      </p>
      <p className="small" style={{ color: "var(--ink-muted)" }}>
        Nothing changes if you keep the locks. This is asked once; Unlock all stays on Components, and{" "}
        <span className="value">nanoscope learn lock --reset</span> turns gating back on.
      </p>
      {unlock.error && <ProblemFromError error={unlock.error} />}
      <EquivalentCommand cli="nanoscope learn unlock --all" />
    </ConfirmDialog>
  );
  return { onSwitch, dialog };
}
