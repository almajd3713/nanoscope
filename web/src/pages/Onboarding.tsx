import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "../components/Button";
import { EquivalentCommand } from "../components/EquivalentCommand";
import { ProblemFromError } from "../components/ProblemView";
import styles from "./Onboarding.module.css";

type Policy = "guided" | "open";

export function Onboarding() {
  const [policy, setPolicy] = useState<Policy>("guided");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const choose = useMutation({
    mutationFn: (chosen: Policy) => unwrap(api.POST("/api/learn/policy", { body: { policy: chosen } })),
    onSuccess: async (view) => {
      // The answer is the new unlocks view (first_run is false now). Put it in the cache before
      // leaving, or the lessons page would read the old "first run" and send you back here.
      queryClient.setQueryData(["learn", "unlocks"], view);
      await queryClient.invalidateQueries({ queryKey: ["learn"] });
      navigate("/learn");
    },
  });

  return (
    <div className={styles.page}>
      <div className={styles.intro}>
        <h1 className="title">How do you want to start?</h1>
        <p className={`body ${styles.muted}`}>
          nanoscope can hold back the bigger blocks (attention, the transformer block, the decoder stack) until you
          have built each one yourself in a lesson. It is a learning aid, and you can change it any time on
          Components.
        </p>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          choose.mutate(policy);
        }}
        className={styles.page}
      >
        <fieldset className={styles.choices} aria-label="Gating policy">
          <label className={styles.choice}>
            <input
              type="radio"
              name="policy"
              value="guided"
              checked={policy === "guided"}
              onChange={() => setPolicy("guided")}
            />
            <span className={styles.text}>
              <span className="body-strong">I'm learning</span>
              <span className={`small ${styles.muted}`}>
                Lessons unlock blocks as you pass them. You write attention from matrix multiplications before you can
                import <span className="value">Attention</span>.
              </span>
              <span className={`caption ${styles.muted}`}>
                Policy <span className="value">guided</span>
              </span>
            </span>
          </label>
          <label className={styles.choice}>
            <input
              type="radio"
              name="policy"
              value="open"
              checked={policy === "open"}
              onChange={() => setPolicy("open")}
            />
            <span className={styles.text}>
              <span className="body-strong">I know this</span>
              <span className={`small ${styles.muted}`}>
                Every block is available now. The lessons stay here if you want them.
              </span>
              <span className={`caption ${styles.muted}`}>
                Policy <span className="value">open</span>
              </span>
            </span>
          </label>
        </fieldset>
        {choose.error && <ProblemFromError error={choose.error} />}
        <div>
          <Button type="submit" variant="primary" disabled={choose.isPending}>
            Continue to lessons
          </Button>
        </div>
      </form>
      <EquivalentCommand
        cli={
          "nanoscope learn start foundations/01-bigram   # the first start turns on guided\nnanoscope learn unlock --all                   # open: everything unlocked"
        }
      />
    </div>
  );
}
