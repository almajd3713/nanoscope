import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { Button } from "./Button";
import { ProblemFromError } from "./ProblemView";
import styles from "./GeneratePanel.module.css";

// Text from a trained run: a worker keeps the model loaded and answers. The settings are the
// API's defaults (200 tokens, temperature 0.8, seed 42).
export function GeneratePanel({ runRef }: { runRef: string }) {
  const [prompt, setPrompt] = useState("");
  const generate = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/runs/{ref}/generate", {
          params: { path: { ref: runRef } },
          body: { prompt, max_new_tokens: 200, temperature: 0.8, seed: 42, timeout: 30 },
        }),
      ) as Promise<{ text: string }>,
  });
  return (
    <section className={styles.panel} aria-label="Generate text">
      <h2 className="heading">Generate text</h2>
      <label className={styles.panel}>
        <span className="label">Prompt</span>
        <textarea
          className={`${styles.input} code`}
          rows={2}
          value={prompt}
          placeholder="Once upon a time"
          onChange={(e) => setPrompt(e.target.value)}
        />
      </label>
      <div>
        <Button disabled={generate.isPending} onClick={() => generate.mutate()}>
          Generate
        </Button>
      </div>
      {generate.isPending && <p className={`small ${styles.muted}`}>Waiting for a worker to generate…</p>}
      {generate.error && <ProblemFromError error={generate.error} />}
      {generate.data && <p className={`${styles.out} code`}>{generate.data.text}</p>}
    </section>
  );
}
