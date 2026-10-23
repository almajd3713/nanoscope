import * as Dialog from "@radix-ui/react-dialog";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { unwrap } from "../api/problem";
import { useBlocks } from "../graph/api";
import { Button } from "./Button";
import { Field } from "./Field";
import { ProblemFromError } from "./ProblemView";
import styles from "./Dialog.module.css";

export const CLASS_NAME = /^[A-Z][A-Za-z0-9]*$/;

// MyModel -> my_model, the file name a class gets under models/
export function fileFor(name: string): string {
  return `models/${name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase()}.py`;
}

const COMPOSED = ["Decoder", "Block", "Attention", "RMSNorm", "RoPE", "SwiGLU"];

type Known = { name: string; lock?: { locked?: boolean } | null }[];

// Whether a model written with blocks can be started: every block the starter names is unlocked.
// `locked` lists the ones still waiting on a lesson (what the empty states and this dialog say).
export function blockModelAccess(known: Known): { available: boolean; locked: string[] } {
  const locked = COMPOSED.filter((n) => known.some((b) => b.name === n && b.lock?.locked));
  return { available: COMPOSED.every((n) => known.some((b) => b.name === n)) && locked.length === 0, locked };
}

// Where the starter begins: the Modern composition when every block it names is unlocked (the model
// page then draws it), else a bigram written with primitives, which builds under any gating policy.
export function starterFor(name: string, composed: boolean): string {
  if (composed) {
    return `from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, RoPE, SwiGLU


class ${name}(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256):
        super().__init__(
            vocab_size, context_length, d_model=128, n_layers=4,
            block=Block(norm=RMSNorm(), mlp=SwiGLU(hidden=344),
                        attn=Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True)),
            final_norm=RMSNorm(),
            tie_weights=True, z_loss=1e-4,
        )
`;
  }
  return `import torch
import torch.nn as nn


class ${name}(nn.Module):
    """Predicts the next token from the current token alone. Change it, then Train."""

    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.token_embedding(idx))
`;
}

type Props = { open: boolean; onOpenChange: (open: boolean) => void };

// Makes models/<name>.py in the workspace and opens it on the model page. The file is new: the
// server refuses (428) to overwrite one that exists, and says so here.
export function NewModelDialog({ open, onOpenChange }: Props) {
  const [name, setName] = useState("MyModel");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const blocks = useBlocks();
  const access = blockModelAccess(blocks.data?.blocks ?? []);
  const [picked, setPicked] = useState<"blocks" | "plain" | null>(null);
  const composed = (picked ?? (access.available ? "blocks" : "plain")) === "blocks" && access.available;
  const valid = CLASS_NAME.test(name);
  const file = fileFor(name);
  const create = useMutation({
    mutationFn: () => unwrap(api.PUT("/api/files/{path}", { params: { path: { path: file } }, body: { content: starterFor(name, composed) } })),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["models"] });
      onOpenChange(false);
      navigate(`/model/${file}?class=${name}`);
    },
  });
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.scrim} />
        <Dialog.Content className={styles.dialog} aria-describedby={undefined}>
          <Dialog.Title className="heading">New model</Dialog.Title>
          <div className={styles.body}>
            <p className="body">A starter in your workspace. Open it on the model page to change it.</p>
            <fieldset className={styles.choice}>
              <legend className="label">Starting point</legend>
              <label className={`small ${styles.option}`}>
                <input type="radio" name="kind" checked={composed} disabled={!access.available} onChange={() => setPicked("blocks")} />
                <span>
                  <span className="body-strong">Built from blocks</span>
                  <br />
                  A small modern model. The model page draws it as a graph, and you can drag a block onto another to swap it.
                  {!access.available && access.locked.length > 0 && (
                    <>
                      {" "}
                      <span className={styles.note}>Locked until their lessons are passed: {access.locked.join(", ")}. Settings can open them all.</span>
                    </>
                  )}
                </span>
              </label>
              <label className={`small ${styles.option}`}>
                <input type="radio" name="kind" checked={!composed} onChange={() => setPicked("plain")} />
                <span>
                  <span className="body-strong">Plain PyTorch</span>
                  <br />
                  A bigram written with nn.Module. Code only: there is no graph to draw.
                </span>
              </label>
            </fieldset>
            <Field label="Class name" value={name} onChange={setName} problem={valid ? undefined : "Start with a capital letter; letters and digits only."} />
            <p className="small">
              {valid ? <>Creates <span className="value">{file}</span>.</> : null}
            </p>
            {create.error && <ProblemFromError error={create.error} />}
          </div>
          <div className={styles.foot}>
            <Dialog.Close asChild>
              <Button>Cancel</Button>
            </Dialog.Close>
            <Button variant="primary" disabled={!valid || create.isPending} onClick={() => create.mutate()}>
              Create {valid ? file : "model"}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
