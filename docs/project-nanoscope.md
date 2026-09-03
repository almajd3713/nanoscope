# Project: `nanoscope`

*A small-model laboratory for architecture, inference efficiency, and interpretability — built entirely on free-tier GPUs, using only pre-existing public datasets.*

---

## Premise

One repository. One model family. Seven milestones that compound.

The organizing idea: **at small scale you cannot compete on capability, so compete on rigor.** A 50M-parameter model will never impress anyone. A correctly-fit scaling law that *predicts a held-out run's loss to within 2%* will, because almost nobody does it and it demonstrates you understand the methodology that frontier labs actually run on.

The three areas you chose interlock naturally:
- Architecture gives you models to study.
- Efficiency gives you the tools to run more experiments in the same compute.
- Evaluation and interpretability give you the ability to tell whether any of it worked.

Every milestone produces a **falsifiable claim with error bars**, not a demo.

---

## Compute reality check

Do this arithmetic before you start; it's what makes the plan credible rather than aspirational.

**Hardware.** Kaggle is your primary compute: 30 GPU-hours/week, sessions up to 12 hours, T4×2 or P100. Colab free is the scratchpad: T4, ~4-hour sessions, unpredictable.

**Two constraints specific to T4 (Turing, sm_75):**
- **No bf16.** Use fp16 with `GradScaler`, and watch for overflow. This is itself a lesson — you'll feel exactly why the field moved to bf16.
- **FlashAttention-2 requires Ampere or newer.** Use `torch.nn.functional.scaled_dot_product_attention` with the memory-efficient backend. Don't waste a day trying to install flash-attn.

**Throughput planning number.** T4 peak is ~65 TFLOPS fp16. At a realistic 15–25% MFU for small models, plan on **~12 TFLOPS effective**.

**Training cost:** `C ≈ 6ND` FLOPs (N = non-embedding params, D = tokens).

| N (non-emb) | D (Chinchilla, 20N) | FLOPs | Wall-clock @12 TFLOPS |
|---|---|---|---|
| 2M | 40M | 4.8e14 | ~40 s |
| 5M | 100M | 3.0e15 | ~4 min |
| 12M | 240M | 1.7e16 | ~24 min |
| 30M | 600M | 1.1e17 | ~2.5 hr |
| **70M** | **1.4B** | **5.9e17** | **~14 hr** |

So: the fitting ladder (2M → 30M, three seeds each) costs roughly **9 GPU-hours**. The 70M held-out prediction run costs **~14 hours**, spread across two or three checkpointed Kaggle sessions. The whole scaling study fits in **one week of Kaggle quota**. That is the single most important number in this document — it means the crown-jewel experiment is genuinely affordable.

**Corollary:** checkpoint/resume is not a nice-to-have. It is milestone zero. A 14-hour run on a platform that kills sessions must survive being killed.

---

## Data — all pre-existing, nothing collected

| Purpose | Dataset | Notes |
|---|---|---|
| Main pretraining | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | Stream it; never store more than a few GB |
| Fast iteration + interp | `roneneldan/TinyStories` | ~1–10M models produce coherent text; runs in minutes |
| Synthetic/controlled | Generate your own toy tasks (modular arithmetic, induction sequences, bio-style synthetic facts) | Zero collection cost, full ground truth |
| Code/math mix ablation | `bigcode/the-stack-smol`, `open-web-math` | For data-mixture experiments |
| Eval | `lm-evaluation-harness` tasks: LAMBADA, HellaSwag, ARC-Easy, PIQA, WinoGrande | Expect near-chance on some — that's a finding, not a failure |
| Tokenizer | GPT-2 or Llama tokenizer off-the-shelf | Don't train your own until M3 |

Checkpoints go to the **HF Hub**, not Drive. Free, versioned, resumable from any machine, and it forces you to write model cards.

---

## Repo layout

```
nanoscope/
  configs/          # YAML; every run is fully specified by one file
  nanoscope/
    data/           # streaming loaders, packing, deterministic shuffling
    model/          # blocks: attention variants, norms, MoE, SSM
    train/          # loop, optimizers (AdamW/Muon), muP, WSD schedule, ckpt
    infer/          # KV cache, batching, quantization, spec decoding
    eval/           # harness + statistics (bootstrap, paired tests)
    interp/         # hooks, patching, SAE, probes
  experiments/      # one dir per study, with results.json + plots
  reports/          # the write-ups
  tests/            # numerical equivalence tests — see below
```

**One discipline to adopt from day one:** every architecture variant ships with a test asserting numerical equivalence to a naive reference implementation. Your GQA must match a loop-over-heads version to 1e-5. Your KV-cache decode must match full recomputation. This catches the class of bug that silently produces a "2% improvement."

---

## Milestones

### M0 — Infrastructure that survives (≈8 hrs)

Boring, and everything depends on it.

- Streaming FineWeb-Edu loader with document packing and deterministic, resumable shuffling.
- Checkpoint/resume: model, optimizer, scheduler, dataloader position, RNG state. Test by killing the process mid-run and confirming the loss curve is bit-identical on resume.
- Config-driven runs; W&B logging; automatic FLOP and MFU accounting logged every step.
- Seed control across torch/numpy/python/CUDA.

**Deliverable:** a run that you kill three times and that produces a loss curve indistinguishable from an uninterrupted one.
**Acceptance:** resumed loss matches uninterrupted loss to <0.1% at every logged step.
**Cements:** nothing glamorous. But without it, M2 is impossible.

---

### M1 — The modern baseline (≈10 hrs)

Build a decoder-only LM with the 2024–26 consensus block: RoPE, RMSNorm (pre-norm), SwiGLU, GQA, QK-norm, no biases, weight tying, z-loss.

Then build a **GPT-2-style control**: learned positional embeddings, LayerNorm, GELU MLP, MHA.

Train both at matched non-embedding parameter count *and* matched FLOPs on FineWeb-Edu. Three seeds each.

**Deliverable:** a plot with confidence bands showing the gap, and a written breakdown of which component contributed what (leave-one-out: modern block minus RoPE, minus SwiGLU, etc.).
**Acceptance:** you can state the modern-block advantage as a number with a confidence interval, and correctly identify at least one component that contributes *nothing* at this scale.
**Cements:** Spine items 14–20. And the first hard lesson — at 30M params, several "essential" components will be within noise.

---

### M2 — The scaling law *(the crown jewel)* (≈25 hrs)

This is the milestone that makes the project worth doing.

1. **Parametrize with muP** so that learning rate transfers across width. Verify the transfer empirically: sweep LR at width 128 and width 512, and show the optimal LR coincides. If it doesn't, your muP implementation is wrong — and finding that out is half the value.
2. Train the ladder: N ∈ {2M, 5M, 12M, 30M}, three seeds each, at Chinchilla-ish D = 20N.
3. Additionally run an **isoFLOP sweep**: at each of three compute budgets, train 4–5 (N, D) combinations along the isoFLOP contour and locate the minimum. This is the Chinchilla Approach 2 methodology.
4. Fit `L(N, D) = E + A/N^α + B/D^β` with proper nonlinear least squares. Report parameter uncertainties. Compare your α, β to Chinchilla's — and read the Besiroglu replication before you trust either.
5. **Pre-register a prediction.** Write down, in the repo, with a commit hash and timestamp, the loss you predict for a 70M model at 1.4B tokens. Then train it.
6. Plot predicted vs. actual.

**Deliverable:** a predicted-vs-actual plot, and a pre-registration commit that predates the run.
**Acceptance:** prediction within 3% of actual loss — or a written diagnosis of why it missed. Both outcomes are publishable-quality learning; the second is more interesting.
**Cements:** Spine 6–12. This is the experiment that demonstrates you understand scaling as a *predictive methodology* rather than a slogan.

**Bonus that costs nothing:** the 12M model from this ladder becomes your speculative-decoding draft model in M4, and the checkpoint sequence from the 30M run becomes your interpretability subject in M6. Plan the artifacts to be reused.

---

### M3 — Architecture ablations, honestly (≈20 hrs)

At matched FLOPs, using muP so hyperparameters transfer, three seeds each:

| Ablation | Question |
|---|---|
| GQA group ratio (1, 2, 4, MHA) | Quality vs. KV-cache size curve |
| **MLA** vs GQA | Does low-rank KV compression beat head sharing at equal cache bytes? |
| Depth vs width at fixed params | Where's the optimum, and does it move with scale? |
| RoPE vs NoPE vs ALiBi | Is explicit positional encoding necessary in a causal decoder? |
| Sliding-window : global layer ratio | Reproduce the Gemma-style hybrid tradeoff |
| Dense vs MoE (upcycled) | Take your 30M dense model, split FFNs into 8 experts, continue training |
| Muon vs AdamW | Does the optimizer story hold at 30M? |

**Deliverable:** an ablation table where every cell has a bootstrap confidence interval, plus an explicit "**Null results**" section listing every change that did nothing.
**Acceptance:** at least three entries in the null-results section, and you can explain for each whether it's genuinely useless or just invisible at 30M params.
**Cements:** Track A, most of it.

The discipline here is the point. The default failure mode of small-scale architecture research is running one seed, seeing a 1.5% improvement, and believing it. M5's statistics feed back into this milestone; expect to invalidate one of your own early findings.

---

### M4 — Inference and efficiency (≈20 hrs)

Your own model, your own serving stack. Build in this order:

1. **KV cache from scratch.** Then verify: cached decode must be numerically identical to full recomputation.
2. **Roofline analysis.** Measure and plot tokens/sec vs. batch size. Compute arithmetic intensity for prefill and decode separately. Identify the batch size at which decode transitions from memory-bound to compute-bound, and check it against the T4's 320 GB/s bandwidth. *This plot is the single most educational artifact in the milestone.*
3. **Continuous batching + prefix caching.** Implement a simple scheduler. Measure throughput gain on a workload with shared prefixes.
4. **Quantization.** Implement INT8 and INT4 weight-only quantization yourself (round-to-nearest first, then GPTQ or AWQ). Measure perplexity degradation vs. memory saved. Then find the outlier channels and show *why* naive per-tensor quantization fails — connect this to the massive-activations literature.
5. **Speculative decoding.** Use the 12M model from M2 as the draft, the 70M as the target. Measure acceptance rate and end-to-end speedup. Sweep draft length γ. Verify output distribution equivalence.
6. **One Triton kernel.** Fused RMSNorm, or fused RoPE. Benchmark against the PyTorch eager version and against `torch.compile`. Expect `torch.compile` to beat your first attempt — that's a useful humbling.

**Deliverable:** a latency/throughput Pareto plot across all configurations, and a written roofline analysis explaining every point on it.
**Acceptance:** you can predict, before measuring, whether a given optimization will help at a given batch size — and be right.
**Cements:** Track B, most of it.

---

### M5 — Evaluation as a discipline (≈12 hrs)

- Integrate `lm-evaluation-harness`. Run your models on LAMBADA, HellaSwag, ARC-Easy, PIQA. Most will be near chance. Report that honestly, with CIs, and explain why length-normalized loglikelihood scoring behaves the way it does at this scale.
- Build **your own** harness for things small models *can* do: TinyStories grammaticality, induction-task accuracy, synthetic factual recall, held-out perplexity by domain.
- Implement the statistics: bootstrap CIs, **paired** bootstrap for model comparisons, clustered standard errors when items are grouped.
- **Variance decomposition:** measure how much of your observed eval variance comes from seed vs. prompt format vs. decoding temperature vs. eval-set sampling. Publish the breakdown.
- **Contamination check:** your training data is FineWeb-Edu and your eval sets are public. Search for overlap. Report what you find.
- **Go back to M3 and re-analyze every ablation with paired tests.** Expect at least one earlier "finding" to evaporate.

**Deliverable:** a variance-decomposition figure, and a revised M3 table with corrected significance.
**Acceptance:** you overturn one of your own prior conclusions and document it.
**Cements:** Track C evaluation half. This is the milestone that most distinguishes a serious researcher from a benchmark-runner.

---

### M6 — Interpretability on a model you built (≈20 hrs)

The advantage of studying your own model: you know the training data, you have every checkpoint, and you can retrain with a variable changed.

1. **Induction heads.** Take the checkpoint sequence from M2's 30M run. Measure prefix-matching score and in-context-learning score at every checkpoint. Reproduce the induction bump — the phase change where ICL ability appears alongside induction head formation. *Doing this on a model you trained yourself, with your own checkpoints, is a substantially stronger artifact than doing it on GPT-2.*
2. **Logit lens and tuned lens** on your model. Where do predictions crystallize by depth?
3. **Activation patching** on a synthetic task with known ground truth (e.g. indirect object identification analogue, or a synthetic factual-recall task in the Physics-of-LLMs style). Then **attribution patching** and a comparison of the two — cost vs. fidelity.
4. **Toy Models of Superposition** — reproduce it directly. It's small.
5. **Train an SAE** on your 30M model's residual stream. Then evaluate it honestly: reconstruction loss, L0, downstream loss recovered, feature interpretability by autointerp scoring, and — critically — **feature consistency across two SAEs trained with different seeds**. The 2025–26 literature is skeptical for good reasons; find out for yourself whether your features are stable.
6. **Optional, high-value:** a Physics-of-LLMs-style controlled experiment. Generate synthetic biographies, train on them, and measure knowledge extraction vs. storage as a function of data augmentation. Pure synthetic data, tiny models, sharp causal conclusions.

**Deliverable:** an induction-bump figure from your own checkpoints, an SAE quality report including the seed-consistency result, and one causal circuit story on a synthetic task.
**Acceptance:** you can state one thing you now believe about transformers that you learned from your own measurements rather than from a paper.
**Cements:** Track C interpretability half.

---

### M7 — The write-up (≈10 hrs)

A single technical report, in the style of an OLMo or FineWeb ablation paper:

- Methods, with full reproducibility details.
- The scaling law with predicted-vs-actual as the headline figure.
- The ablation table with null results given equal prominence.
- The inference Pareto frontier with roofline analysis.
- The interpretability findings, including the negative ones.
- A **limitations** section that is honest about what 30–70M parameters can and cannot tell you about frontier models. This section is the one that will most impress a reader who knows the field.

Publish the repo, the model checkpoints on HF Hub, and the report. Total artifact: one repo, ~6 trained model families, ~40 logged runs, one report.

---

## Estimated total

| Milestone | Build hours | GPU hours |
|---|---|---|
| M0 Infra | 8 | 1 |
| M1 Baseline | 10 | 6 |
| M2 Scaling | 15 | 25 |
| M3 Ablations | 12 | 20 |
| M4 Inference | 20 | 6 |
| M5 Evaluation | 12 | 4 |
| M6 Interpretability | 20 | 8 |
| M7 Write-up | 10 | 0 |
| **Total** | **~107** | **~70** |

GPU hours fit in **~3 weeks of Kaggle quota**. Build hours are the real constraint: at 10 hrs/week this is roughly 5–6 months; at 20 hrs/week, about 3 months. Reading from the roadmap runs alongside, ideally just ahead of each milestone.

---

## What to deliberately skip

- **Anything requiring multi-node.** Read ZeRO and Megatron; don't implement them.
- **RLHF/GRPO training.** You didn't pick post-training, and it's the worst fit for free-tier compute. Read DeepSeek-R1, skip the implementation.
- **Multimodality.** Different project.
- **Training your own tokenizer** before M3. Use GPT-2's until you have a specific question.
- **Chasing capability numbers.** Your model will lose to a 2019 model on every benchmark. That is not the axis you're competing on.

---

## Stretch goals, in priority order

1. **Hybrid architecture ablation.** Add a Gated DeltaNet or Mamba-2 layer type and sweep the linear:attention layer ratio at fixed FLOPs. This puts you directly on the 2026 frontier question, and at 30M params it's affordable.
2. **Inference-aware scaling law.** Extend M2's fit with the Sardana & Frankle objective: given an expected serving volume, what's the optimal (N, D)? Ties Tracks A and B together and is a genuinely under-explored empirical area.
3. **FP8 simulation.** T4 has no FP8 hardware, but you can simulate the numerics and reproduce a slice of the precision scaling law.
4. **SAE on a hybrid model.** Nobody knows whether dictionary learning behaves the same on linear-attention layers. Small, open, and cheap to investigate.

Stretch goal 2 is the one most likely to produce something novel enough to write up as an actual preprint.
