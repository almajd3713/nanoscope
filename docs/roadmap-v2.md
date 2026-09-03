# LLM Research Roadmap v2 — Refined for a Pretraining / Efficiency / Interpretability Track

*Revision of the "Transformer to the 2026 Frontier" roadmap. Written for someone who has already trained transformers from scratch in PyTorch, has free-tier GPU access only, and wants depth in architecture, inference efficiency, and evaluation/interpretability.*

---

## 0. The core change

v1 answered: **"What has the field published?"**

v2 answers: **"What do I need to have read, and what do I need to have *built*, to do credible work in these three areas?"**

Three structural changes:

1. **Everything is tiered.** No more flat lists where [Reformer](https://arxiv.org/pdf/2001.04451) sits next to [FlashAttention](https://arxiv.org/pdf/2205.14135).
2. **Every stage has an artifact.** If a stage produces no code, plot, or measurement, it's reading, not roadmap.
3. **Depth is allocated by track.** Post-training, agents, and multimodality drop to literacy level. Architecture, efficiency, and interpretability go deep.

### The tier system

| Tier | Meaning | Time |
|---|---|---|
| **B** | **Build.** Implement it. You do not understand it until it runs. | days |
| **R** | **Read closely.** Follow the derivations, understand the ablations, be able to explain the failure modes. | 2–4 hrs |
| **S** | **Skim.** Abstract, figures, conclusion. Know what it claims and why it mattered. | 20 min |
| **K** | **Know it exists.** One line in your notes. Look it up if it becomes relevant. | 2 min |

Roughly: 25 B items, 60 R items, 120 S items, and an unbounded K tail. That is a realistic year, not a fourteen-month reading sentence.

---

## 1. Diagnosis of v1

**What v1 got right:** the phase structure, the "eleven transitions" summary (keep this verbatim — it's the best paragraph in the document), the cross-cutting reorganization by research problem, and the insistence that 2026 is a *systems* frontier rather than an architectural one.

**What v1 got wrong:**

| Problem | Consequence |
|---|---|
| No prioritization | [Reformer](https://arxiv.org/pdf/2001.04451) and [FlashAttention](https://arxiv.org/pdf/2205.14135) look equally important |
| No implementation anywhere | Produces someone who can cite papers but hasn't profiled a kernel |
| Over-weights 2018–2020 encoder era | ~40 papers of archaeology for a decoder-focused researcher |
| Six major content gaps (§8) | Optimization, numerics, parallelism, tokenization, data science, eval methodology |
| Phase IX is stale and partly unverifiable | Product names age in weeks; the section reads as authoritative anyway |
| Chronology drifts | [Let's Verify](https://arxiv.org/pdf/2305.20050) (2023) filed under 2024; [RMSNorm](https://arxiv.org/pdf/1910.07467) (2019) under 2021 |

**Cut outright (≈40 papers).** These are history-of-science for a decoder/efficiency researcher, not working knowledge:

- 2019 encoder branch: [RoBERTa](https://arxiv.org/pdf/1907.11692), [XLNet](https://arxiv.org/pdf/1906.08237), [ALBERT](https://arxiv.org/pdf/1909.11942), [SpanBERT](https://arxiv.org/pdf/1907.10529), [MASS](https://arxiv.org/pdf/1905.02450), [CTRL](https://arxiv.org/pdf/1909.05858), [ELECTRA](https://arxiv.org/pdf/2003.10555), [BART](https://arxiv.org/pdf/1910.13461), [MetaICL](https://arxiv.org/pdf/2110.15943), [CrossFit](https://arxiv.org/pdf/2104.08835), [ExT5](https://arxiv.org/pdf/2111.10952), [Universal Transformers](https://arxiv.org/pdf/1807.03819), [Compressive Transformer](https://arxiv.org/pdf/1911.05507), [Adaptive Attention Span](https://arxiv.org/pdf/1905.07799), [Synthesizer](https://arxiv.org/pdf/2005.00743), [Routing Transformer](https://arxiv.org/pdf/2003.05997), [Reformer](https://arxiv.org/pdf/2001.04451), [Linformer](https://arxiv.org/pdf/2006.04768), [Performer](https://arxiv.org/pdf/2009.14794).
  - *Keep one sentence:* [BERT](https://arxiv.org/pdf/1810.04805) established bidirectional masked pretraining; [ELECTRA](https://arxiv.org/pdf/2003.10555) showed discriminative objectives are more sample-efficient; [RoBERTa](https://arxiv.org/pdf/1907.11692) showed recipe beats architecture. That's the whole lesson.
- Most of the 2023 synthetic-instruction list (Alpaca, [Baize](https://arxiv.org/pdf/2304.01196), Vicuna, [UltraChat](https://arxiv.org/pdf/2305.14233), [Evol-Instruct](https://arxiv.org/pdf/2304.12244), [Orca 2](https://arxiv.org/pdf/2311.11045)) — collapse to [Self-Instruct](https://arxiv.org/pdf/2212.10560) + [LIMA](https://arxiv.org/pdf/2305.11206).
- The DPO variant zoo ([IPO](https://arxiv.org/pdf/2310.12036)/[KTO](https://arxiv.org/pdf/2402.01306)/[ORPO](https://arxiv.org/pdf/2403.07691)/[SimPO](https://arxiv.org/pdf/2405.14734)/[CPO](https://arxiv.org/pdf/2401.08417)/[SLiC](https://arxiv.org/pdf/2305.10425)/[RRHF](https://arxiv.org/pdf/2304.05302)/[RSO](https://arxiv.org/pdf/2309.06657)) — collapse to [DPO](https://arxiv.org/pdf/2305.18290) + one survey.
- Most agent benchmarks — keep [SWE-bench Verified](https://arxiv.org/pdf/2310.06770), [τ-bench](https://arxiv.org/pdf/2406.12045), [OSWorld](https://arxiv.org/pdf/2404.07972); the rest is K-tier.

**Demote to a single paragraph:** the 2020 efficient-attention zoo. v1 already admits none of them replaced dense attention. One paragraph explaining *why* (approximation quality vs. hardware efficiency; the winners optimized memory movement, not asymptotic complexity) is worth more than seven papers.

---

## 2. The Spine

If you read nothing else, read these, in this order. Roughly 40 items. This is the causal skeleton.

### Stage 1 — Foundation (B/R)
1. [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762) (2017) — **B**. Implement from scratch, no reference. Again.
2. [**GPT-2**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) — **R**. The decoder-only + zero-shot-as-text framing.
3. [**BERT**](https://arxiv.org/pdf/1810.04805) (2018) — **S**. Know the branch exists and why it lost for generation.
4. [**T5**](https://arxiv.org/pdf/1910.10683) (2019) — **R**. Read for the *ablation methodology*, not the model. This is how you run a systematic architecture study.
5. **The Illustrated / Annotated Transformer**, and **nanoGPT** (Karpathy) — **B**. Read the code line by line.

### Stage 2 — Scaling and the optimization substrate (B/R)
6. [**Scaling Laws for Neural Language Models**](https://arxiv.org/pdf/2001.08361) (Kaplan, 2020) — **R**
7. [**Training Compute-Optimal LLMs**](https://arxiv.org/pdf/2203.15556) (Chinchilla, Hoffmann, 2022) — **B**. Reproduce the isoFLOP analysis at toy scale.
8. [**Chinchilla Scaling: A Replication Attempt**](https://arxiv.org/pdf/2404.10102) (Besiroglu et al., 2024) — **R**. *New.* The original fit had problems. Read this to learn that scaling laws are curve fits with error bars, not laws of nature.
9. [**Beyond Chinchilla-Optimal: Accounting for Inference**](https://arxiv.org/pdf/2401.00448) (Sardana & Frankle, 2023) — **R**. *New.* Once you serve a model, compute-optimal is the wrong objective. Directly bridges your architecture and efficiency tracks.
10. [**Tensor Programs V / muP**](https://arxiv.org/pdf/2203.03466) (Yang & Hu, 2022) — **B**. *New, and the biggest single addition to v1.* Hyperparameter transfer across width. Without this, small-scale ablations are noise generators.
11. [**Scaling Laws for Precision**](https://arxiv.org/pdf/2411.04330) (Kumar et al., 2024) — **R**. *New.* Unifies quantization and scaling. Beautiful paper.
12. [**An Empirical Model of Large-Batch Training**](https://arxiv.org/pdf/1812.06162) (McCandlish et al., 2018) — **R**. *New.* Critical batch size; why you can't just crank batch size.
13. [**GPT-3**](https://arxiv.org/pdf/2005.14165) (2020) — **S**. In-context learning as the headline; the rest is now assumed.

### Stage 3 — The modern block (B)
14. [**RoFormer / RoPE**](https://arxiv.org/pdf/2104.09864) (Su, 2021) — **B**
15. [**RMSNorm**](https://arxiv.org/pdf/1910.07467) (Zhang & Sennrich, 2019) — **B**
16. [**GLU Variants / SwiGLU**](https://arxiv.org/pdf/2002.05202) (Shazeer, 2020) — **B**
17. [**Multi-Query Attention**](https://arxiv.org/pdf/1911.02150) (Shazeer, 2019) and [**GQA**](https://arxiv.org/pdf/2305.13245) (Ainslie, 2023) — **B**
18. [**DeepSeek-V2 — Multi-head Latent Attention**](https://arxiv.org/pdf/2405.04434) (2024) — **B**. *New. Absent from v1 entirely.* Low-rank KV compression; the most important cache innovation since GQA, and now standard (Kimi K2.5, GLM-5, Ling 2.5).
19. [**LLaMA**](https://arxiv.org/pdf/2302.13971) (2023) + [**Mistral 7B**](https://arxiv.org/pdf/2310.06825) (2023) — **R**. The convergent recipe.
20. [**OLMo 2**](https://arxiv.org/pdf/2501.00656) (2024) — **R**. Read for stability engineering: QK-norm, z-loss, why runs diverge.
21. [**DeepSeek-V3 Technical Report**](https://arxiv.org/pdf/2412.19437) (2024) — **R**. The most complete public frontier recipe: MLA, DeepSeekMoE, aux-loss-free load balancing, FP8 training, MTP.

### Stage 4 — Systems and inference (B)
22. [**FlashAttention**](https://arxiv.org/pdf/2205.14135) (Dao, 2022) — **R**, and [**FlashAttention-2**](https://arxiv.org/pdf/2307.08691) — **S**. Read for the IO-aware framing.
23. **A roofline / arithmetic-intensity primer** — **B**. *New.* Kipply's *Transformer Inference Arithmetic*, or the HF *Ultra-Scale Playbook*. The prefill-is-compute-bound / decode-is-memory-bound distinction is the single most useful mental model in inference, and v1 never states it.
24. [**PagedAttention / vLLM**](https://arxiv.org/pdf/2309.06180) (2023) — **R**
25. [**Speculative Decoding**](https://arxiv.org/pdf/2211.17192) (Leviathan, 2023) — **B**
26. [**EAGLE-2**](https://arxiv.org/pdf/2406.16858) / [**EAGLE-3**](https://arxiv.org/pdf/2503.01840) (2024–25) — **S**. *New.* Where speculative decoding actually went.
27. [**GPTQ**](https://arxiv.org/pdf/2210.17323) (2022) and [**AWQ**](https://arxiv.org/pdf/2306.00978) (2023) — **B**. Implement one weight-only quantizer yourself.
28. [**LLM.int8() / outlier features**](https://arxiv.org/pdf/2208.07339) (Dettmers, 2022) — **R**. Why quantization breaks at scale.
29. [**Mixed Precision Training**](https://arxiv.org/pdf/1710.03740) (Micikevicius, 2017) — **R**. *New.* Plus a bf16-vs-fp16 and FP8 writeup.
30. [**ZeRO**](https://arxiv.org/pdf/1910.02054) (Rajbhandari, 2019) + [**Megatron-LM**](https://arxiv.org/pdf/1909.08053) (2019) — **R**. *New (ZeRO).* Even if you never run multi-GPU, the memory accounting is essential.

### Stage 5 — Beyond dense attention (R)
31. [**Switch Transformer**](https://arxiv.org/pdf/2101.03961) (2021) — **R**, [**DeepSeekMoE**](https://arxiv.org/pdf/2401.06066) (2024) — **R**. *New (DeepSeekMoE):* shared + fine-grained experts, the design that actually won.
32. [**Mamba**](https://arxiv.org/pdf/2312.00752) (2023) — **R**, [**Mamba-2 / State Space Duality**](https://arxiv.org/pdf/2405.21060) (2024) — **R**
33. [**Gated DeltaNet**](https://arxiv.org/pdf/2412.06464) (2024) — **R**. *New.* The linear-attention block that displaced Mamba in most 2026 hybrids.
34. [**Native Sparse Attention**](https://arxiv.org/pdf/2502.11089) (DeepSeek, 2025) / [**DeepSeek Sparse Attention**](https://arxiv.org/pdf/2502.11089) — **R**. *New.* Trainable sparse attention; absent from v1.
35. **A 2026 hybrid technical report** — **R**. *New.* [Kimi Linear](https://arxiv.org/pdf/2510.26692), Qwen3-Next/3.5, or Nemotron 3. Pick one and read the layer-interleaving ratios carefully.

### Stage 6 — Evaluation and interpretability (B)
36. [**Adding Error Bars to Evals**](https://arxiv.org/pdf/2411.00640) (Miller, 2024) — **B**. *New.* Bootstrap CIs, paired tests, clustered standard errors. Do this before you trust any ablation you run.
37. [**Are Emergent Abilities a Mirage?**](https://arxiv.org/pdf/2304.15004) (Schaeffer, 2023) — **R**
38. [**A Mathematical Framework for Transformer Circuits**](https://transformer-circuits.pub/2021/framework/index.html) (2021) — **R**, and [**In-context Learning and Induction Heads**](https://arxiv.org/pdf/2209.11895) (2022) — **B**. Reproduce the induction bump on your own model.
39. [**Toy Models of Superposition**](https://arxiv.org/pdf/2209.10652) (2022) — **B**. Reproduce it. It's genuinely small.
40. [**Towards Monosemanticity**](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (2023) → [**Scaling Monosemanticity**](https://arxiv.org/pdf/2605.29358) (2024) — **R**; [**Circuit Tracing / Attribution Graphs**](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Ameisen & Lindsey, 2025) — **R**. *New.* Cross-layer transcoders; the current frontier method.
41. **Physics of Language Models** series ([Part 1](https://arxiv.org/pdf/2305.13673), [Part 3.1](https://arxiv.org/pdf/2309.14316), [Part 3.2](https://arxiv.org/pdf/2309.14402), [Part 3.3](https://arxiv.org/pdf/2404.05405); Allen-Zhu & Li, 2023–25) — **R**. *New, and badly missed by v1.* Controlled synthetic experiments on knowledge storage, extraction, and reasoning. The most directly imitable research program for someone on one GPU.

### Stage 7 — Literacy on everything else (S)
42. [InstructGPT](https://arxiv.org/pdf/2203.02155) — **R** (it's foundational enough to earn R even off-track)
43. [Chain-of-Thought](https://arxiv.org/pdf/2201.11903) + [Self-Consistency](https://arxiv.org/pdf/2203.11171) — **S**
44. [DPO](https://arxiv.org/pdf/2305.18290) — **R**
45. [Let's Verify Step by Step](https://arxiv.org/pdf/2305.20050) — **S**
46. [DeepSeek-R1](https://arxiv.org/pdf/2501.12948) — **R**. Read for GRPO and the emergence-of-reasoning claim.
47. [RAG](https://arxiv.org/pdf/2005.11401) + one 2025 long-context survey — **S**
48. [ReAct](https://arxiv.org/pdf/2210.03629) + [SWE-bench Verified](https://arxiv.org/pdf/2310.06770) — **S**

---

## 3. Track A — Pretraining and Architecture (deep)

Beyond the spine.

**Positional information.** [Sinusoidal](https://arxiv.org/pdf/1706.03762) → learned → [T5 relative bias](https://arxiv.org/pdf/1910.10683) → [**RoPE**](https://arxiv.org/pdf/2104.09864) → [ALiBi](https://arxiv.org/pdf/2108.12409) → [position interpolation](https://arxiv.org/pdf/2306.15595) → NTK-aware scaling → [**YaRN**](https://arxiv.org/pdf/2309.00071) → [LongRoPE](https://arxiv.org/pdf/2402.13753) → [**NoPE**](https://arxiv.org/pdf/2203.16634) (no positional encoding at all — surprisingly competitive) → [**p-RoPE**](https://arxiv.org/pdf/2603.11611) (partial RoPE for very long context, 2026). Key question to hold: what does a positional scheme *have* to provide, given that attention is permutation-equivariant only up to it?

**Normalization and stability.** [Pre-LN vs Post-LN](https://arxiv.org/pdf/2002.04745) → [DeepNorm](https://arxiv.org/pdf/2203.00555) → **QK-norm** → [z-loss](https://arxiv.org/pdf/2202.08906) → sandwich/peri-LN → [**nGPT**](https://arxiv.org/pdf/2410.01131) (normalized transformer on the hypersphere). Plus: [**The Super Weight in LLMs**](https://arxiv.org/pdf/2411.07191) and [**Massive Activations in LLMs**](https://arxiv.org/pdf/2402.17762) — a handful of outlier weights/activations dominate, which is why naive quantization fails. Bridges into Track B.

**Attention variants.** [MHA](https://arxiv.org/pdf/1706.03762) → [MQA](https://arxiv.org/pdf/1911.02150) → [GQA](https://arxiv.org/pdf/2305.13245) → [**MLA**](https://arxiv.org/pdf/2405.04434) → [**DiffTransformer**](https://arxiv.org/pdf/2410.05258) → [**Gated Attention**](https://arxiv.org/pdf/2505.06708) (Qwen3-Next) → sliding-window/global interleaving ([Gemma 2](https://arxiv.org/pdf/2408.00118)/[3](https://arxiv.org/pdf/2503.19786), ratios matter) → [**Native Sparse Attention**](https://arxiv.org/pdf/2502.11089) → [attention sinks](https://arxiv.org/pdf/2309.17453).

**Linear-time and hybrid.** [S4](https://arxiv.org/pdf/2111.00396) → [Mamba](https://arxiv.org/pdf/2312.00752) → [Mamba-2](https://arxiv.org/pdf/2405.21060) → [**Mamba-3**](https://arxiv.org/pdf/2603.15569) → [DeltaNet](https://arxiv.org/pdf/2406.06484) → [**Gated DeltaNet**](https://arxiv.org/pdf/2412.06464) → [**Gated DeltaNet-2**](https://arxiv.org/pdf/2605.22791) → [RWKV-7](https://arxiv.org/pdf/2503.14456) → [Kimi Linear](https://arxiv.org/pdf/2510.26692). The 2026 consensus: pure linear models lose on precise recall; hybrids at roughly 3:1 to 7:1 linear:attention layer ratios match full attention at far lower long-context cost. Understand *why* attention is uniquely good at content-addressable lookup.

**MoE.** [GShard](https://arxiv.org/pdf/2006.16668) → [Switch](https://arxiv.org/pdf/2101.03961) → [ST-MoE](https://arxiv.org/pdf/2202.08906) → [**DeepSeekMoE**](https://arxiv.org/pdf/2401.06066) (shared + fine-grained experts) → [aux-loss-free load balancing](https://arxiv.org/pdf/2412.19437) → [**MoE upcycling**](https://arxiv.org/pdf/2212.05055) (convert a trained dense model's FFNs into experts — the only MoE approach you can actually afford). Also: expert specialization analyses, router entropy collapse.

**Optimization** *(entirely new section — v1 had none)*.
- [AdamW](https://arxiv.org/pdf/1711.05101), warmup, cosine vs [**WSD (warmup-stable-decay)**](https://arxiv.org/pdf/2404.06395) schedules — WSD lets you branch checkpoints, which matters enormously for a scaling study on limited compute.
- [**muP**](https://arxiv.org/pdf/2203.03466) and [Depth-μP](https://arxiv.org/pdf/2310.02244).
- [**Muon**](https://kellerjordan.github.io/posts/muon/) (Jordan et al.) — orthogonalized momentum; now used in frontier runs. [**SOAP**](https://arxiv.org/pdf/2409.11321), [**Shampoo**](https://arxiv.org/pdf/1802.09568).
- Loss spikes, gradient clipping, embedding norm growth, initialization schemes.
- Critical batch size; LR–batch-size scaling.

**Tokenization** *(new)*. [BPE](https://arxiv.org/pdf/1508.07909) (Sennrich 2015) → [SentencePiece](https://arxiv.org/pdf/1808.06226) → byte-level BPE → tokenizer arithmetic pathologies → [**Super-BPE**](https://arxiv.org/pdf/2503.13423) → byte-level models ([**MEGABYTE**](https://arxiv.org/pdf/2305.07185), [**Byte Latent Transformer**](https://arxiv.org/pdf/2412.09871)). Include: tokenizer choice measurably shifts loss curves, which corrupts cross-tokenizer comparisons — a trap you will hit in your own ablations.

**Numerics** *(new)*. [Mixed precision](https://arxiv.org/pdf/1710.03740) → bf16 vs fp16 → [**FP8 training**](https://arxiv.org/pdf/2412.19437) (DeepSeek-V3) → FP4 → stochastic rounding → [**Scaling Laws for Precision**](https://arxiv.org/pdf/2411.04330).

**Data as a research variable** *(expanded)*. [FineWeb / FineWeb-Edu](https://arxiv.org/pdf/2406.17557) (read the *ablation methodology* — it's a model for how to evaluate data cheaply) → [DataComp-LM](https://arxiv.org/pdf/2406.11794) → [Nemotron-CC](https://arxiv.org/pdf/2412.02595) → [deduplication](https://arxiv.org/pdf/2107.06499) (Lee et al. 2021) → [**Scaling Data-Constrained Language Models**](https://arxiv.org/pdf/2305.16264) (Muennighoff, 2023: how many epochs before repetition stops helping — critical when you're token-limited) → data mixing laws ([**DoReMi**](https://arxiv.org/pdf/2305.10429), [**RegMix**](https://arxiv.org/pdf/2407.01492)) → mid-training and curriculum.

---

## 4. Track B — Inference, Serving, Efficiency (deep)

**The mental model first.** Prefill is compute-bound; decode is memory-bandwidth-bound. Almost every inference technique is explained by which of those two it attacks. Batching helps decode because it raises arithmetic intensity. KV-cache compression helps decode because the cache is the bandwidth cost. Speculative decoding helps decode because it converts sequential memory-bound steps into one parallel compute-bound step. Hold this and the literature organizes itself.

**Kernels.** [FlashAttention 1](https://arxiv.org/pdf/2205.14135)/[2](https://arxiv.org/pdf/2307.08691)/[3](https://arxiv.org/pdf/2407.08608) → [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) → fused RMSNorm/RoPE/SwiGLU → [**Triton**](https://arxiv.org/pdf/2104.07093) (write one yourself) → torch.compile and what it does and doesn't fuse → [**Liger-Kernel**](https://arxiv.org/pdf/2410.10989) as a readable reference implementation.

**Serving.** Continuous batching ([Orca](https://www.usenix.org/system/files/osdi22-yu.pdf)) → [PagedAttention/vLLM](https://arxiv.org/pdf/2309.06180) → [**RadixAttention / SGLang**](https://arxiv.org/pdf/2312.07104) (prefix caching) → [**chunked prefill**](https://arxiv.org/pdf/2403.02310) → **disaggregated prefill/decode** ([DistServe](https://arxiv.org/pdf/2401.09670), [Splitwise](https://arxiv.org/pdf/2311.18677)) → TensorRT-LLM. *All new except the first two.*

**KV cache.** [GQA](https://arxiv.org/pdf/2305.13245) → [MLA](https://arxiv.org/pdf/2405.04434) → KV quantization ([**KIVI**](https://arxiv.org/pdf/2402.02750)) → eviction ([H2O](https://arxiv.org/pdf/2306.14048), [SnapKV](https://arxiv.org/pdf/2404.14469), [StreamingLLM/attention sinks](https://arxiv.org/pdf/2309.17453)) → [**DuoAttention**](https://arxiv.org/pdf/2410.10819) (retrieval heads vs streaming heads) → cross-layer sharing ([YOCO](https://arxiv.org/pdf/2405.05254)).

**Quantization.** [LLM.int8()](https://arxiv.org/pdf/2208.07339) → [GPTQ](https://arxiv.org/pdf/2210.17323) → [AWQ](https://arxiv.org/pdf/2306.00978) → [SmoothQuant](https://arxiv.org/pdf/2211.10438) → [**QServe (W4A8KV4)**](https://arxiv.org/pdf/2405.04532) → [AQLM](https://arxiv.org/pdf/2401.06118)/[QuIP#](https://arxiv.org/pdf/2402.04396) (extreme low-bit) → [**QAT**](https://arxiv.org/pdf/2305.17888) → FP8/FP4 inference. Anchor on the outlier-feature explanation from Track A.

**Speculative and parallel decoding.** [Leviathan](https://arxiv.org/pdf/2211.17192)/[Chen](https://arxiv.org/pdf/2302.01318) → [Medusa](https://arxiv.org/pdf/2401.10774) → [**EAGLE-1**](https://arxiv.org/pdf/2401.15077)/[**2**](https://arxiv.org/pdf/2406.16858)/[**3**](https://arxiv.org/pdf/2503.01840) → [Lookahead](https://arxiv.org/pdf/2402.02057) → [**multi-token prediction**](https://arxiv.org/pdf/2412.19437) (DeepSeek-V3 trains for it) → [self-speculation](https://arxiv.org/pdf/2404.16710).

**Distillation and small models.** [Hinton](https://arxiv.org/pdf/1503.02531) → [sequence-level KD](https://arxiv.org/pdf/1606.07947) → [MiniLLM](https://arxiv.org/pdf/2306.08543) → the [SmolLM](https://arxiv.org/pdf/2502.02737)/[Qwen-small](https://arxiv.org/pdf/2412.15115)/[Gemma-small](https://arxiv.org/pdf/2408.00118) reports → [**on-policy distillation**](https://arxiv.org/pdf/2306.13649) (2025). Capability density as an explicit axis.

**Determinism and reproducibility** *(new)*. [**Defeating Nondeterminism in LLM Inference**](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Thinking Machines, 2025) — batch-invariant kernels. Directly relevant to your evals track: nondeterministic inference silently inflates measured variance.

---

## 5. Track C — Evaluation and Interpretability (deep)

### Evaluation as a craft, not a leaderboard

v1 listed benchmarks. Benchmarks saturate; methodology doesn't. Prioritize:

- [**Adding Error Bars to Evals**](https://arxiv.org/pdf/2411.00640) (Miller, 2024) — **B**. The central reference.
- Prompt sensitivity and format variance: [*Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*](https://arxiv.org/pdf/2310.11324); [*State of What Art? A Call for Multi-Prompt Evaluation*](https://aclanthology.org/2024.tacl-1.52.pdf).
- Contamination: detection methods, canary strings, [**LiveBench**](https://arxiv.org/pdf/2406.19314)-style continuous refresh, [*Rethinking Benchmark and Contamination*](https://arxiv.org/pdf/2311.04850).
- LLM-as-judge: [**MT-Bench**](https://arxiv.org/pdf/2306.05685), position/verbosity/self-preference bias, judge calibration, [*Judging LLM-as-a-Judge*](https://arxiv.org/pdf/2306.05685).
- Arena critiques: [**The Leaderboard Illusion**](https://arxiv.org/pdf/2504.20879) (2025) — how leaderboard dynamics distort what gets measured.
- Tooling: **lm-evaluation-harness** (read the source; understand how it does length-normalized loglikelihood scoring and why that choice changes rankings), [HELM](https://arxiv.org/pdf/2211.09110), Inspect.
- Scaling-law-based prediction of downstream performance; observational scaling laws.

The skill to develop: given a claimed 1.5% improvement, determine whether it's real. Paired bootstrap, seed variance, prompt variance, decoding variance.

### Interpretability

**Foundations (B/R).** [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) → [Induction Heads](https://arxiv.org/pdf/2209.11895) → [Toy Models of Superposition](https://arxiv.org/pdf/2209.10652) → [the Linear Representation Hypothesis](https://arxiv.org/pdf/2311.03658) → logit lens → [tuned lens](https://arxiv.org/pdf/2303.08112).

**Causal methods (B).** [Activation patching](https://arxiv.org/pdf/2202.05262) → [path patching](https://arxiv.org/pdf/2211.00593) → [**attribution patching**](https://arxiv.org/pdf/2310.10348) (linear approximation, scales far better and is what you can afford) → [causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) → [ACDC](https://arxiv.org/pdf/2304.14997). Tooling: **TransformerLens**, [**nnsight**](https://arxiv.org/pdf/2407.14561), **SAELens**.

**Dictionary learning (B/R).** [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) → [Scaling Monosemanticity](https://arxiv.org/pdf/2605.29358) → [JumpReLU](https://arxiv.org/pdf/2407.14435) and [TopK SAEs](https://arxiv.org/pdf/2406.04093) → [**Gemma Scope**](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf) / [**Gemma Scope 2**](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf) → [**transcoders**](https://arxiv.org/pdf/2406.11944) and [**cross-layer transcoders**](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) → [**Circuit Tracing / Attribution Graphs**](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (2025, tooling open-sourced) → [*Automatically Interpreting Millions of Features*](https://arxiv.org/pdf/2410.13928).

**Read the negative results too.** SAEs have had a hard 2025–26: feature-consistency problems, disappointing downstream results, identifiability concerns ([*Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?*](https://arxiv.org/pdf/2502.20914)). Anthropic's own circuit tracing on Claude 3.5 Haiku yielded satisfying accounts for only a minority of tested prompts. A roadmap that presents SAEs as solved is misleading you; the open problems are where the work is. Start from [**Open Problems in Mechanistic Interpretability**](https://arxiv.org/pdf/2501.16496) (2025).

**Small-scale-friendly programs.** These are the ones you can actually run on free compute:
- **Physics of Language Models** ([Part 1](https://arxiv.org/pdf/2305.13673), [Part 3.1](https://arxiv.org/pdf/2309.14316), [Part 3.2](https://arxiv.org/pdf/2309.14402), [Part 3.3](https://arxiv.org/pdf/2404.05405); Allen-Zhu) — synthetic data, tiny models, sharp causal claims.
- [**Grokking**](https://arxiv.org/pdf/2201.02177) and [progress measures](https://arxiv.org/pdf/2301.05217) (Nanda) — modular arithmetic, minutes per run.
- [**Toy Models of Superposition**](https://arxiv.org/pdf/2209.10652) — reproducible in a notebook.
- Induction-head emergence across training checkpoints — you can produce this on your own 30M model.

---

## 6. Literacy tracks (compressed)

Keep these at S-tier unless you change specialization.

**Post-training.** [InstructGPT](https://arxiv.org/pdf/2203.02155) → [DPO](https://arxiv.org/pdf/2305.18290) → RLVR/GRPO ([DeepSeek-R1](https://arxiv.org/pdf/2501.12948)) → process reward models ([Let's Verify](https://arxiv.org/pdf/2305.20050)) → reward hacking and verifier gaming. Know the shape of the pipeline and the failure modes; skip the variant zoo.

**Reasoning and test-time compute.** [CoT](https://arxiv.org/pdf/2201.11903) → [self-consistency](https://arxiv.org/pdf/2203.11171) → [STaR](https://arxiv.org/pdf/2203.14465) → best-of-N vs sequential revision → [*s1: Simple Test-Time Scaling*](https://arxiv.org/pdf/2501.19393) → adaptive compute allocation. One good survey beats twenty papers.

**Retrieval.** [RAG](https://arxiv.org/pdf/2005.11401) → [DPR](https://arxiv.org/pdf/2004.04906) → [RETRO](https://arxiv.org/pdf/2112.04426) → long-context vs retrieval tradeoff → [*Lost in the Middle*](https://arxiv.org/pdf/2307.03172) → [RULER](https://arxiv.org/pdf/2404.06654). Know when retrieval beats context and why.

**Agents.** [ReAct](https://arxiv.org/pdf/2210.03629) → [SWE-agent](https://arxiv.org/pdf/2405.15793)/[OpenHands](https://arxiv.org/pdf/2407.16741) → [SWE-bench Verified](https://arxiv.org/pdf/2310.06770) → [τ-bench](https://arxiv.org/pdf/2406.12045) → [OSWorld](https://arxiv.org/pdf/2404.07972) → prompt injection and instruction hierarchy. K/S tier.

**Multimodality.** [CLIP](https://arxiv.org/pdf/2103.00020) → [LLaVA](https://arxiv.org/pdf/2304.08485) → native tokenization ([Chameleon](https://arxiv.org/pdf/2405.09818)/[Transfusion](https://arxiv.org/pdf/2408.11039)) → a recent [Qwen-VL](https://arxiv.org/pdf/2308.12966) or [InternVL](https://arxiv.org/pdf/2312.14238) report. S tier.

---

## 7. The six gaps in v1, summarized

If you only patch six things in the original document, patch these:

1. **Optimization and training dynamics** — [muP](https://arxiv.org/pdf/2203.03466), [Muon](https://arxiv.org/pdf/2502.16982), [WSD](https://arxiv.org/pdf/2404.06395), [critical batch size](https://arxiv.org/pdf/1812.06162), loss spikes.
2. **Numerics and precision** — [mixed precision](https://arxiv.org/pdf/1710.03740) through FP8/FP4, [scaling laws for precision](https://arxiv.org/pdf/2411.04330).
3. **Parallelism and memory** — [ZeRO](https://arxiv.org/pdf/1910.02054)/FSDP, activation recomputation, the memory accounting.
4. **Tokenization** — [BPE](https://arxiv.org/pdf/1508.07909) through byte-level models; the cross-tokenizer comparison trap.
5. **Data as a measured variable** — [data-constrained scaling](https://arxiv.org/pdf/2305.16264), mixing laws, [FineWeb's ablation methodology](https://arxiv.org/pdf/2406.17557).
6. **Evaluation methodology** — error bars, prompt variance, contamination, judge bias.

Plus one missing *paper*: [**MLA (DeepSeek-V2)**](https://arxiv.org/pdf/2405.04434).

---

## 8. Phase IX, corrected

v1 dates itself "as of June 19, 2026" and cites GPT-5.4 and GPT-5.5. That snapshot has already aged: GPT-5.4 arrived March 2026, GPT-5.5 in April, and GPT-5.6 (Sol/Terra/Luna) plus the GPT-Live voice models followed in June–July, alongside a period of government-mediated restricted release for several frontier models. Any version of this section will be wrong within two months.

**The fix: replace named models with durable claims.** Rewrite Phase IX as trends, and treat specific model names as illustrations with explicit dates attached.

Corrections to the substance:

- **Hybrid architectures are no longer speculative; they are the mainstream open-weight design.** v1 lists them under "emerging." As of 2026: Qwen3.5/3.6 use [Gated DeltaNet](https://arxiv.org/pdf/2412.06464) layers, [Kimi Linear](https://arxiv.org/pdf/2510.26692) runs Kimi DeltaNet on ~75% of layers with [MLA](https://arxiv.org/pdf/2405.04434) on the rest, Nemotron 3 alternates [Mamba-2](https://arxiv.org/pdf/2405.21060) on ~85% of layers with [GQA](https://arxiv.org/pdf/2305.13245), Ling 2.5 combines Lightning Attention with MLA. [Mamba-3](https://arxiv.org/pdf/2603.15569) and [Gated DeltaNet-2](https://arxiv.org/pdf/2605.22791) exist. Move this into the architecture track as current practice.
- [**Trainable sparse attention**](https://arxiv.org/pdf/2502.11089) (DeepSeek NSA/DSA, GLM-5's IndexShare) is a distinct line from linear attention and is missing entirely from v1.
- [**MoE upcycling**](https://arxiv.org/pdf/2212.05055) deserves a mention as the practical path for small labs.
- **Interpretability's status changed:** [attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) applied to a production model, tooling open-sourced, [Gemma Scope 2](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf) scaled to 27B, and MIT Tech Review naming mech interp a 2026 breakthrough technology. But so did the skepticism — include the negative results.
- **Add a "confidence" convention.** Mark each Phase IX claim as *documented* (in a paper/system card), *reported* (press/blog), or *inferred*. v1 presents all three with equal confidence, which is the section's real flaw.

---

## 9. Errata in v1

- [*Let's Verify Step by Step*](https://arxiv.org/pdf/2305.20050) is Lightman et al., **2023**, filed under Phase VII (2024).
- [RMSNorm](https://arxiv.org/pdf/1910.07467) is Zhang & Sennrich, **2019**, listed under the 2021 section.
- [*Multi-Query Attention*](https://arxiv.org/pdf/1911.02150) is Shazeer **2019**; [GQA](https://arxiv.org/pdf/2305.13245) (Ainslie) is **2023** — v1 says "later generalized" without the citation.
- [ELECTRA](https://arxiv.org/pdf/2003.10555) is dated "released in early 2020" but listed under 2019.
- [*Improving Language Models by Retrieving from Trillions of Tokens*](https://arxiv.org/pdf/2112.04426) is listed twice in the 2022 retrieval section — it is RETRO, already the first entry.
- [ReAct](https://arxiv.org/pdf/2210.03629) is dated "2022/2023" — arXiv Oct 2022, ICLR 2023.
- [Mamba's](https://arxiv.org/pdf/2312.00752) claim that it "could approach Transformer quality" needs the 2025–26 correction: pure SSMs underperform on precise recall, which is exactly why hybrids won.
- The formula rendering in Phase VII ("+++++high-quality pretraining...") is corrupted.

---

## 10. Suggested allocation

Assuming this runs alongside the project (see companion document), roughly:

- **30%** reading (spine first, then Track A/B/C depth)
- **60%** building (the project)
- **10%** writing up (notes, ablation tables, a public write-up)

The ratio is the point. v1 implied 100/0/0.
