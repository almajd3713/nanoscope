# Landscape: what exists, what nanoscope can add

Companion to [`plan-tool.md`](plan-tool.md). Surveyed 2026-10-05.

**Method and confidence.**

- Entries with a [source] link were checked with a web search on 2026-10-05.
- Entries marked *(from memory)* rest on general knowledge and were not re-checked. Treat
  their details (feature lists, current status) as unverified.
- Claims about what a tool *lacks* are judgements about its default workflow, not proof
  that no plugin or report adds the feature.

## 1. Training and education projects for small LLMs

| Project | What it does well | What learners use it for | Gap nanoscope can fill | Borrow / avoid |
|---|---|---|---|---|
| **nanoGPT / minGPT** (Karpathy) *(from memory)* | Tiny, readable GPT-2 training and model code; the canonical "read the whole thing" codebase | Reading, then forking to try a change | No comparison or statistics: a fork plus one run is the usual "result". No baselines to measure against. | Borrow: one-file readability (our `gpt2.py`/`modern.py` already follow it). Avoid: "fork to experiment", which loses provenance. |
| **nanochat** [1] | The full pipeline (tokenizer → pretrain → midtrain → SFT → RL → eval → web UI) in about 8K lines of code; one `speedrun.sh` on 8xH100 for about 4 h / $100; a wall-clock leaderboard for reaching GPT-2 grade | Seeing the *whole* LLM lifecycle; following a cost-to-capability story | Assumes an 8xH100 node, so no CPU on-ramp. Measures capability per dollar, not whether a component matters. | Borrow: one script that tells the whole story, and a leaderboard as motivation. Avoid: assuming datacenter compute for learning. |
| **modded-nanogpt speedrun** [2][3] | A community optimization race with a *statistical gate*: a new record needs enough runs for p<0.01 that mean val loss ≤ 3.28 [2]. The optimization track uses a one-sided z-test with an assumed σ [2] | Discovering and validating training tricks (Muon, architecture tweaks) | Its statistics live in PR text and scripts, not in a reusable library, and only at H100 scale | **Borrow the gate**: the evidence rule behind ideas 5 and 9 in the plan. Their framing, "records sit near the seed-noise floor" [3], is exactly our pitch. |
| **llm.c** *(from memory)* | GPT-2 training in raw C/CUDA; a systems education | Learning the kernel level | Out of scope for us | Link to it from the Efficiency path. |
| **LitGPT / torchtitan** *(from memory)* | Production-grade recipes, many architectures, scaling up | Fine-tuning and pretraining at real scale | Heavy for learners; no ablation statistics | Avoid competing; point users there for scale. |
| **tinygrad** *(from memory)* | A minimal framework, teaching autograd and the compiler | Learning frameworks, not LLM design | Different layer of the stack | None. |
| **Karpathy course material** (Zero to Hero videos; LLM101n) [4] | The most-followed path from bigram to GPT. LLM101n was announced as 20 modules but its repo was archived in Aug 2024 while Eureka Labs builds the course [4] | Video plus code-along | Videos can't check your work and don't use seeds or CIs | Borrow the *sequence* (bigram → MLP → attention → GPT), which our Foundations path follows. |
| **Smol Training Playbook** (Hugging Face) [5][6] | A candid account of real ablation practice: change one variable at a time, run many small ablations because intuition is wrong, cloze-style evals early. Ablations and debugging cost more than half of SmolLM3's final run [6] | Learning how labs decide architecture and data | It is prose about 1B-3B ablations on H100 nodes. No tool lets a learner *do* a small version with statistics. | Borrow its rules as lesson content (Honest ablations path) and as study defaults ("one change per variant" is a natural Study lint). |
| **ARENA** [7] | Exercise sets with tests, from transformer-from-scratch through mechanistic interpretability (induction heads, SAEs, superposition) [7] | Structured, test-checked self-study, mostly on pretrained models | Checks cover *implementation*; nothing checks a *training claim*. Uses fixed pretrained models, not your own training run. | Borrow the exercise-with-test format for `equivalent` checks. Go further with `trains`/`verdict`/`predicted` checks. |

## 2. Experiment tracking and research tooling

| Tool | What it does well | Where it falls short for small ablations | Borrow / avoid |
|---|---|---|---|
| **W&B** [8] | Live curves, run grouping showing mean with a std band [8], reports, sweeps | A grouped band is descriptive. There are no paired seed-level CIs, no "within noise" verdict, no refusal when eval text or tokenizers differ, no parameter/FLOP matching. Users ask on its forums how to get the variance of grouped metrics [8]. | Keep the existing `wandb_hook` and export finished runs. Don't rebuild its dashboards. |
| **TensorBoard / Aim / MLflow** *(from memory)* | Local or open-source tracking; MLflow adds a model registry; Aim has fast run comparison | Same as W&B: descriptive, with no experimental-design layer | Export (phase 13) rather than compete. Their "local server reading a log dir" model matches ours. |
| **Hydra** *(from memory)* | Composable configs, multirun sweeps | Configs, not conclusions. Sweeps don't pair seeds or match budgets. | Avoid YAML config trees (the old stack had them and the redesign removed them). TOML `StudySpec` is the narrower equivalent. |
| **lm-evaluation-harness** *(from memory)* | The standard downstream eval suite | At 1-10M parameters most tasks are at chance. Per-task noise dominates (see section 5). | Integrate later as an *Evaluation path* lesson about noise, not as a default metric. |

**The common gap:** these tools record *what happened*. None of them decides *whether a
difference is real* by default, or prevents an unfair comparison. nanoscope's
`compare()` and `Study` already do both; the GUI just needs to put that first.

## 3. Visual builders and model visualizers

| Tool | What it does well | Lesson for nanoscope |
|---|---|---|
| **Netron** [9] | A viewer for 40+ formats: the computational graph, shapes, attributes. PyTorch models usually have to be exported to ONNX first [9] | A static op-level graph is too low-level for learners and is detached from training. Our graph is at block level, generated from source, and annotated with live state. Borrow: collapse and expand subgraphs, and shapes on edges. |
| **Transformer Explainer** (Georgia Tech Polo Club) [10] | Live GPT-2 small in the browser; smooth transitions across levels of abstraction from overview to math; no install [10] | This is the best model for our "depth dial". Borrow the abstraction transitions. The gap: it explains *one fixed pretrained model*. We show *the learner's own model, across training time*. |
| **bbycroft LLM visualization** [11] | 3D animated walkthrough of a tiny 85K-parameter GPT sorting letters, with GPT-2 and GPT-3 scale comparisons [11] | Excellent for intuition about computation. Not interactive for architecture changes. Borrow the guided walkthrough tied to highlighted components. Avoid 3D; it is expensive and rarely needed for block-level reasoning. |
| **TensorSpace / TensorBoard graph** *(from memory)* | 3D layer views; op graphs | Mostly abandoned or low-level. Confirms "block-level, 2D, auto-layout" as the right choice. |
| **ComfyUI-style node editors** [12] | Graphs of typed nodes saved as JSON, custom nodes as Python classes, an HTTP API that queues a workflow [12]; advanced users generate the JSON from Python to get git diffs [12] | Two lessons. (1) When the graph JSON is the artifact, users end up scripting around it; we make **code** the artifact from the start. (2) Workflows silently depend on custom node packages and their dependencies [12]; we record `model.ref` + `source_sha256` and the certification state. |
| **Keras functional API / ONNX tooling** *(from memory)* | Graph and code are the same object in Keras | Supports code-first composition with parse-based visualization, rather than a visual-first builder. |

**Takeaway:** every visual builder that matters either has code as its source, or grows a
way to generate its artifact from code. Free-form wiring is mostly unnecessary for
decoder-only LMs, which are stacks of identical or patterned blocks. Graph editing should
be palette swaps and layer patterns over a parseable code subset, as `plan-tool.md` 5.2
proposes.

## 4. Interactive learning formats and curricula

| Format | Strengths | Weakness for us | Use |
|---|---|---|---|
| **Distill-style explorables** *(from memory)* | Deep intuition through direct manipulation | Expensive to author; fixed models | One or two per path, e.g. attention patterns over training. Not the default format. |
| **Colab / Kaggle notebooks** *(from memory)* | Free GPUs, where learners already are | Hidden state, no checks, sessions die (we already handle resume and Hub sync) | Keep as the "no install" path. Lessons reference the existing notebooks. |
| **Jupyter widgets** *(from memory)* | Interactivity inside notebooks | Fragile across Colab, Kaggle and JupyterLab versions | Avoid relying on them. The GUI does the interactivity. |
| **marimo** [13] | Reactive notebooks stored as pure `.py` files: git-friendly, run as scripts, testable with pytest [13] | Not native on Kaggle or Colab *(unverified)* | Strong candidate for the in-stack notebook and possibly the lesson format; open question 7 in the plan. |
| **ARENA / course platforms** [7] | Sequenced exercises with tests | No compute-aware or statistical checks | `lesson.toml` + checks, as in the plan's section 6. |

## 5. Methodology worth building in

| Work | Finding | Built into nanoscope as |
|---|---|---|
| **Dodge et al. 2020**, "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping" [14] | Changing only the seed (init and data order) gives large performance variance; init and data order contribute about equally [14] | Already present: the seed sets both init (`run.py:324`) and data order. Teach it as Honest ablations lesson 1. Possible later feature: separate `init_seed`/`data_seed` to decompose the variance. |
| **"We need to talk about random seeds"** (2022) [15] | Seed handling in NLP papers is often careless | Lesson content. The `predicted` check makes careless seed handling visible. |
| **modded-nanogpt record rule** [2] | Significance gate at p<0.01 over many runs | Ideas 5 and 9: a gate for ablation cards and the CPU speedrun. |
| **AI2 Signal and Noise** (NeurIPS 2025) [16][17] | Benchmarks with a higher signal-to-noise ratio make small-scale decisions more reliable and scaling-law predictions more accurate. Switching to perplexity-like metrics improves both [16] | Supports our choice of bits per byte on fixed text. Idea 8 (show the noise floor per preset and metric). |
| **AI2 DataDecide** (ICML 2025) [18] | Ranking at one small scale predicted the 1B winner about 80% of the time, and none of 8 scaling-law methods beat it; continuous proxy metrics help [18] | A sober note for the Scaling-lite path: small-scale *rankings* are useful but not certain. Teach that, and report the decision with its uncertainty. |
| **Chinchilla / Kaplan-style FLOP accounting** *(from memory)* | Comparisons need matched compute | Already present: `Tokens`/`FLOPs` budgets, non-embedding params, `match="params"`. |
| **Preregistration** (social-science practice) *(from memory)* | Reduces outcome-driven analysis | Already present as record mode. Extended into a learning mechanic (predict, then run). |

**Not yet in nanoscope, worth adding later** *(from memory, standard statistics)*:

- **Precision planning.** Expected CI half-width for n seeds, roughly
  t(0.975, n-1) · s/√n with s taken from the shipped baselines' seed spread.
- **A guard against "run more seeds until significant".** Either a fixed n declared in the
  spec (record mode), or a properly sequential test. Never silent peeking.
- **Multiple-comparison note** when a study has many variants against one baseline.

## 6. Ideas that follow from nanoscope's specifics

Ranked by value to users and by how well they fit what already exists. Bold = differentiated.

1. **Verdict-first results with precision planning.** Seed-level paired CIs and the
   "within noise" verdict are the first thing shown, along with a pre-launch estimate of
   how many seeds are needed. *Differentiated*: trackers show bands [8]; the speedrun has a
   gate but no tool [2].
2. **Predict, then run.** Generalize `Study.predict` to every run and lesson, and score
   predictions against CIs. *Differentiated as teaching*. It builds calibration, which is
   the skill the Smol playbook says intuition lacks [6].
3. **Blocks with equivalence certificates.** Extends `tests/test_models.py` to user
   blocks; a graph node shows that it matches a naive reference. *Differentiated*: no
   builder in section 3 checks numerical correctness.
4. **Lessons with implement / train / claim checks.** *Differentiated in combination*:
   ARENA checks implementation [7]; nobody checks a training claim.
5. **Ablation cards as shareable data.** A record-mode study exports spec, per-seed
   finals, provenance and commit to a Hub dataset. Because eval text is fixed per preset,
   different people's cards compare directly (Welch, unpaired), with the speedrun's
   significance discipline [2]. *Differentiated*. It needs your decision on hosting
   (plan, open question 9).
6. **Explorables of your own model over training time.** Transformer Explainer's depth
   transitions [10] applied to *your* checkpoints. *Partly differentiated.*
7. **Compute-honest launches** from saved bench history. *Partly; easy.*
8. **Noise-floor display** per preset and metric [16]. *Partly.*
9. **CPU speedrun track** with a significance gate. *Engagement, weakly differentiated.*

**Table stakes** (needed, not a reason to choose nanoscope): live curves, run tables, a
code editor, a model graph viewer, docker-compose, W&B export.

## Sources

1. nanochat: https://github.com/karpathy/nanochat ; MarkTechPost summary: https://www.marktechpost.com/2025/10/14/andrej-karpathy-releases-nanochat-a-minimal-end-to-end-chatgpt-style-pipeline-you-can-train-in-4-hours-for-100/
2. modded-nanogpt (record rules): https://github.com/kellerjordan/modded-nanogpt ; example record PR: https://github.com/KellerJordan/modded-nanogpt/pull/360
3. Reproducing the nanoGPT speedrun: https://frontiercheckpoint.com/reproductions/reproducing-nanogpt-speedrun/
4. LLM101n: https://github.com/karpathy/LLM101n ; https://deepwiki.com/karpathy/LLM101n
5. The Smol Training Playbook: https://huggingfacetb-smol-training-playbook.hf.space/
6. Playbook summary (ablation cost, principles): https://www.theneuron.ai/explainer-articles/hugging-faces-new-playbook-reveals-the-messy-bug-filled-secrets-to-training-world-class-llms/
7. ARENA chapter 1: https://learn.arena.education/chapter1_transformer_interp/ ; curriculum: https://www.arena.education/curriculum
8. W&B grouping threads: https://community.wandb.ai/t/how-to-group-runs-e-g-different-random-seeds-together-on-the-wandb-report-function-for-plots/2634 ; https://community.wandb.ai/t/variance-of-grouped-metric/5461
9. Netron guide: https://briancohn.com/2025/11/12/netron-guide/
10. Transformer Explainer: https://poloclub.github.io/transformer-explainer/ ; paper: https://arxiv.org/html/2408.04619v1
11. LLM Visualization: https://bbycroft.net/llm ; https://github.com/bbycroft/llm-viz
12. ComfyUI custom nodes: https://docs.comfy.org/custom-nodes/walkthrough ; workflow reuse: https://eastondev.com/blog/en/posts/ai/20260602-comfyui-workflow-reuse-guide/
13. marimo: https://github.com/marimo-team/marimo ; https://docs.marimo.io/
14. Dodge et al. 2020: https://arxiv.org/pdf/2002.06305
15. We need to talk about random seeds: https://arxiv.org/pdf/2210.13393
16. Signal and Noise (AI2 blog): https://allenai.org/blog/signal-noise ; paper: https://arxiv.org/pdf/2508.13144
17. NeurIPS 2025 poster: https://neurips.cc/virtual/2025/poster/115712
18. DataDecide: https://arxiv.org/abs/2504.11393
