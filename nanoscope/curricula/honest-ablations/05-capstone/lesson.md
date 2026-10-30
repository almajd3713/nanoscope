Four lessons, four habits: know the noise, pair your runs, match the sizes, predict first. This
one uses them all on a real question. You will run the ablation behind the modern block (the
one from the "modern block" path) and write down what it does and does not support.

## Surface

The claim being tested: *the modern decoder (RoPE, RMSNorm, SwiGLU, grouped-query attention,
QK-norm, z-loss) beats GPT-2.* Two questions follow, and a careful answer needs both.

1. **How big is the gap**, with an interval, against a fairly sized GPT-2?
2. **Which components carry it?** Remove one at a time and see what happens. At this scale,
   expect that you cannot tell some of them from noise. Saying so, with the interval, is a
   result.

### 1. Predict first

Before you run anything, use lesson 4's habit on this one:

```bash
nanoscope learn predict honest-ablations/05-capstone --verdict better --low <x> --high <y> \
    --note "which component I expect to matter most, and which I expect not to"
```

(The check below does not grade it, but your future self will want the record.)

### 2. Run the study

Save this as `studies/ha05_capstone.py` in your workspace and run it. It builds eight variants:
GPT-2, the full modern model, and the modern model with one component removed each. It sizes
every modern variant to GPT-2's non-embedding parameters with `match_params`, so the only
difference between variants is the component (lesson 3), and all variants share the seeds so
`compare` can pair them (lesson 2).

```python
# studies/ha05_capstone.py
from nanoscope import Study
from nanoscope.models import GPT2, Modern
from nanoscope.sizing import build_on_meta, count_params, match_params

ON_GPU = False  # True: the GPU variant, 5 seeds on tinystories-30min with the models' full size
SIZE = dict(d_model=128, n_layers=4) if ON_GPU else dict(d_model=64, n_layers=2)

study = Study(
    "ha05-capstone", preset="tinystories-30min" if ON_GPU else "tinystories-5min",
    seeds=5 if ON_GPU else 3, match="params", baseline="gpt2",
    **({} if ON_GPU else {"max_steps": 200}),
)
target = count_params(build_on_meta(GPT2, vocab_size=4096, **SIZE))[1]


def modern(**removed):
    """Modern with one component removed, its MLP width chosen to match GPT-2's size."""
    kwargs = match_params(Modern, target, "ffn_hidden", range(96, 640, 2),
                          vocab_size=4096, **SIZE, **removed)
    kwargs.pop("vocab_size")
    return kwargs


study.add("gpt2", GPT2, **SIZE)
study.add("modern", Modern, **modern())
study.add("no-rope", Modern, **modern(rope=False))
study.add("no-qk-norm", Modern, **modern(qk_norm=False))
study.add("no-swiglu", Modern, **modern(swiglu=False))
study.add("no-rmsnorm", Modern, **modern(rmsnorm=False))
study.add("no-zloss", Modern, **modern(z_loss=0.0))
study.add("no-gqa", Modern, **modern(n_kv_heads=4))
```

```bash
nanoscope study studies/ha05_capstone.py        # about 20 minutes on a laptop CPU
nanoscope report studies/ha05_capstone.py       # the table, with intervals
```

The CPU version trains each variant for only 200 steps on three seeds, which is short on
purpose. Read the table with lesson 1 in mind: at this budget the noise floor is a large share
of the effects. If an interval for a removed component includes zero, that component is *not
detected*; it may still matter at a larger scale.

### 3. Write what you found, and check it

Open `starter.py` and fill in:

- `GAP = (low, high)`: the interval for the full model against GPT-2 that you stand behind.
  The measured difference must fall inside it, and it must not be wider than about ten times
  the study's own interval (the rule from lesson 4: a range that covers everything claims
  nothing).
- `NULL_COMPONENT`: the name of a variant whose removal is within noise (`"no-zloss"`, for
  example, if that is what your table shows). If your table shows every removal mattering,
  write `None`. Naming a component that does matter fails, with the numbers.

```bash
nanoscope learn check honest-ablations/05-capstone
```

The check reads your study. It trains nothing. If a variant is missing, or a variant has fewer
seeds than the lesson needs, it says which.

### What honest looks like

- The gap is reported **with its interval**, not as a single number.
- Components you cannot detect are described as *not detected at this scale*, not as useless.
- The sizes are matched, and you can say how (and what the match cannot fix: removing GQA
  gives up the memory saving it exists for, which this size comparison does not measure).
- With eight comparisons against one baseline, one "significant" row in twenty is luck. The
  report notes the multiple-comparison risk; do not hide it.

## Deep

1. **Run the GPU variant** (`ON_GPU = True`; 5 seeds on `tinystories-30min`). Which components
   moved from "not detected" to detected when the noise fell?
2. **Pair on one source of randomness.** Rerun a pair with `data_seed` fixed and `init_seed`
   varying, then the reverse. Which kind of seed noise dominates your comparisons?
3. **Record it.** Add `mode="record"` to the study (a clean git tree and a committed study
   file), and a `study.predict(...)` for the gap. This is the research tier: the same code, with
   the guarantees that make a result citable.
4. **The M1 milestone.** [The project plan](https://github.com/almajd3713/nanoscope/blob/main/docs/dev/project-nanoscope.md#m1--the-modern-baseline-10-hrs) describes this ablation at full scale on
   FineWeb-Edu (several GPU-hours, record mode). Everything you did here is its small version.

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: the study above. It is the ablation table of a model paper in miniature.
- **R**: [OLMo 2](https://arxiv.org/pdf/2501.00656) (2024): ablations of exactly these components, with the sizes and budgets stated.
- **R**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683) (Raffel et al., 2019): the model of a careful ablation study.
- **R**: [Show Your Work](https://arxiv.org/pdf/1909.03004) (Dodge et al., 2019): what to report beside the number.
- **S**: [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/pdf/2405.14782) (Biderman et al., 2024): where the noise in language-model evaluation comes from.
- **S**: [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/pdf/2404.10102) (Besiroglu et al., 2024): a published result that a careful refit overturned.
