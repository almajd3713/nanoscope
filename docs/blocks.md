# Blocks

`nanoscope.blocks` is a library of short modules, one idea each, that models are composed
from. `GPT2` and `Modern` are themselves compositions of these blocks, so everything you read
here is what the shipped models run.

## Compose a model

```python
from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, RoPE, SwiGLU


class MyLM(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256):
        super().__init__(
            vocab_size, context_length, d_model=128, n_layers=4,
            block=Block(norm=RMSNorm(), mlp=SwiGLU(hidden=344),
                        attn=Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True)),
            final_norm=RMSNorm(),
            tie_weights=True, z_loss=1e-4,
        )
```

`run(MyLM)` trains it like any model. Calling a block with only its options (`SwiGLU(hidden=344)`)
gives a *spec*, a description with no weights. `Decoder` builds the spec once per layer, so
layers never share parameters. A spec knows its arguments, so `spec.to_dict()` is what the graph
and `describe` show.

Useful `Decoder` options:

| Option | Meaning |
|---|---|
| `block` | One block spec used for every layer. |
| `pattern=[a, b]` | Instead of `block`: repeat a list of specs through the layers (`a, b, a, b, ...`), e.g. sliding-window and global attention. |
| `final_norm` | A norm after the last layer. |
| `pos_emb` | A position embedding added after the token embedding (`LearnedPosition()`); leave it out when attention carries positions (`RoPE`). |
| `tie_weights` | Share the output matrix with the token embedding. |
| `z_loss` | Add a penalty that keeps the softmax normaliser near 1. The model then returns `(logits, aux_loss)`. |

Parameters start as N(0, 0.02), biases at zero, and each layer's output projections at
`0.02 / sqrt(2 * n_layers)`.

## The representable subset

The architecture graph and `nanoscope graph` read your file with `ast`, without importing or
running it. A class is shown as a graph when it is a `Decoder` (or `Composite`) subclass whose
`__init__` is one `super().__init__(...)` call. Each argument is one of:

- a literal (`128`, `True`, `1e-4`),
- one of the class's own `__init__` parameters (`vocab_size`),
- a call to a registered block, with its options by keyword,
- a list of those (`pattern=[...]`).

A call to anything else becomes an *opaque* node: it is drawn, and `describe` reports its
shapes, but its inside is not editable. Any other expression is kept as text. Any other
*statement* in `__init__` (an assignment, a loop, an `if`) makes the class **code-only**: it
still appears, with the reason and the line, and you edit it as source.

```
$ nanoscope graph my_model.py:MyLM
MyLM(Decoder)  my_model.py:5
  vocab_size = <vocab_size>
  context_length = <context_length>
  d_model = 128
  block = Block
    norm = RMSNorm
    ...
```

Edits go back into the file with `blocks.graph.emit(graph, source)`, which changes only the
arguments that changed (keeping comments and formatting) and adds a missing import when a new
block is used. An unedited round trip is byte-identical.

## Blocks and their references

Every block has a typed constructor, a docstring, `flops_per_token(context_length)` (6 FLOPs per
weight used, plus the two attention matmuls) and a naive reference in `nanoscope/reference/`
that its test compares it with. The references import only `torch` and `math`.

| Family | Blocks | Reference |
|---|---|---|
| embedding | `TokenEmbedding`, `LearnedPosition`, `Head` (tied or untied) | `embed_one_hot`, `add_learned_position`, `tied_head` |
| positional | `RoPE`, `NoPE`, `ALiBi` | `naive_rope` (complex rotation; scores depend only on distance), `naive_alibi_attention` (a per-head penalty on distance, added to the scores) |
| norm | `LayerNorm`, `RMSNorm` | `layer_norm`, `rms_norm` |
| attention | `Attention(n_heads, n_kv_heads, pos, qk_norm, window, bias)`: MHA, GQA, MQA and sliding window | `naive_causal_attention` (loops over heads and positions) |
| mlp | `GELUMLP(hidden, bias)`, `SwiGLU(hidden)`, `MoE(experts, top_k, hidden, aux_loss)` | `gelu`, `swiglu`, `naive_moe` (loops over tokens and experts; the load-balancing loss comes back as `(logits, aux)`) |
| structure | `Block(norm, attn, mlp, order)` (pre or post norm), `Decoder` | causality and an untrained loss near `ln(vocab)` |
| primitive | `Linear`, `Activation`, `CausalMask`, `ScaledDotScores`, `Softmax`, `WeightedSum`, `SplitHeads`, `MergeHeads`, `Residual` | one formula each |

`nanoscope blocks` prints the same table with every option, and `nanoscope blocks --json` prints
it as `blocks.v1`. *Primitives* are the smallest pieces: build your own attention from them to
see how the bigger blocks work. `Attention.attention_weights(x)` shows the softmax the fused
kernel never exposes.

## Templates

A `Composite` is a block made of other blocks in named slots. A lesson template is a
`Composite` with empty slots for you to fill:

```python
from nanoscope.blocks import Composite


class PreNormAttention(Composite):
    SLOTS = ("norm", "attn")

    def forward(self, x):
        return x + self.attn(self.norm(x))


block = PreNormAttention(norm=RMSNorm(), attn=Attention(n_heads=4))
```

Each slot is a spec, built once per instance. A `Composite` has slots and a `forward`, not an
`__init__`.

## Register your own block

```python
from nanoscope.blocks import register_block

def naive_gate(x, weight):
    return x * torch.sigmoid(weight)

@register_block(reference=naive_gate, family="mlp")
class Gate(nn.Module):
    """x times sigmoid of a learned vector."""
    def __init__(self, d_model, context_length, scale=1.0): ...
```

The constructor takes `d_model` and `context_length` first, like every block; the rest are the
options a model file writes as `Gate(scale=2.0)`. `reference` is a plain function that computes
the same thing slowly and obviously. A block without one is listed as uncertified.
`nanoscope blocks --workspace my_folder` finds registered blocks by reading the files.

## Inspect

| Command | Shows |
|---|---|
| `nanoscope graph file.py[:Class] [--json]` | the architecture graph, without running the file |
| `nanoscope describe file.py:Class [--preset p] [--json] [--set k=v ...]` | shapes, parameters, FLOPs per token and memory per module, traced on the `meta` device (it imports your file). A shape error names the module and the line. |
| `nanoscope blocks [--json] [--workspace dir]` | the palette |
| `run(Model, block_stats=True)` | at each eval step, per block: activation size, gradient norm, update-to-weight ratio and attention entropy, appended to `blockstats.jsonl` |
| `nanoscope status --blocks <run ref>` | the latest block statistics of a run |
