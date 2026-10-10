A block that is in the palette but unproven is a guess. nanoscope can *certify* one: it builds the
block with random weights, feeds it random inputs, and compares the output with a plain function
you wrote that does the same thing slowly and obviously. A block that matches shows a certificate;
change the file and the certificate goes stale until it is checked again.

## Surface

You will write a feed-forward block that squares the ReLU (used in some small fast models):
`relu(x @ w_up) ** 2 @ w_down`.

In `starter.py`:

- `squared_relu_mlp(x, w_up, w_down)`: the reference, written with no module and no tricks.
  The certifier passes the parameters to it **by name**, which is why the names matter.
- `SquaredReLUMLP`, an `nn.Module` taking `d_model` and `context_length` first, then `hidden`.
  Its parameters are `w_up` and `w_down`.
- `@register_block(reference=squared_relu_mlp, family="mlp")` on the class.

```bash
nanoscope learn check extend/01-certified-block
```

The check does two things. It certifies your block against *your* reference, and it also compares
it with the lesson's own. The second is there because a formula written wrong twice would pass
the first.

### Using it

Once it passes, the block can be used like a shipped one in a model file, and certified from the
model page (or `POST /api/blocks/SquaredReLUMLP/certify`). Edit one character of the forward pass
and check again: the certificate no longer holds.

## Deep

1. **Zero init.** Certification randomises the weights. What bug would a zero-initialised
   `w_down` hide if it did not?
2. **Names.** What error do you get when the reference asks for `w_out`? Is the message enough
   to fix it?
3. **What it does not prove.** Certification checks forward values. Name two ways a certified
   block can still train badly.

## Reading

- **B**: the block and reference you wrote.
- **R**: [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668) (squared ReLU).
- **S**: `docs/blocks.md` in the repo, the section on registering a block.
