A chip has two limits: how fast it can do arithmetic, and how fast it can read memory. Which one
stops you decides what will make the model faster.

## Surface

`nanoscope bench` prints TFLOPs: the model's FLOPs per step (the library counts them per block)
divided by the step time. That number means little alone. Divided by the chip's peak it becomes
**MFU**, model FLOPs utilisation: the share of the chip you actually use.

```bash
nanoscope bench nanoscope/models/gpt2.py:GPT2 --compile reduce-overhead
```

On an RTX 4070 laptop GPU this gave 1.52 TFLOPs. `bench` prints an MFU only for chips it knows the
peak of (T4, P100). For yours, look up the peak FP32 (or BF16) TFLOP/s and divide. The cases in
`lesson.toml` use a made-up 22 TFLOP/s chip so that the numbers are checkable, not your hardware's.

Write four functions in `starter.py`:

- `mfu(tflops, peak_tflops)`.
- `arithmetic_intensity(flops, bytes_moved)`: FLOPs per byte of memory traffic. A matrix multiply
  of two `n x n` matrices does `2n^3` FLOPs on `3n^2` numbers, so its intensity grows with `n`.
  Adding two vectors does one FLOP per two numbers read: intensity near zero.
- `attainable_tflops(intensity, peak_tflops, bandwidth_gbs)`: the roofline. Memory delivers
  `intensity x bandwidth` FLOP/s; the chip cannot exceed its peak, so take the smaller.
- `memory_bound(intensity, peak_tflops, bandwidth_gbs)`: true when the memory side is the smaller.

```bash
nanoscope learn check efficiency/04-roofline
```

### Reading the result

For a chip with a 20 TFLOP/s peak and 300 GB/s of bandwidth, the ridge sits at
`20e12 / 300e9 = 66.7` FLOPs per byte. Below it a kernel is memory-bound, and a faster
multiplier changes nothing; fuse kernels or move less data. Above it, arithmetic is the limit.
Normalisation, softmax and elementwise activations sit far below the ridge. Large matrix
multiplies sit above it. That is why `torch.compile` fuses the former and leaves the latter alone.

A small model can sit below the roofline for a third reason: overhead (lesson 1). MFU low and the
GPU busy fraction low means Python is the limit, not memory or arithmetic.

## Deep

1. **Sweep d_model.** Bench GPT2 at `d_model` 64, 128, 256, 512. Plot achieved TFLOPs. Where does it
   stop rising, and what is the busy fraction there?
2. **Where is the ridge for your chip?** Find its peak and bandwidth from the spec sheet and compute
   it. Is a `d_model x d_model` matrix multiply on a batch of 2,048 tokens above or below?
3. **Why is MFU of 40% good?** Published large-model runs report 30-50%. List what a real
   training run spends time on besides matrix multiplies.

## Reading

- **B**: the four functions you wrote.
- **R**: [Roofline: an insightful visual performance model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) (Williams, Waterman, Patterson, 2009).
- **R**: [PaLM](https://arxiv.org/pdf/2204.02311), appendix B: where the MFU definition comes from.
- **S**: [FlashAttention](https://arxiv.org/pdf/2205.14135): an attention kernel designed around memory traffic.
