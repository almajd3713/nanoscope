"""Lesson 3, built from blocks: one causal attention head as eight slots to fill.

The same OneHead as starter.py, without writing forward(): drag a block from the palette into
each slot (the model page's graph), then run:  nanoscope learn check foundations/03-attention-head
The slots are q, k, v (the three projections), scores, mask, normalize, mix and out.
"""

from nanoscope.blocks import AttentionTemplate


class OneHead(AttentionTemplate):
    def __init__(self, d_model, context_length):
        super().__init__(
            d_model, context_length,
            q=None, k=None, v=None, scores=None, mask=None, normalize=None, mix=None, out=None,
        )
