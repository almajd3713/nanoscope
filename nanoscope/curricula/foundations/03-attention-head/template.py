"""Lesson 3, built from blocks: one causal attention head as eight slots to fill.

The same OneHead as the code version, without writing forward(): on the model page, drag a block
from the palette into each slot of the graph, then run the check. The slots are q, k, v (the three
projections), scores, mask, normalize, mix and out, in the order of the five steps.
"""

from nanoscope.blocks import AttentionTemplate


class OneHead(AttentionTemplate):
    def __init__(self, d_model, context_length):
        super().__init__(
            d_model, context_length,
            q=None, k=None, v=None, scores=None, mask=None, normalize=None, mix=None, out=None,
        )
