"""Lesson 5, built from blocks: the transformer layer as four slots to fill.

The same MyBlock as the code version, without writing forward(): the residual connections
(x + attention of the normalised x, then x + the MLP of the normalised x) are fixed in
BlockTemplate. On the model page, drag a block into each slot: two LayerNorms, Attention with
4 heads, and a GELUMLP. Then run the check.
"""

from nanoscope.blocks import BlockTemplate


class MyBlock(BlockTemplate):
    def __init__(self, d_model, context_length):
        super().__init__(d_model, context_length, norm1=None, attn=None, norm2=None, mlp=None)
