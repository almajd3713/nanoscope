"""Extend 1: a certified block.   Check: nanoscope learn check extend/01-certified-block"""
import torch
import torch.nn as nn

from nanoscope.blocks import register_block


def squared_relu_mlp(x, w_up, w_down):
    """The slow, obvious version: relu(x @ w_up) squared, then @ w_down."""
    # TODO
    raise NotImplementedError


# TODO: decorate the class so the palette knows it and its reference:
#   @register_block(reference=squared_relu_mlp, family="mlp")
class SquaredReLUMLP(nn.Module):
    def __init__(self, d_model, context_length, hidden=None):
        super().__init__()
        # TODO: two parameters named w_up (d_model x hidden) and w_down (hidden x d_model);
        # hidden defaults to 4 * d_model
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError
