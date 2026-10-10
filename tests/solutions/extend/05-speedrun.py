from nanoscope.models import Modern


class Speedrun(Modern):
    """What the ablation kept: RoPE and RMSNorm, with a GELU MLP, full multi-head attention and
    neither QK-norm nor z-loss. SwiGLU, GQA, QK-norm and z-loss all lost ground in 100 steps."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        super().__init__(vocab_size, context_length, d_model, n_layers, n_kv_heads=4,
                         swiglu=False, ffn_hidden=240, qk_norm=False, z_loss=0.0)
