from nanoscope.models import Modern


class MySwiGLU(Modern):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        super().__init__(vocab_size, context_length, d_model, n_layers, swiglu=True,
                         ffn_hidden=171)
