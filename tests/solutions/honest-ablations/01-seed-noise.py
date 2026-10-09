from nanoscope.models import GPT2


class MyGPT(GPT2):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__(vocab_size, context_length, d_model, n_layers, n_heads)
