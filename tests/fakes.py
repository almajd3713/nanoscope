"""A tiny preset over the synthetic stories from conftest.fake_data; trains in seconds."""

from nanoscope import Preset


def tiny(**kw):
    base = Preset(
        name="test-tiny", dataset="fake/stories", tokenizer="bpe", vocab_size=300,
        context_length=32, max_steps=20, batch_size=4, learning_rate=1e-2, warmup_steps=2,
        train_docs=200, tokenizer_train_docs=100, eval_docs=10, eval_interval=10,
        sample_interval=10, sample_length=20, checkpoint_interval=10,
    )
    return base.override(**kw)
