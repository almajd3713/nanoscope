import torch

from nanoscope.config import DataConfig, TokenizerConfig
from nanoscope.data import build_batch_stream


def make_stream(seed: int = 7):
    return build_batch_stream(
        DataConfig(
            source="fixture",
            documents=["alpha", "beta", "gamma"],
            sequence_length=5,
            shuffle_buffer=3,
        ),
        TokenizerConfig(name="byte", eos_token_id=256),
        seed=seed,
        batch_size=2,
    )


def test_stream_resume_preserves_next_batch_and_buffer() -> None:
    original = make_stream()
    original.next_batch()
    state = original.state_dict()
    expected = original.next_batch()

    restored = make_stream()
    restored.load_state_dict(state)
    assert torch.equal(restored.next_batch(), expected)
    assert restored.state_dict() == original.state_dict()


def test_seed_controls_document_order() -> None:
    one = make_stream(7).next_batch()
    two = make_stream(7).next_batch()
    different = make_stream(8).next_batch()
    assert torch.equal(one, two)
    assert not torch.equal(one, different)


def test_packing_inserts_eos_without_padding() -> None:
    stream = build_batch_stream(
        DataConfig(source="fixture", documents=["a"], sequence_length=3),
        TokenizerConfig(name="byte", eos_token_id=256),
        seed=1,
        batch_size=1,
    )
    assert stream.next_batch().tolist() == [[97, 256, 97, 256]]

