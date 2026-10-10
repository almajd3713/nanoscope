from nanoscope.presets import get_preset, register_preset

base = get_preset("tinystories-5min")

register_preset(base.override(name="quick-smoke", max_steps=60, warmup_steps=10,
                               eval_interval=20, sample_interval=60, checkpoint_interval=60))
