"""Extend 2: a preset.   Check: nanoscope learn check extend/02-preset"""
from nanoscope.presets import get_preset, register_preset

base = get_preset("tinystories-5min")

# TODO: register a preset named "quick-smoke", a hundred steps or fewer, built from `base` with
# base.override(...). Keep the data fields as they are; change the schedule so that it makes sense.
