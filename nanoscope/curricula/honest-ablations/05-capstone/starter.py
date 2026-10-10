"""Honest ablations 5: the capstone.   Check: nanoscope learn check honest-ablations/05-capstone

Run the study from lesson.md first (it trains 8 variants on 3 seeds and takes about 20 minutes
on a laptop CPU). Read its report. Then fill in the two answers below from what you saw, and run
the check. It does not train anything: it reads your study and holds your answers against it.
"""

# The difference in val_bpb between the full modern model and GPT-2, as an interval you stand
# behind: GAP = (low, high). Negative means the modern model is better. The measured value must
# fall inside it, and it must not be so wide that it says nothing.
GAP = None  # TODO: e.g. (-0.2, -0.1), from the interval your study reports

# One component whose removal changed nothing you can detect: the name of the variant, e.g.
# "no-rope". Say None if you think every component matters. Naming one that does matter fails.
NULL_COMPONENT = "?"  # TODO

# What you would tell a colleague in two sentences. Not checked; it is the part that matters.
SUMMARY = """
TODO: the gap, its interval, which components you can and cannot tell from noise, and what you
would run next to settle the ones you cannot.
"""
