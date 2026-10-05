"""Slow, obvious implementations that the fast library code is checked against.

Everything here uses only torch, math and the other modules of this package, so a reader can
trust it without reading the library. Loops over heads and positions beat clever tensor code.
"""
