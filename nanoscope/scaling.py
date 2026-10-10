"""Fit a power law to a ladder of runs, and say how sure the fit is.

    fit = fit_power_law([(1e15, 1.52), (4e15, 1.38), (1.6e16, 1.27)])
    fit.exponent, fit.exponent_ci95    # the slope on log-log axes, with its 95% interval
    fit.predict(6.4e16)                # (estimate, low, high) for a point you have not run

The fit is ordinary least squares on log y against log x, so it assumes y = a * x**b with no
floor. A loss that flattens toward an irreducible term is not a pure power law, and fitting
one gives a flatter exponent than the truth; a short ladder cannot tell the two apart. With n
points the interval uses n - 2 degrees of freedom, so three points give a very wide one on
purpose: that width is the honest answer to "what did a three-run ladder tell us?".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PowerLawFit:
    exponent: float
    exponent_ci95: tuple[float, float] | None
    coefficient: float
    r_squared: float | None
    n: int
    status: str
    _log_x_mean: float = 0.0
    _sxx: float = 0.0
    _resid_sd: float | None = None
    _intercept: float = 0.0

    def predict(self, x: float) -> tuple[float, float | None, float | None]:
        """(estimate, low, high): the fitted y at x, with a 95% prediction interval.

        The interval is for one new run, so it includes the run-to-run scatter around the
        line. It is None when there are fewer than three points. Far outside the ladder it
        widens, and it still assumes the power law keeps holding there."""
        if not x > 0 or not math.isfinite(x):
            raise ValueError(f"predict needs a positive finite x, got {x!r}")
        log_x = math.log(x)
        mid = self._intercept + self.exponent * log_x
        if self._resid_sd is None:
            return math.exp(mid), None, None
        from scipy.stats import t

        margin = float(t.ppf(0.975, self.n - 2)) * self._resid_sd * math.sqrt(
            1 + 1 / self.n + (log_x - self._log_x_mean) ** 2 / self._sxx)
        return math.exp(mid), math.exp(mid - margin), math.exp(mid + margin)

    def to_dict(self) -> dict[str, Any]:
        low, high = self.exponent_ci95 or (None, None)
        return {
            "exponent": self.exponent, "exponent_ci95_low": low, "exponent_ci95_high": high,
            "coefficient": self.coefficient, "r_squared": self.r_squared, "n": self.n,
            "status": self.status,
        }


def fit_power_law(points: Iterable[tuple[float, float]]) -> PowerLawFit:
    """Fit y = coefficient * x**exponent to (x, y) points by least squares in log space.

    Repeated x values (several seeds per rung) are fine: pass every run as its own point."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        raise ValueError("a power-law fit needs at least two points")
    if any(not (x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)) for x, y in pts):
        raise ValueError("a power-law fit needs positive, finite x and y (it works in logs)")
    log_x = [math.log(x) for x, _ in pts]
    log_y = [math.log(y) for _, y in pts]
    n = len(pts)
    x_mean, y_mean = sum(log_x) / n, sum(log_y) / n
    sxx = sum((x - x_mean) ** 2 for x in log_x)
    if sxx == 0:
        raise ValueError("a power-law fit needs at least two different x values")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(log_x, log_y, strict=True)) / sxx
    intercept = y_mean - slope * x_mean
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(log_x, log_y, strict=True))
    sst = sum((y - y_mean) ** 2 for y in log_y)
    r2 = 1 - sse / sst if sst > 0 else None
    ci = resid_sd = None
    status = "exploratory: two points fit exactly and carry no interval"
    if n >= 3:
        from scipy.stats import t

        resid_sd = math.sqrt(sse / (n - 2))
        margin = float(t.ppf(0.975, n - 2)) * resid_sd / math.sqrt(sxx)
        ci = (slope - margin, slope + margin)
        status = "estimated" if n >= 5 else f"estimated from only {n} points: read the interval"
    return PowerLawFit(
        exponent=slope, exponent_ci95=ci, coefficient=math.exp(intercept), r_squared=r2, n=n,
        status=status, _log_x_mean=x_mean, _sxx=sxx, _resid_sd=resid_sd, _intercept=intercept)
