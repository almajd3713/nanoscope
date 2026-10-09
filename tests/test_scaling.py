import math
import random

import pytest

from nanoscope.scaling import fit_power_law


def ladder(exponent, coefficient, xs, noise=0.0, seed=0):
    rng = random.Random(seed)
    return [(x, coefficient * x**exponent * math.exp(rng.gauss(0, noise))) for x in xs]


def test_an_exact_power_law_is_recovered():
    fit = fit_power_law(ladder(-0.3, 5.0, [1e3, 1e4, 1e5, 1e6, 1e7]))
    assert fit.exponent == pytest.approx(-0.3)
    assert fit.coefficient == pytest.approx(5.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.exponent_ci95 is not None
    low, high = fit.exponent_ci95
    assert low <= -0.3 <= high and high - low < 1e-9


def test_the_interval_covers_the_truth_about_95_percent_of_the_time():
    hits = 0
    trials = 300
    for seed in range(trials):
        xs = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5]
        fit = fit_power_law(ladder(-0.25, 3.0, xs, noise=0.02, seed=seed))
        assert fit.exponent_ci95 is not None
        hits += fit.exponent_ci95[0] <= -0.25 <= fit.exponent_ci95[1]
    assert 0.92 <= hits / trials <= 0.98


def test_three_points_give_a_wide_interval_and_say_so():
    few = fit_power_law(ladder(-0.3, 5.0, [1e3, 1e4, 1e5], noise=0.02, seed=1))
    many = fit_power_law(ladder(-0.3, 5.0, [1e3, 1e4, 1e5] * 4, noise=0.02, seed=1))
    assert few.exponent_ci95 is not None and many.exponent_ci95 is not None
    assert few.exponent_ci95[1] - few.exponent_ci95[0] > 3 * (
        many.exponent_ci95[1] - many.exponent_ci95[0])
    assert "only 3 points" in few.status and many.status == "estimated"


def test_two_points_fit_exactly_without_an_interval():
    fit = fit_power_law([(1.0, 4.0), (16.0, 1.0)])
    assert fit.exponent == pytest.approx(-0.5)
    assert fit.exponent_ci95 is None and fit.predict(4.0)[1:] == (None, None)
    assert fit.predict(4.0)[0] == pytest.approx(2.0)


def test_prediction_interval_widens_away_from_the_ladder():
    fit = fit_power_law(ladder(-0.3, 5.0, [1e3, 1e4, 1e5, 1e6], noise=0.03, seed=2))
    mid_in, lo_in, hi_in = fit.predict(1e4)
    mid_out, lo_out, hi_out = fit.predict(1e9)
    assert lo_in is not None and hi_in is not None and lo_out is not None and hi_out is not None
    assert lo_in < mid_in < hi_in
    assert math.log(hi_out / lo_out) > math.log(hi_in / lo_in)


def test_bad_input_is_refused_in_plain_words():
    with pytest.raises(ValueError, match="at least two points"):
        fit_power_law([(1.0, 1.0)])
    with pytest.raises(ValueError, match="positive, finite"):
        fit_power_law([(1.0, 1.0), (2.0, -1.0)])
    with pytest.raises(ValueError, match="two different x"):
        fit_power_law([(2.0, 1.0), (2.0, 3.0)])
    with pytest.raises(ValueError, match="positive finite x"):
        fit_power_law([(1.0, 1.0), (2.0, 2.0)]).predict(0)


def test_to_dict_is_plain_data():
    d = fit_power_law(ladder(-0.3, 5.0, [1e3, 1e4, 1e5, 1e6])).to_dict()
    assert set(d) >= {"exponent", "exponent_ci95_low", "exponent_ci95_high", "n", "status"}
