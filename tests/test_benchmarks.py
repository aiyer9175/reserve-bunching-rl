import numpy as np
import pytest

from rbrl import benchmarks as bm
from rbrl.distributions import Uniform


@pytest.mark.parametrize("N", [2, 4, 6])
@pytest.mark.parametrize("r", [0.0, 0.1, 0.5])
def test_closed_form_matches_numeric_integral(N, r):
    v = np.linspace(max(r, 1e-3), 1, 201)
    np.testing.assert_allclose(bm.b_star_uniform(v, r, N), bm.b_star_numeric(v, r, N, Uniform()), atol=1e-10)


@pytest.mark.parametrize("N", [2, 4, 6])
def test_equilibrium_is_flat_at_reserve(N):
    r, h = 0.5, 1e-5
    assert bm.b_star_uniform(np.array(r), r, N) == pytest.approx(r)
    slope = (bm.b_star_uniform(np.array(r + h), r, N) - r) / h
    assert abs(slope) < 1e-3


def test_equilibrium_bid_below_value_and_nan_below_reserve():
    v = np.linspace(0, 1, 101)
    b = bm.b_star_uniform(v, 0.5, 4)
    assert np.isnan(b[v < 0.5]).all()
    assert (b[v >= 0.5] <= v[v >= 0.5] + 1e-12).all()


def test_heuristic_bunches_exactly_at_reserve_up_to_v_tilde():
    N, r = 4, 0.5
    vt = bm.v_tilde(r, N)
    assert vt == pytest.approx(2 / 3)
    v = np.array([0.4, 0.5, 0.6, vt - 1e-9, 0.8])
    b = bm.heuristic_bid(v, r, N)
    assert np.isnan(b[0])
    assert (b[1:4] == r).all()
    assert b[4] == pytest.approx(0.6)


def test_v_tilde_numeric_matches_closed_form():
    # Force the numeric branch with a non-standard support.
    d = Uniform(0.0, 2.0)
    assert bm.v_tilde(0.5, 4, d) == pytest.approx(2 / 3, rel=1e-6)


def test_revenues():
    assert bm.equilibrium_revenue(0.5, 4, Uniform()) == pytest.approx(49 / 80, abs=1e-6)
    expected = 0.5 * ((2 / 3) ** 4 - 0.5**4) + 0.6 * (1 - (2 / 3) ** 5)  # closed form
    assert bm.heuristic_revenue(0.5, 4, Uniform()) == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(0.5885, abs=5e-5)
