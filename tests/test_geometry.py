"""Lightweight unit tests that need no network or data downloads."""

import numpy as np
import pytest

from bridgesar.geometry import perp_separation_px, polygon_corners, polygon_centerline
from bridgesar.stripes import thickness_corrections, snr_sigma_spacing
from bridgesar.config import HyperParams


def test_perp_separation_horizontal_lines():
    # Two horizontal lines (slope 0) offset by 5 px -> separation 5 px.
    line_a = (0.0, 10.0, None)   # y = 0*x + 10
    line_b = (0.0, 15.0, None)   # y = 0*x + 15
    assert perp_separation_px(line_a, line_b, eval_row=123.0) == pytest.approx(5.0)


def test_perp_separation_three_stripe_geometry():
    # S, D, T parallel lines at intercepts 0, 8, 18 -> SD=8, DT=10, ST=18.
    S = (0.0, 0.0, None)
    D = (0.0, 8.0, None)
    T = (0.0, 18.0, None)
    r = 50.0
    assert perp_separation_px(S, D, r) == pytest.approx(8.0)
    assert perp_separation_px(D, T, r) == pytest.approx(10.0)
    assert perp_separation_px(S, T, r) == pytest.approx(18.0)


def test_polygon_centerline_roundtrip():
    corners = polygon_corners(center_col=100.0, half_width_px=10.0,
                              rotation_deg=0.0, length_px=200.0, center_row=50.0)
    slope, intercept, _ = polygon_centerline(corners)
    # Unrotated rectangle -> vertical centerline at column 100 for all rows.
    assert intercept == pytest.approx(100.0, abs=1e-6)
    assert slope == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("model", ["symmetric", "lower_triple", "lower_single", "mixed"])
def test_thickness_corrections_finite(model):
    c = thickness_corrections(model, 9.0)
    assert len(c) == 3
    assert all(np.isfinite(v) for v in c)


def test_thickness_corrections_unknown():
    with pytest.raises(ValueError):
        thickness_corrections("bogus", 1.0)


def test_snr_sigma_spacing_positive():
    hp = HyperParams()
    q = {"sigma_px": 2.0, "snr_S": 5.0, "snr_D": 4.0, "snr_T": 3.0}
    sd, dt = snr_sigma_spacing(q, dx=4.0, hp=hp)
    assert sd > 0 and dt > 0
