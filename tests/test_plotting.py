"""Smoke test for the clearance time-series plot (no data download)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from bridgesar.plotting import plot_clearance_timeseries


def test_plot_clearance_timeseries_runs():
    t = pd.date_range("2022-01-01", "2022-12-31", periods=20, tz="UTC")
    matched = pd.DataFrame({"sensing_time": t,
                            "H_mean_m": 62 + np.random.randn(20),
                            "sigma_H_mean_m": np.abs(np.random.randn(20)) * 0.3})
    ref_curve = pd.DataFrame({"t": pd.date_range("2022-01-01", "2022-12-31",
                                                 periods=200, tz="UTC"),
                              "v": 62 + np.sin(np.linspace(0, 6, 200))})
    # ref_points built from .values (tz dropped) — must still plot.
    ref_points = pd.DataFrame({"sensing_time": matched["sensing_time"].values,
                               "v": 62 + np.random.randn(20)})
    fig, ax = plot_clearance_timeseries(matched, ref_curve=ref_curve,
                                        ref_points=ref_points, deck_height_m=62.0,
                                        ref_label="air gap", title="test")
    assert len(ax.lines) >= 3
