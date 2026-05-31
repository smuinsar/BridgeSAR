"""Plotting helpers: the per-date profile-fit diagnostic and the clearance
time-series comparison used by the example notebook (time-series only — no
scatter)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .geometry import perp_separation_px


def _naive(series):
    """Return a tz-naive datetime Series whether the input is tz-aware or not."""
    s = pd.Series(series).reset_index(drop=True)
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s

# Consistent scattering-bounce colors.
BOUNCE_COLORS = {"single": "#d62728", "double": "#2ca02c", "triple": "#1f77b4"}


def plot_accumulated_profile_fit(res, center_row, dx, *, title=None, save_path=None,
                                 xwin_px=200):
    """Per-date accumulated-energy profile + three-peak fit figure.

    Worker-safe: never calls plt.show; always closes the figure.
    """
    import matplotlib as mpl
    p = res["params"]
    u = np.asarray(res["profile_u"])
    prof = np.asarray(res["profile"])
    sigma = max(p["sigma_px"], 0.5)
    cls = res["centerlines"]
    centers = np.array([cls[k][0] * center_row + cls[k][1] for k in range(3)])

    mask = np.ones(prof.shape, bool)
    excl = max(4.0, 3.0 * sigma)
    for c in centers:
        mask &= np.abs(u - c) > excl
    bg = prof[mask] if mask.sum() > 10 else prof
    bg_med = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_med)))
    bg_sig = max(1.4826 * bg_mad, 1e-6)

    span = float(np.max(centers) - np.min(centers))
    margin = max(8.0 * sigma, 0.9 * span, 28.0)
    win = min(float(xwin_px), span + 2.0 * margin)
    half = win / 2.0
    xc = 0.5 * (float(np.max(centers)) + float(np.min(centers)))
    lo, hi = xc - half, xc + half

    _rc_keys = ("font.family", "font.serif", "mathtext.fontset", "axes.unicode_minus")
    _saved_rc = {k: mpl.rcParams[k] for k in _rc_keys}
    mpl.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "axes.unicode_minus": True})
    try:
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.axhspan(bg_med - bg_sig, bg_med + bg_sig, color="grey", alpha=0.15,
                   label=f"background +/-sigma ({bg_med:+.2f} +/- {bg_sig:.2f})")
        ax.axhline(bg_med, color="grey", lw=0.7, ls=":")
        ax.plot(u, prof, lw=1.1, color="k", label="accumulated profile")

        fp_idx = np.asarray(res.get("fp_idx", np.empty(0, int)), dtype=int)
        if fp_idx.size:
            ax.plot(u[fp_idx], prof[fp_idx], "|", color="0.45", alpha=0.7, ms=11,
                    mew=1.2, label="find_peaks candidates")
        dpk = np.asarray(res.get("deriv_peaks_u", np.empty(0, float)), dtype=float)
        if dpk.size:
            dvals = np.interp(dpk, u, prof)
            ax.plot(dpk, dvals, "x", color="0.25", ms=7, mew=1.4,
                    label="d'=0, d''<0 (derivative)")

        in_win = (u >= lo) & (u <= hi)
        pv = prof[in_win] if in_win.any() else prof
        pmin = float(min(pv.min(), bg_med - bg_sig))
        pmax = float(pv.max())
        yr = max(pmax - pmin, 1e-6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(pmin - 0.06 * yr, pmax + 0.48 * yr)

        bounce_keys = ["single", "double", "triple"]
        labels = ["single (S)", "double (D)", "triple (T)"]
        vx = np.asarray(centers, dtype=float)
        vy = np.array([float(np.interp(c, u, prof)) for c in centers])
        for c, v, key in zip(vx, vy, bounce_keys):
            col = BOUNCE_COLORS[key]
            ax.axvspan(c - sigma, c + sigma, color=col, alpha=0.15)
            ax.axvline(c, color=col, lw=1.3, ls="--")
            ax.plot([c], [v], "o", color=col, mec="k", ms=7, zorder=5)

        inset = 0.16 * win
        min_gap = 0.24 * win
        xa = vx.copy()
        order_idx = np.argsort(xa)
        for j in range(1, order_idx.size):
            a, b = order_idx[j - 1], order_idx[j]
            if xa[b] - xa[a] < min_gap:
                xa[b] = xa[a] + min_gap
        right_over = xa[order_idx[-1]] - (hi - inset)
        if right_over > 0:
            xa -= right_over
        left_over = (lo + inset) - xa[order_idx[0]]
        if left_over > 0:
            xa += left_over
        y_lab = pmax + 0.12 * yr
        for c, v, x_lab, key, lab in zip(vx, vy, xa, bounce_keys, labels):
            col = BOUNCE_COLORS[key]
            snr = (v - bg_med) / bg_sig
            ax.annotate(f"{lab}\nu = {c:.1f} pixel\nSNR = {snr:.1f}",
                        xy=(c, v), xytext=(x_lab, y_lab), textcoords="data",
                        ha="center", va="bottom", fontsize=9, color=col,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.6,
                                        alpha=0.7, shrinkA=2, shrinkB=4),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=col, lw=0.6, alpha=0.9))

        sd_px = perp_separation_px(cls[0], cls[1], center_row)
        dt_px = perp_separation_px(cls[1], cls[2], center_row)
        sd_m, dt_m = sd_px * dx, dt_px * dx
        info = (f"SD = {sd_px:.2f} pixel ({sd_m:.2f} m)\n"
                f"DT = {dt_px:.2f} pixel ({dt_m:.2f} m)")
        ax.text(0.01, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.92))

        ax.set_xlabel("Shear-corrected column at center row (pixel)", fontsize=11)
        ax.set_ylabel("Accumulated energy", fontsize=11)
        ax.set_title(title or "Profile fit", fontsize=13, pad=12)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
            fig.savefig(Path(save_path).with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    finally:
        mpl.rcParams.update(_saved_rc)
    return save_path


# Time-series palette (matches the manuscript clearance_timeseries figure).
_COLOR_CONT = "#888888"     # continuous reference trace
_COLOR_REF_ACQ = "#1f77b4"  # acquisition-matched reference
_COLOR_SAR = "#d62728"      # BridgeSAR separation-based clearance


def plot_clearance_timeseries(matched, *, ref_curve=None, ref_points=None,
                              h_col="H_mean_m", sigma_col="sigma_H_mean_m",
                              deck_height_m=None, ref_label="reference clearance",
                              title=None, save_path=None, dpi=300, ax=None,
                              halfwin_hours=3.0, figsize=(14, 5), y_pad_frac=0.35):
    """Time-series comparison of BridgeSAR clearance against a reference.

    Styled after the manuscript ``clearance_timeseries`` figure: a gray
    continuous reference trace, blue acquisition-matched reference markers, and
    red open-square BridgeSAR clearance with ±1σ error bars, in Times New Roman.

    Parameters
    ----------
    matched : DataFrame with ``sensing_time``, ``h_col`` and ``sigma_col``.
    ref_curve : optional DataFrame with columns ``t`` (UTC) and ``v`` — a
        continuous reference-clearance curve (e.g. the rolling-mean air-gap).
    ref_points : optional DataFrame with ``sensing_time`` and ``v`` — the
        acquisition-matched reference clearance at each SAR date.
    deck_height_m : optional horizontal reference line.
    save_path : if given, the figure is written as a 300-DPI PNG (tight layout,
        ``pad_inches=0``). The default is ``None`` (no file); the example
        notebook passes an explicit path so a PNG is always saved.
    ax : draw into an existing Axes instead of creating a figure.

    Returns
    -------
    (fig, ax) when a new figure is created, else the Axes. The created figure is
    closed before returning so notebooks display it exactly once.
    """
    import matplotlib as mpl
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    created = ax is None
    _rc_keys = ("font.family", "font.serif", "mathtext.fontset", "axes.unicode_minus")
    _saved_rc = {k: mpl.rcParams[k] for k in _rc_keys}
    mpl.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "axes.unicode_minus": True})
    try:
        if created:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Continuous reference trace (gray).
        if ref_curve is not None and len(ref_curve):
            ax.plot(_naive(ref_curve["t"]), ref_curve["v"].values,
                    color=_COLOR_CONT, alpha=0.35, lw=0.8, zorder=1,
                    label=f"{ref_label} (continuous)")
        # Acquisition-matched reference (blue line + circles).
        if ref_points is not None and len(ref_points):
            tp = _naive(ref_points["sensing_time"])
            ax.plot(tp, ref_points["v"].values, color=_COLOR_REF_ACQ, lw=0.8,
                    marker="o", ms=3.5, mfc=_COLOR_REF_ACQ, mec=_COLOR_REF_ACQ,
                    zorder=2, label=f"{ref_label} (±{halfwin_hours:g} h)")

        # BridgeSAR clearance (red open squares + ±1σ error bars).
        t = _naive(matched["sensing_time"])
        yerr = (matched[sigma_col].fillna(0).values
                if sigma_col in matched else None)
        if yerr is not None:
            ax.errorbar(t, matched[h_col].values, yerr=yerr, fmt="none",
                        ecolor=_COLOR_SAR, elinewidth=0.6, capsize=0,
                        alpha=0.35, zorder=2)
        ax.plot(t, matched[h_col].values, color=_COLOR_SAR, lw=0, marker="s",
                ms=3.6, mfc="none", mec=_COLOR_SAR, mew=0.8, zorder=3,
                label="BridgeSAR clearance ($\\pm1\\sigma$)")

        if deck_height_m is not None:
            ax.axhline(deck_height_m, color="black", ls=":", lw=0.8, zorder=1,
                       label=f"nominal clearance = {deck_height_m:.0f} m")

        ax.set_ylabel("Clearance (m)", fontsize=13)
        ax.set_xlabel("Date", fontsize=13)
        ax.tick_params(labelsize=11)
        if title:
            ax.set_title(title, fontsize=14, pad=10)

        # Expand the y-limits beyond the data so the series are not crammed
        # against the frame (y_pad_frac of the data span added on each side).
        ydata = [matched[h_col].values]
        if yerr is not None:
            ydata += [matched[h_col].values - yerr, matched[h_col].values + yerr]
        if ref_points is not None and len(ref_points):
            ydata.append(ref_points["v"].values)
        if ref_curve is not None and len(ref_curve):
            ydata.append(ref_curve["v"].values)
        yall = np.concatenate([np.asarray(a, float).ravel() for a in ydata])
        yall = yall[np.isfinite(yall)]
        if yall.size:
            lo, hi = float(yall.min()), float(yall.max())
            span = hi - lo if hi > lo else max(abs(hi), 1.0)
            ax.set_ylim(lo - y_pad_frac * span, hi + y_pad_frac * span)

        loc = AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
        ax.grid(axis="x", which="major", lw=0.4, alpha=0.35, color="0.5")
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

        if created:
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)
            return fig, ax
        return ax
    finally:
        mpl.rcParams.update(_saved_rc)
