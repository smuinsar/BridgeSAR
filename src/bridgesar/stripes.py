"""stripe extraction: joint single/double/triple-bounce energy fit.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks

from .geometry import (
    polygon_corners,
    polygon_centerline,
    perp_separation_px as perp_sep,
)


# ---------------------------------------------------------------------------
# Profile / image preprocessing
# ---------------------------------------------------------------------------
def robust_fit_image(rot_img):
    x = np.array(rot_img, dtype=np.float32, copy=True)
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    y = np.log1p(x)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(y, [1.0, 99.7])
        y = np.clip(y, lo, hi)
        med = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - med))
    scale = max(1.4826 * mad, 1e-6)
    zimg = (y - med) / scale
    return np.where(np.isfinite(zimg), zimg, np.nan).astype(np.float32)


def make_accumulated_profile(rot_img, center_row, length_px, rotation_deg,
                             center_col_hint, exp_dx, hp, half_search_px=None):
    PROFILE_PAD_PX = hp.profile_pad_px
    PROFILE_CLIP_Z = hp.profile_clip_z
    PROFILE_SMOOTH_SIGMA = hp.profile_smooth_sigma

    nrows, ncols = rot_img.shape
    zimg = robust_fit_image(rot_img)
    if half_search_px is None:
        half_search_px = max(3.5 * exp_dx + PROFILE_PAD_PX, 180.0)
    cmin = int(max(0, np.floor(center_col_hint - half_search_px)))
    cmax = int(min(ncols - 1, np.ceil(center_col_hint + half_search_px)))
    r0 = int(max(0, np.floor(center_row - length_px / 2.0)))
    r1 = int(min(nrows - 1, np.ceil(center_row + length_px / 2.0)))
    rows = np.arange(r0, r1 + 1, dtype=float)
    u = np.arange(cmin, cmax + 1, dtype=float)
    tan_rot = np.tan(np.deg2rad(rotation_deg))
    cols = u[None, :] + tan_rot * (rows[:, None] - center_row)
    rr = np.broadcast_to(rows[:, None], cols.shape)
    vals = map_coordinates(np.nan_to_num(zimg, nan=0.0), [rr, cols], order=1,
                           mode="constant", cval=0.0)
    vld = map_coordinates(np.isfinite(zimg).astype(np.float32), [rr, cols],
                          order=0, mode="constant", cval=0.0) > 0.5
    vals = np.where(vld, vals, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        row_med = np.nanmedian(vals, axis=1, keepdims=True)
        row_mad = np.nanmedian(np.abs(vals - row_med), axis=1, keepdims=True)
        vals = (vals - row_med) / np.maximum(1.4826 * row_mad, 1e-6)
        vals = np.clip(vals, -2.0, PROFILE_CLIP_Z)
        profile = np.nanmean(vals, axis=0)
    profile = np.where(np.isfinite(profile), profile, 0.0)
    if PROFILE_SMOOTH_SIGMA > 0:
        profile = gaussian_filter1d(profile, PROFILE_SMOOTH_SIGMA)
    return u, profile, {"r0": r0, "r1": r1, "cmin": cmin, "cmax": cmax}


def weighted_profile_value(u, profile, center, sigma):
    sigma = max(float(sigma), 0.5)
    off = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1, dtype=float)
    uu = center + off
    vals = np.interp(uu, u, profile, left=np.nan, right=np.nan)
    valid = np.isfinite(vals)
    if valid.sum() < 3:
        return -1e6
    w = np.exp(-0.5 * (off[valid] / sigma) ** 2)
    return float(np.sum(vals[valid] * w) / np.sum(w))


def _profile_noise_floor(profile):
    med = float(np.median(profile))
    mad = float(np.median(np.abs(profile - med))) + 1e-12
    return med, 1.4826 * mad


def find_profile_peaks(u, profile, spacing_init):
    med, sigma_n = _profile_noise_floor(profile)
    distance = max(3, int(round(0.4 * float(spacing_init))))
    dyn = float(np.max(profile) - med)
    prominence = max(0.5 * sigma_n, 0.05 * dyn, 1e-6)
    idx, _ = find_peaks(profile, prominence=prominence, distance=distance)
    return np.asarray(idx, dtype=int)


def derivative_peaks(u, profile):
    if len(u) < 5:
        return np.empty(0, dtype=float)
    d1 = np.gradient(profile, u)
    d2 = np.gradient(d1, u)
    s = np.sign(d1)
    zc = np.where((s[:-1] > 0) & (s[1:] <= 0) & (d2[:-1] < 0))[0]
    bg_med, bg_sig = _profile_noise_floor(profile)
    thr = bg_med + bg_sig
    out = []
    for i in zc:
        y0, y1 = d1[i], d1[i + 1]
        x0, x1 = u[i], u[i + 1]
        if y1 == y0:
            continue
        xv = x0 - y0 * (x1 - x0) / (y1 - y0)
        pv = max(float(profile[i]), float(profile[i + 1]))
        if pv >= thr:
            out.append(xv)
    return np.asarray(out, dtype=float)


def local_subpixel_peak(u, profile, center, search_half_px, dpk=None, fp_idx=None,
                        method="derivative_only"):
    if method == "derivative_only":
        if dpk is None or len(dpk) == 0:
            return float(center)
        d = np.asarray(dpk, dtype=float)
        for mult in (1.0, 2.0, 3.0, 4.0):
            half = mult * search_half_px
            in_win = d[(d >= center - half) & (d <= center + half)]
            if len(in_win):
                vals = np.interp(in_win, u, profile)
                return float(in_win[int(np.argmax(vals))])
        return float(center)

    if dpk is not None and len(dpk):
        d = np.asarray(dpk, dtype=float)
        for half in (search_half_px, 2.0 * search_half_px):
            in_win = d[(d >= center - half) & (d <= center + half)]
            if len(in_win):
                vals = np.interp(in_win, u, profile)
                return float(in_win[int(np.argmax(vals))])
    if fp_idx is not None and len(fp_idx):
        u_fp = u[fp_idx]
        for half in (search_half_px, 2.0 * search_half_px):
            sel = fp_idx[(u_fp >= center - half) & (u_fp <= center + half)]
            if len(sel):
                i = int(sel[int(np.argmax(profile[sel]))])
                if 0 < i < len(profile) - 1:
                    x3 = u[i - 1:i + 2]
                    y3 = profile[i - 1:i + 2]
                    try:
                        a, b, _ = np.polyfit(x3, y3, 2)
                        if a < 0:
                            xv = -b / (2 * a)
                            if x3[0] <= xv <= x3[-1]:
                                return float(xv)
                    except Exception:
                        pass
                return float(u[i])
    m = (u >= center - search_half_px) & (u <= center + search_half_px)
    if m.sum() < 5:
        return float(center)
    uu = u[m]
    pp = profile[m]
    i = int(np.argmax(pp))
    if i == 0 or i == len(pp) - 1:
        return float(uu[i])
    x3 = uu[i - 1:i + 2]
    y3 = pp[i - 1:i + 2]
    try:
        a, b, _ = np.polyfit(x3, y3, 2)
        if a < 0:
            xv = -b / (2 * a)
            if x3[0] <= xv <= x3[-1]:
                return float(xv)
    except Exception:
        pass
    return float(uu[i])


def _nearest_dist(arr, x):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.min(np.abs(arr - float(x))))


# ---------------------------------------------------------------------------
# Joint 3-peak Method-E++
# ---------------------------------------------------------------------------
def method_E(rot_img, exp_dx, osm_rot, dx, hp, los_e, length_buffer_frac=None):
    """Joint 3-peak energy-accumulation fit (single + double + triple bounce)."""
    SAT_LOOK_DIR = hp.sat_look_dir
    NOMINAL_STRIPE_WIDTH_M = hp.nominal_stripe_width_m
    WIDTH_FACTOR_BOUNDS = hp.width_factor_bounds
    SPACING_REL_BOUND = hp.spacing_rel_bound
    LAMBDA_WIDTH_PRIOR = hp.lambda_width_prior
    LAMBDA_SPACING_PRIOR = hp.lambda_spacing_prior
    SIGMA_FACTOR = hp.sigma_factor
    ROT_DELTA_FIT_BOUND_DEG = hp.rot_delta_fit_bound_deg
    ROT_DELTA_BOUND_DEG = hp.rot_delta_bound_deg
    STRIPE_WEIGHTS = np.asarray(hp.stripe_weights, dtype=float)
    PEAK_PICK_METHOD = hp.peak_pick_method
    if length_buffer_frac is None:
        length_buffer_frac = hp.length_buffer_frac

    nrows, ncols = rot_img.shape
    length_px = osm_rot["length_px"] * (1.0 + 2.0 * length_buffer_frac)
    rotation_osm_deg = float(osm_rot["rotation_deg"])
    center_row = float(osm_rot["center_row"])
    center_col_hint = float(osm_rot["center_col"])
    NOMINAL_HALF_WIDTH_PX = NOMINAL_STRIPE_WIDTH_M / (2.0 * dx)
    spacing_init = float(exp_dx)
    shift0_init = center_col_hint - spacing_init
    sp_lo = spacing_init * (1.0 - SPACING_REL_BOUND)
    sp_hi = spacing_init * (1.0 + SPACING_REL_BOUND)
    hw_lo = WIDTH_FACTOR_BOUNDS[0] * NOMINAL_HALF_WIDTH_PX
    hw_hi = WIDTH_FACTOR_BOUNDS[1] * NOMINAL_HALF_WIDTH_PX
    sigma_lo = max(hw_lo / SIGMA_FACTOR, 0.5)
    sigma_hi = max(hw_hi / SIGMA_FACTOR, 0.5)

    profile_cache = {}

    def profile_for(rd):
        key = round(float(rd), 3)
        if key not in profile_cache:
            profile_cache[key] = make_accumulated_profile(
                rot_img, center_row, length_px, rotation_osm_deg + key,
                center_col_hint, exp_dx, hp)
        return profile_cache[key]

    def centers_from_params(shift0, sp_SD, sp_DT, rd):
        rot_total = np.deg2rad(rotation_osm_deg + rd)
        cos_rt = max(abs(np.cos(rot_total)), 1e-6)
        return np.array([shift0, shift0 + sp_SD / cos_rt,
                         shift0 + (sp_SD + sp_DT) / cos_rt], dtype=float)

    def background_for_centers(u, profile, centers, sigma):
        mask = np.ones(profile.shape, dtype=bool)
        excl = max(4.0, 3.0 * sigma)
        for c in centers:
            mask &= np.abs(u - c) > excl
        bg = profile[mask]
        if bg.size < 10:
            bg = profile
        med = float(np.median(bg))
        mad = float(np.median(np.abs(bg - med)))
        return med, max(1.4826 * mad, 1e-6)

    def objective_joint(x):
        shift0, sp_SD, sp_DT, rd, sigma = x
        u, prof, _ = profile_for(rd)
        centers = centers_from_params(shift0, sp_SD, sp_DT, rd)
        bg_med, bg_sig = background_for_centers(u, prof, centers, sigma)
        vals = np.array([weighted_profile_value(u, prof, c, sigma) for c in centers])
        if not np.all(np.isfinite(vals)):
            return 1e6
        snr = (vals - bg_med) / bg_sig
        signal = float(np.sum(STRIPE_WEIGHTS * snr) / STRIPE_WEIGHTS.sum())
        width_pen = LAMBDA_WIDTH_PRIOR * ((sigma * SIGMA_FACTOR - NOMINAL_HALF_WIDTH_PX) / NOMINAL_HALF_WIDTH_PX) ** 2
        spacing_pen = LAMBDA_SPACING_PRIOR * (((sp_SD - spacing_init) / spacing_init) ** 2 +
                                              ((sp_DT - spacing_init) / spacing_init) ** 2)
        return -signal + width_pen + spacing_pen

    shift_lo = max(0.0, shift0_init - 1.3 * spacing_init)
    shift_hi = min(ncols - 1.0, shift0_init + 1.3 * spacing_init)
    boundsJ = [(shift_lo, shift_hi),
               (sp_lo, sp_hi),
               (sp_lo, sp_hi),
               (-ROT_DELTA_FIT_BOUND_DEG, ROT_DELTA_FIT_BOUND_DEG),
               (sigma_lo, sigma_hi)]
    starts = []
    for ds in (-1.0, -0.5, 0.0, 0.5):
        for dsp_sd in (0.85, 1.0, 1.15):
            for dsp_dt in (0.85, 1.0, 1.15):
                for dr in (-2.0, 0.0, 2.0):
                    starts.append([np.clip(shift0_init + ds * spacing_init, shift_lo, shift_hi),
                                   np.clip(dsp_sd * spacing_init, sp_lo, sp_hi),
                                   np.clip(dsp_dt * spacing_init, sp_lo, sp_hi),
                                   np.clip(dr, -ROT_DELTA_FIT_BOUND_DEG, ROT_DELTA_FIT_BOUND_DEG),
                                   np.clip(NOMINAL_HALF_WIDTH_PX / SIGMA_FACTOR, sigma_lo, sigma_hi)])
    scored = sorted(((objective_joint(np.asarray(x0, float)), x0) for x0 in starts),
                    key=lambda t: t[0])
    top_starts = [x0 for _, x0 in scored[:8]]
    bestJ = None
    for x0 in top_starts:
        res = minimize(objective_joint, np.asarray(x0, float), method="L-BFGS-B",
                       bounds=boundsJ, options={"maxiter": 100, "eps": 0.15, "ftol": 1e-4})
        if bestJ is None or res.fun < bestJ.fun:
            bestJ = res
    resJ = bestJ
    if (not resJ.success) or resJ.nit < 3:
        resJ2 = minimize(objective_joint, resJ.x, method="Powell", bounds=boundsJ,
                         options={"maxiter": 400, "xtol": 1e-3, "ftol": 1e-4})
        if resJ2.fun < resJ.fun:
            resJ = resJ2
    shift0, sp_SD, sp_DT, rd, sigma_locked = [float(v) for v in resJ.x]
    if ROT_DELTA_BOUND_DEG == 0.0:
        rd = 0.0

    uA, profA, prof_meta = profile_for(rd)
    cA = centers_from_params(shift0, sp_SD, sp_DT, rd)
    rot_total_rad = np.deg2rad(rotation_osm_deg + rd)
    cos_rt_val = max(abs(np.cos(rot_total_rad)), 1e-6)
    dpk = derivative_peaks(uA, profA)
    if PEAK_PICK_METHOD == "derivative_only":
        fp_idx = np.empty(0, int)
    else:
        fp_idx = find_profile_peaks(uA, profA, spacing_init)
    c0_ref = local_subpixel_peak(uA, profA, cA[0], 0.25 * spacing_init, dpk=dpk,
                                 fp_idx=fp_idx, method=PEAK_PICK_METHOD)
    c1_ref = local_subpixel_peak(uA, profA, cA[1], 0.25 * spacing_init, dpk=dpk,
                                 fp_idx=fp_idx, method=PEAK_PICK_METHOD)
    c2_ref = local_subpixel_peak(uA, profA, cA[2], 0.25 * spacing_init, dpk=dpk,
                                 fp_idx=fp_idx, method=PEAK_PICK_METHOD)
    if len(fp_idx):
        fp_u = uA[fp_idx]
        dpeak_S = _nearest_dist(fp_u, c0_ref)
        dpeak_D = _nearest_dist(fp_u, c1_ref)
        dpeak_T = _nearest_dist(fp_u, c2_ref)
    else:
        dpeak_S = dpeak_D = dpeak_T = float("nan")
    sp_SD = float(abs(c1_ref - c0_ref) * cos_rt_val)
    sp_DT = float(abs(c2_ref - c1_ref) * cos_rt_val)
    shift0 = float(c0_ref)

    rot_total = rotation_osm_deg + rd
    cs_unsorted = [float(c0_ref), float(c1_ref), float(c2_ref)]
    hw = float(sigma_locked * SIGMA_FACTOR)
    polys_u = [polygon_corners(c, hw, rot_total, length_px, center_row) for c in cs_unsorted]
    cls_u = [polygon_centerline(c) for c in polys_u]

    sign = +1 if SAT_LOOK_DIR == "right" else -1
    is_desc = (los_e * sign > 0)
    intercepts = [il for _, il, _ in cls_u]
    if SAT_LOOK_DIR == "right":
        reverse_order = is_desc
    else:
        reverse_order = not is_desc
    order = np.argsort(intercepts)[::-1] if reverse_order else np.argsort(intercepts)
    polys = [polys_u[i] for i in order]
    cls = [cls_u[i] for i in order]
    dpeak_arr = np.array([dpeak_S, dpeak_D, dpeak_T])[order]

    score = float(resJ.fun)
    return {"corners": polys, "centerlines": cls,
            "profile_u": uA, "profile": profA,
            "fp_idx": fp_idx, "deriv_peaks_u": dpk,
            "track_direction": "descending" if is_desc else "ascending",
            "params": {"shift0_px": float(shift0), "spacing_SD_px": float(sp_SD),
                       "spacing_DT_px": float(sp_DT), "rotation_delta_deg": float(rd),
                       "half_width_px": float(hw),
                       "sigma_px": float(sigma_locked),
                       "rotation_osm_deg": float(rotation_osm_deg),
                       "length_px": float(length_px),
                       "center_row": float(center_row)},
            "dpeak_S_px": float(dpeak_arr[0]),
            "dpeak_D_px": float(dpeak_arr[1]),
            "dpeak_T_px": float(dpeak_arr[2]),
            "score": score, "score_SD": score, "score_DT": score,
            "converged": bool(resJ.success),
            "n_iter": int(getattr(resJ, "nit", 0))}


# ---------------------------------------------------------------------------
# Amplitude-domain quality
# ---------------------------------------------------------------------------
def _centers_grid_from_params(p, rows):
    rot_total = np.deg2rad(p["rotation_osm_deg"] + p["rotation_delta_deg"])
    cos_rt = max(abs(np.cos(rot_total)), 1e-6)
    tan_rt = np.tan(rot_total)
    sh = p["shift0_px"]
    cs0 = np.array([sh, sh + p["spacing_SD_px"] / cos_rt,
                    sh + (p["spacing_SD_px"] + p["spacing_DT_px"]) / cos_rt])
    return cs0[:, None] + tan_rt * (rows - p["center_row"])[None, :]


def _amp_stripe_stats(rot_img, centers_col_grid, sigma, rows, ncols):
    img = np.where(np.isfinite(rot_img), rot_img, 0.0).astype(np.float32)
    valid_img = np.isfinite(rot_img)
    K = int(np.ceil(3.0 * sigma))
    off = np.arange(-K, K + 1, dtype=float)
    w = np.exp(-0.5 * (off / sigma) ** 2)
    stripe_mean = np.zeros(3)
    excl = np.zeros((rows.size, ncols), bool)
    EXCL_HALF = max(K + 2, 6)
    for k in range(3):
        cc = centers_col_grid[k]
        cols = np.round(cc[:, None] + off[None, :]).astype(int)
        valid = (cols >= 0) & (cols < ncols)
        cols_cl = np.clip(cols, 0, ncols - 1)
        vals = img[rows[:, None], cols_cl]
        vmask = valid & valid_img[rows[:, None], cols_cl]
        wmat = np.broadcast_to(w, vals.shape) * vmask
        wsum = wmat.sum()
        stripe_mean[k] = float((vals * wmat).sum() / max(wsum, 1e-9))
        cc_int = np.round(cc).astype(int)
        for i, c in enumerate(cc_int):
            c0 = max(0, c - EXCL_HALF)
            c1 = min(ncols, c + EXCL_HALF + 1)
            excl[i, c0:c1] = True
    band = img[rows[:, None], np.arange(ncols)[None, :]]
    band_valid = valid_img[rows[:, None], np.arange(ncols)[None, :]]
    bg_vals = band[(~excl) & band_valid & (band > 0)]
    if bg_vals.size:
        bg_med = float(np.median(bg_vals))
        bg_mad = float(np.median(np.abs(bg_vals - bg_med)))
    else:
        bg_med, bg_mad = 0.0, 1.0
    return stripe_mean, bg_med, max(bg_mad, 1e-6)


def fit_quality(rot_img, res, exp_dx, dx, hp):
    CONTRAST_TARGET = hp.contrast_target
    STRIPE_WEIGHTS = np.asarray(hp.stripe_weights, dtype=float)
    SOFTMIN_TAU = hp.softmin_tau
    SPACING_REL_BOUND = hp.spacing_rel_bound

    p = res["params"]
    nrows, ncols = rot_img.shape
    cr, L = p["center_row"], p["length_px"]
    sigma = max(p["sigma_px"], 0.5)
    r0 = int(max(0, np.floor(cr - L / 2.0)))
    r1 = int(min(nrows - 1, np.ceil(cr + L / 2.0)))
    rows = np.arange(r0, r1 + 1)
    centers_grid = _centers_grid_from_params(p, rows)
    stripe_mean, bg_med, bg_mad = _amp_stripe_stats(rot_img, centers_grid, sigma, rows, ncols)
    bg_sig = max(1.4826 * bg_mad, 1e-6)
    contrast = stripe_mean / max(bg_med, 1e-6)
    snr = (stripe_mean - bg_med) / bg_sig
    sd = perp_sep(res["centerlines"][0], res["centerlines"][1], cr)
    dt = perp_sep(res["centerlines"][1], res["centerlines"][2], cr)
    asym = abs(sd - dt) / max(sd, dt)
    prior_dev = abs(0.5 * (sd + dt) - exp_dx) / exp_dx
    qk = np.clip((contrast - 1.0) / (CONTRAST_TARGET - 1.0), 0.0, 1.0)
    softmin = -SOFTMIN_TAU * np.log(np.sum(STRIPE_WEIGHTS * np.exp(-qk / SOFTMIN_TAU)) /
                                    STRIPE_WEIGHTS.sum())
    q_contrast = float(np.clip(softmin, 0.0, 1.0))
    q_symmetry = float(np.clip(1.0 - asym / 0.25, 0.0, 1.0))
    q_prior = float(np.clip(1.0 - prior_dev / SPACING_REL_BOUND, 0.0, 1.0))
    Q = float(0.4 * q_contrast + 0.35 * q_symmetry + 0.25 * q_prior)
    return {"quality_index": Q, "q_contrast": q_contrast,
            "q_symmetry": q_symmetry, "q_prior": q_prior,
            "contrast_min": float(contrast.min()),
            "stripe_contrast_S": float(contrast[0]),
            "stripe_contrast_D": float(contrast[1]),
            "stripe_contrast_T": float(contrast[2]),
            "bg_median": bg_med, "bg_sig": float(bg_sig),
            "snr_S": float(snr[0]), "snr_D": float(snr[1]), "snr_T": float(snr[2]),
            "sigma_px": float(sigma),
            "spacing_asymmetry": float(asym),
            "spacing_prior_dev": float(prior_dev)}


# ---------------------------------------------------------------------------
# SNR-based spacing uncertainty
# ---------------------------------------------------------------------------
def snr_sigma_spacing(q, dx, hp, snr_floor=None):
    if snr_floor is None:
        snr_floor = hp.snr_floor
    sigma_px = float(q.get("sigma_px", float("nan")))
    snr_S = max(float(q.get("snr_S", 0.0)), snr_floor)
    snr_D = max(float(q.get("snr_D", 0.0)), snr_floor)
    snr_T = max(float(q.get("snr_T", 0.0)), snr_floor)
    sig_cS = sigma_px / snr_S
    sig_cD = sigma_px / snr_D
    sig_cT = sigma_px / snr_T
    sig_sd_px = float(np.sqrt(sig_cS ** 2 + sig_cD ** 2))
    sig_dt_px = float(np.sqrt(sig_cD ** 2 + sig_cT ** 2))
    return sig_sd_px * dx, sig_dt_px * dx


def thickness_corrections(model, T):
    if model == "symmetric":
        return -T, -T, -T
    if model == "lower_triple":
        return -T, 0.0, -0.5 * T
    if model == "lower_single":
        return 0.0, -T, -0.5 * T
    if model == "mixed":
        return 0, 0.0, -T
    raise ValueError(f"unknown scatter_model: {model}")
