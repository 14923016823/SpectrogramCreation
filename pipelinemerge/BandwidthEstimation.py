from __future__ import annotations
import numpy as np

LN2 = np.log(2.0)


def _floor_from_bins(bins: np.ndarray, method: str = "median") -> float:
    """Noise-floor estimate from noise-only bins.
    'median': debiased for single-look exponential bins via /ln2 (estimates mean N0).
    'mean'  : unbiased for exponential but fragile to spurs.
    """
    bins = bins[np.isfinite(bins)]
    if bins.size == 0:
        return np.nan
    if method == "mean":
        return float(np.mean(bins))
    return float(np.median(bins) / LN2)


def _walk_edge(psd, center_idx, step, thr_low, debounce, max_bins):
    """Walk outward from center while PSD stays above thr_low.
    Tolerates up to `debounce` consecutive sub-threshold bins before stopping.
    """
    n, edge, below, moved, i = psd.size, center_idx, 0, 0, center_idx
    while 0 <= i + step < n and moved < max_bins:
        i += step
        moved += 1
        if psd[i] >= thr_low:
            edge, below = i, 0
        else:
            below += 1
            if below >= debounce:
                break
    return edge


def select_strong_frames(power, f_axis, fc, max_halfband_hz, frac=0.15, min_frames=5):
    """Rank frames by coarse in-band SNR proxy; return indices of the top `frac`."""
    df = float(np.median(np.diff(f_axis)))
    proxy = np.full(power.shape[1], -np.inf)
    for k in range(power.shape[1]):
        if not np.isfinite(fc[k]):
            continue
        win = np.abs(f_axis - fc[k]) <= max_halfband_hz
        if not win.any():
            continue
        floor = np.median(power[~win, k]) / LN2 if (~win).any() else 0.0
        proxy[k] = np.sum(power[win, k] - floor) * df
    order = np.argsort(proxy)[::-1]
    order = order[np.isfinite(proxy[order])]
    n = max(min_frames, int(round(frac * order.size)))
    return order[:max(1, min(n, order.size))]


def estimate_band(power, f_axis, fc, frame_idx, max_halfband_hz,
                  alpha_hi_db=9.0, alpha_lo_db=1.0, debounce=3):
    """Carrier-align strong frames, average, walk band edges.
    Returns dict with 'present', 'offset_lo', 'offset_hi' (Hz relative to carrier).
    """
    df = float(np.median(np.diff(f_axis)))
    g = np.arange(-max_halfband_hz, max_halfband_hz + df, df)
    acc, cnt = np.zeros_like(g), 0
    for k in frame_idx:
        if not np.isfinite(fc[k]):
            continue
        acc += np.interp(g, f_axis - fc[k], power[:, k], left=np.nan, right=np.nan)
        cnt += 1
    if cnt == 0:
        raise ValueError("No usable frames for band estimation.")
    bar = acc / cnt
    valid = np.isfinite(bar)

    n0 = float(np.median(bar[valid]))
    thr_hi = n0 * 10 ** (alpha_hi_db / 10)
    thr_lo = n0 * 10 ** (alpha_lo_db / 10)

    z = int(np.argmin(np.abs(g)))
    w = max(1, int(round((max_halfband_hz / 4) / df)))
    lo, hi = max(0, z - w), min(g.size, z + w + 1)
    c = lo + int(np.nanargmax(np.where(valid[lo:hi], bar[lo:hi], -np.inf)))

    if not (bar[c] >= thr_hi):
        return dict(present=False, offset_lo=np.nan, offset_hi=np.nan)

    nmax = int(round(max_halfband_hz / df))
    walk = np.where(valid, bar, -np.inf)
    ei = _walk_edge(walk, c, +1, thr_lo, debounce, nmax)
    ej = _walk_edge(walk, c, -1, thr_lo, debounce, nmax)
    return dict(present=True, offset_lo=float(g[ej]), offset_hi=float(g[ei]))


def get_bandwidth(power, f_axis, fc, max_halfband_hz=2500.0):
    """Public entry point. Returns (offset_lo, offset_hi) tuple or None if estimation fails.
    Pass the result directly to strest as bw=get_bandwidth(...) or bw=fixed_hz as fallback.
    """
    try:
        strong = select_strong_frames(power, f_axis, fc, max_halfband_hz)
        band = estimate_band(power, f_axis, fc, strong, max_halfband_hz)
        if not band["present"]:
            return None
        return (band["offset_lo"], band["offset_hi"])
    except Exception:
        return None