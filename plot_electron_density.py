#!/usr/bin/env python
"""
Plot electron density as a function of altitude and solar longitude (Ls)
from GITM NetCDF files.

Supports two file layouts:
  1. gitm_sza_*_var32_*.nc  – var_32(ls, alt), Ls already present as a dimension
  2. *_reduced.nc           – e_point(time, altitude, mode_point), Ls as a
                              coordinate of the time dimension

If Ls is absent entirely, it is derived from the time coordinate using
marstiming.getMarsSolarGeometry (~/VCProjects/marstiming/src/marstiming.py).

Usage
-----
    python plot_electron_density.py file1.nc file2.nc ... [options]

Options
-------
    --output  / -o   Output filename (default: electron_density.png)
    --mode           mode_point to use from reduced files (default: sza_day)
    --vmin           Colorbar lower bound, actual e-density units (auto if omitted)
    --vmax           Colorbar upper bound, actual e-density units (auto if omitted)
    --cmap           Matplotlib colormap name (default: viridis)
    --linear         Use a linear (instead of log10) colour scale
    --oplotmax       Overplot the altitude of maximum density (parabolic interpolation)
    --smooth N       Smooth the peak altitude line with a Savitzky-Golay filter
                     of window length N degrees (must be odd; requires --oplotmax)
    --title          Optional overall figure title
"""

import sys
import os
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr

# ── make marstiming importable ───────────────────────────────────────────────
_MT_PATH = os.path.expanduser('~/VCProjects/marstiming/src')
if _MT_PATH not in sys.path:
    sys.path.insert(0, _MT_PATH)


# ────────────────────────────────────────────────────────────────────────────
# Data extraction helpers
# ────────────────────────────────────────────────────────────────────────────

def _ls_from_time(time_values):
    """Compute Ls for an array of numpy datetime64 values using marstiming."""
    import marstiming as mt
    import pandas as pd

    ls_out = np.full(len(time_values), np.nan)
    for i, t in enumerate(time_values):
        dt = pd.Timestamp(t).to_pydatetime()
        itime = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
        try:
            ls_out[i] = mt.getMarsSolarGeometry(itime).ls
        except Exception as exc:
            warnings.warn(f"marstiming failed for time {dt}: {exc}")
    return ls_out


def _ls_from_filename(fpath):
    """Try to parse an Ls value from a GITM-style filename
    (e.g. gitm_sza_55_65_var32_19980715_000321.nc) using marstiming."""
    import re, marstiming as mt

    m = re.search(r'(\d{8})_(\d{6})', os.path.basename(fpath))
    if m is None:
        return None
    date_str, time_str = m.group(1), m.group(2)
    itime = [
        int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]),
        int(time_str[0:2]), int(time_str[2:4]), int(time_str[4:6]),
    ]
    try:
        return mt.getMarsSolarGeometry(itime).ls
    except Exception:
        return None


def extract(ds, fpath, mode_point='sza_day'):
    """Return (ne, alt_km, ls_deg) as 2-D / 1-D numpy arrays.

    ne has shape (n_ls, n_alt).
    """

    # ── Layout 1: var_32(ls, alt) ─────────────────────────────────────────
    if 'var_32' in ds.data_vars:
        for ls_name in ['ls', 'Ls', 'LS']:
            if ls_name in ds.coords or ls_name in ds.dims:
                ls = ds[ls_name].values.astype(float)
                break
        else:
            ls = None

        for alt_name in ['alt', 'altitude', 'z', 'height']:
            if alt_name in ds.coords or alt_name in ds.dims:
                alt = ds[alt_name].values.astype(float)
                break
        else:
            raise ValueError("Cannot find altitude coordinate.")

        ne = ds['var_32'].values.astype(float)   # shape (ls, alt)

        if ls is None:
            # Fall back to marstiming via the filename timestamp
            single_ls = _ls_from_filename(fpath)
            if single_ls is not None:
                ls = np.array([single_ls])
                ne = ne[np.newaxis, :]            # shape (1, alt)
            else:
                raise ValueError("Cannot determine Ls for this file.")

        return ne, alt, ls

    # ── Layout 2: e_point(time, altitude, mode_point) ────────────────────
    if 'e_point' in ds.data_vars:
        # Altitude
        for alt_name in ['altitude', 'alt', 'z', 'height']:
            if alt_name in ds.coords or alt_name in ds.dims:
                alt = ds[alt_name].values.astype(float)
                break
        else:
            raise ValueError("Cannot find altitude coordinate.")

        # Ls
        for ls_name in ['Ls', 'ls', 'LS']:
            if ls_name in ds.coords:
                ls = ds[ls_name].values.astype(float)
                break
        else:
            if 'time' in ds.coords:
                print("  Ls not found – computing from time via marstiming …")
                ls = _ls_from_time(ds['time'].values)
            else:
                raise ValueError("Cannot determine Ls for this file.")

        # Select mode_point
        available = list(ds['mode_point'].values)
        if mode_point not in available:
            warnings.warn(
                f"mode_point '{mode_point}' not in {available}; "
                f"using '{available[0]}' instead."
            )
            mode_point = available[0]

        ne = ds['e_point'].sel(mode_point=mode_point).values.astype(float)
        # shape (time, altitude) → already (n_ls, n_alt)
        return ne, alt, ls

    raise ValueError(
        "No recognised electron-density variable found "
        "(expected 'var_32' or 'e_point')."
    )


# ────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ────────────────────────────────────────────────────────────────────────────

def _parabolic_peak_alt(profile, alts):
    """Sub-grid peak altitude via parabolic interpolation.

    Fits a parabola through the grid-point maximum and its two neighbours
    and returns the analytic vertex — identical to the approach used in
    gitm_plot_one_loc.py (-oplotmax).  Falls back to the grid-point altitude
    when the peak is at the boundary or when the denominator is zero.
    """
    i = np.nanargmax(profile)
    if i == 0 or i == len(profile) - 1:
        return alts[i]
    y0, y1, y2 = profile[i - 1], profile[i], profile[i + 1]
    x0, x1, x2 = alts[i - 1],   alts[i],    alts[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return alts[i]
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    return -b / (2 * a)


def peak_altitude(ne, alt):
    """Parabolic-interpolated altitude of maximum electron density per Ls step.

    Returns array of shape (n_ls,), NaN where the profile is all NaN/non-positive.
    """
    peak = np.full(ne.shape[0], np.nan)
    for i in range(ne.shape[0]):
        col = ne[i, :]
        valid = np.isfinite(col) & (col > 0)
        if valid.any():
            peak[i] = _parabolic_peak_alt(np.where(valid, col, np.nan), alt)
    return peak


def savgol_smooth_circular(y, window, polyorder=3):
    """Savitzky-Golay filter with circular (wrap-around) padding.

    The Ls axis is periodic (0° ≡ 360°), so we pad both ends of the array
    with data from the opposite end before filtering, then trim.  This
    eliminates edge artefacts at the start/end of the Mars year.

    Parameters
    ----------
    y         : 1-D array of values to smooth (NaNs are tolerated; they
                are linearly interpolated over before filtering, then
                restored afterwards)
    window    : int – window length in samples; forced odd if even
    polyorder : int – polynomial order (default 3)

    Returns
    -------
    smoothed  : 1-D array, same shape as y
    """
    from scipy.signal import savgol_filter

    if window % 2 == 0:
        window += 1          # SG requires an odd window

    n    = len(y)
    half = window // 2

    # Interpolate over NaNs so the filter doesn't propagate them
    nan_mask = ~np.isfinite(y)
    y_filled = y.copy()
    if nan_mask.any():
        x = np.arange(n)
        y_filled[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])

    # Circular pad
    padded   = np.concatenate([y_filled[-half:], y_filled, y_filled[:half]])
    smoothed = savgol_filter(padded, window, polyorder)[half: half + n]

    # Restore original NaNs
    smoothed[nan_mask] = np.nan
    return smoothed


def file_stats(rec, pk=None):
    """Compute summary statistics for one record.

    Parameters
    ----------
    rec : dict  – record produced by the load loop
    pk  : 1-D array, optional – pre-computed (and optionally smoothed) peak
          altitude values.  If None the raw parabolic peak is computed here.

    Returns a dict with:
      ne_min, ne_max, ne_mean      – density stats (positive values only)
      peak_alt_min, peak_alt_max, peak_alt_mean – peak-altitude stats across all Ls steps
      ls_at_peak_alt_max           – Ls where the peak altitude is highest
      ls_at_peak_alt_min           – Ls where the peak altitude is lowest
      ls_at_peak_ne_max            – Ls where the peak density is largest
      ls_at_peak_ne_min            – Ls where the peak density is smallest
      alt_min, alt_max             – altitude range of the data
    """
    ne  = rec['ne']
    alt = rec['alt']
    ls  = rec['ls']

    pos = ne[ne > 0]
    ne_min  = float(np.nanmin(pos))  if pos.size else np.nan
    ne_max  = float(np.nanmax(pos))  if pos.size else np.nan
    ne_mean = float(np.nanmean(pos)) if pos.size else np.nan

    if pk is None:
        pk = peak_altitude(ne, alt)      # parabolic interpolation, shape (n_ls,)
    valid_pk = pk[np.isfinite(pk)]

    pk_min  = float(np.nanmin(valid_pk))  if valid_pk.size else np.nan
    pk_max  = float(np.nanmax(valid_pk))  if valid_pk.size else np.nan
    pk_mean = float(np.nanmean(valid_pk)) if valid_pk.size else np.nan

    # Ls where peak altitude is greatest / smallest
    if valid_pk.size:
        ls_at_peak_alt_max = float(ls[np.nanargmax(pk)])
        ls_at_peak_alt_min = float(ls[np.nanargmin(pk)])
    else:
        ls_at_peak_alt_max = np.nan
        ls_at_peak_alt_min = np.nan

    # Peak density: maximum ne value in each Ls column
    peak_ne = np.array([
        float(np.nanmax(row[row > 0])) if np.any(row > 0) else np.nan
        for row in ne
    ])
    valid_pne = peak_ne[np.isfinite(peak_ne)]
    peak_ne_min  = float(np.nanmin(valid_pne))  if valid_pne.size else np.nan
    peak_ne_max  = float(np.nanmax(valid_pne))  if valid_pne.size else np.nan
    peak_ne_mean = float(np.nanmean(valid_pne)) if valid_pne.size else np.nan

    # Ls where column-peak density is largest / smallest
    if valid_pne.size:
        ls_at_peak_ne_max = float(ls[np.nanargmax(peak_ne)])
        ls_at_peak_ne_min = float(ls[np.nanargmin(peak_ne)])
    else:
        ls_at_peak_ne_max = np.nan
        ls_at_peak_ne_min = np.nan

    return dict(
        ne_min=ne_min, ne_max=ne_max, ne_mean=ne_mean,
        peak_alt_min=pk_min, peak_alt_max=pk_max, peak_alt_mean=pk_mean,
        ls_at_peak_alt_max=ls_at_peak_alt_max,
        ls_at_peak_alt_min=ls_at_peak_alt_min,
        peak_ne_min=peak_ne_min, peak_ne_max=peak_ne_max, peak_ne_mean=peak_ne_mean,
        ls_at_peak_ne_max=ls_at_peak_ne_max,
        ls_at_peak_ne_min=ls_at_peak_ne_min,
        alt_min=float(alt.min()), alt_max=float(alt.max()),
    )


def format_stats_block(label, s):
    """Format a stats dict as a readable text block."""
    lines = [
        f"{'─' * 60}",
        f"  {label}",
        f"{'─' * 60}",
        f"  Altitude range (plotted) : {s['alt_min']:.1f} – {s['alt_max']:.1f} km",
        f"",
        f"  Electron density (m⁻³)",
        f"    min  : {s['ne_min']:.4e}",
        f"    max  : {s['ne_max']:.4e}",
        f"    mean : {s['ne_mean']:.4e}",
        f"",
        f"  Peak altitude (km)",
        f"    mean : {s['peak_alt_mean']:.2f}",
        f"    max  : {s['peak_alt_max']:.2f}  (Ls = {s['ls_at_peak_alt_max']:.1f}°)",
        f"    min  : {s['peak_alt_min']:.2f}  (Ls = {s['ls_at_peak_alt_min']:.1f}°)",
        f"",
        f"  Peak density (m⁻³)",
        f"    mean : {s['peak_ne_mean']:.4e}",
        f"    max  : {s['peak_ne_max']:.4e}  (Ls = {s['ls_at_peak_ne_max']:.1f}°)",
        f"    min  : {s['peak_ne_min']:.4e}  (Ls = {s['ls_at_peak_ne_min']:.1f}°)",
    ]
    return '\n'.join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot electron density vs altitude and Ls.'
    )
    parser.add_argument('files', nargs='+', help='Input .nc files')
    parser.add_argument('--output', '-o', default='electron_density.png',
                        help='Output figure filename (default: electron_density.png)')
    parser.add_argument('--mode', default='sza_day',
                        help='mode_point for reduced files (default: sza_day)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Colorbar lower bound (actual density units)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Colorbar upper bound (actual density units)')
    parser.add_argument('--cmap', default='viridis',
                        help='Matplotlib colormap (default: viridis)')
    parser.add_argument('--linear', action='store_true',
                        help='Use a linear colour scale (default is log10)')
    parser.add_argument('--oplotmax', action='store_true',
                        help='Overplot altitude of peak density (parabolic interpolation)')
    parser.add_argument('--smooth', type=int, default=-1,
                        help='Savitzky-Golay window length (degrees) applied to the '
                             'peak altitude line (odd integer; requires --oplotmax)')
    parser.add_argument('--minalt', type=float, default=None,
                        help='Minimum altitude to plot in km')
    parser.add_argument('--maxalt', type=float, default=None,
                        help='Maximum altitude to plot in km')
    parser.add_argument('--title', default=None,
                        help='Overall figure title')
    parser.add_argument('--titles', default=None,
                        help='Comma- or space-separated list of subplot titles '
                             '(one per file, in order). Wrap in quotes if using spaces.')
    parser.add_argument('--data', action='store_true',
                        help='Print density and peak-altitude statistics to stdout '
                             'and save to a .txt file alongside the output figure.')
    args = parser.parse_args()

    if args.smooth > 0 and not args.oplotmax:
        parser.error('--smooth requires --oplotmax (smoothing is applied only to the peak altitude line)')

    use_log = not args.linear

    # ── Parse subplot titles ──────────────────────────────────────────────
    subplot_titles = None
    if args.titles:
        # split on commas first; fall back to whitespace if no commas present
        if ',' in args.titles:
            subplot_titles = [t.strip() for t in args.titles.split(',')]
        else:
            subplot_titles = args.titles.split()

    smoothing = args.smooth > 0

    # ── Load all datasets ─────────────────────────────────────────────────
    records = []
    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"WARNING: file not found – skipping: {fpath}")
            continue
        print(f"Loading {fpath} …")
        try:
            ds = xr.open_dataset(fpath)
            ne, alt, ls = extract(ds, fpath, mode_point=args.mode)
            ds.close()

            # ── Altitude trimming ─────────────────────────────────────────
            alt_mask = np.ones(len(alt), dtype=bool)
            if args.minalt is not None:
                alt_mask &= alt >= args.minalt
            if args.maxalt is not None:
                alt_mask &= alt <= args.maxalt
            if not alt_mask.all():
                alt = alt[alt_mask]
                ne  = ne[:, alt_mask]

            label = os.path.splitext(os.path.basename(fpath))[0]
            records.append(dict(ne=ne, alt=alt, ls=ls, label=label))
            print(f"  OK  ne shape={ne.shape}, "
                  f"alt=[{alt.min():.0f}–{alt.max():.0f}] km, "
                  f"Ls=[{ls.min():.1f}–{ls.max():.1f}]°")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not records:
        print("No valid data loaded – exiting.")
        sys.exit(1)

    # ── Determine global colour bounds ────────────────────────────────────
    positive_vals = np.concatenate([
        r['ne'][r['ne'] > 0].ravel() for r in records
        if np.any(r['ne'] > 0)
    ])

    if args.vmin is not None:
        vmin = args.vmin
    else:
        vmin = np.nanpercentile(positive_vals, 2)

    if args.vmax is not None:
        vmax = args.vmax
    else:
        vmax = np.nanpercentile(positive_vals, 98)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if use_log \
        else mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Font sizes ────────────────────────────────────────────────────────
    FS_TITLE  = 18   # in-plot panel labels
    FS_LABEL  = 18   # axis labels
    FS_TICK   = 16   # tick labels
    FS_SUPTITLE = 20 # overall figure title
    FS_CBAR   = 16   # colourbar label / ticks

    # ── Build figure ──────────────────────────────────────────────────────
    n = len(records)
    fig_h = max(3.2 * n + 0.5, 4)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), squeeze=False)

    if args.title:
        fig.suptitle(args.title, fontsize=FS_SUPTITLE, y=1.01)

    last_im = None

    for i, rec in enumerate(records):
        ax = axes[i, 0]
        ne  = rec['ne']
        alt = rec['alt']
        ls  = rec['ls']

        # mask non-positive values for log scale
        plot_data = np.where(ne > 0, ne, np.nan)

        last_im = ax.pcolormesh(
            ls, alt, plot_data.T,
            norm=norm, cmap=args.cmap, shading='auto'
        )

        # Peak-density altitude line (parabolic interpolation ± SG smoothing)
        if args.oplotmax:
            pk = peak_altitude(ne, alt)
            if smoothing:
                pk = savgol_smooth_circular(pk, args.smooth)
            ax.plot(ls, pk, color='black', linewidth=1.5, label='Peak altitude')

        if i == n // 2:
            ax.set_ylabel('Altitude (km)', fontsize=FS_LABEL)
        ax.set_xlim(ls.min(), ls.max())
        alt_lo = args.minalt if args.minalt is not None else alt.min()
        alt_hi = args.maxalt if args.maxalt is not None else alt.max()
        ax.set_ylim(alt_lo, alt_hi)

        # in-plot label: user-supplied list takes priority, then filename
        if subplot_titles is not None and i < len(subplot_titles):
            panel_title = subplot_titles[i]
        else:
            panel_title = rec['label']
        ax.text(0.02, 0.97, panel_title, transform=ax.transAxes,
                fontsize=FS_TITLE, color='white', va='top', ha='left')

        ax.set_xticks(np.arange(0, 361, 30))
        ax.tick_params(labelsize=FS_TICK)

        # Only put x-label on bottom panel
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('L$_s$ (°)', fontsize=FS_LABEL)

    # ── Single shared colourbar ───────────────────────────────────────────
    fig.subplots_adjust(right=0.86, hspace=0.15)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
    cb = fig.colorbar(last_im, cax=cbar_ax, extend='both')
    cb_label = ('log$_{10}$ ' if use_log else '') + 'Electron Density (m$^{-3}$)'
    cb.set_label(cb_label, fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_CBAR)

    # ── Statistics ────────────────────────────────────────────────────────
    if args.data:
        stat_lines = []
        for i, rec in enumerate(records):
            label = subplot_titles[i] if (subplot_titles and i < len(subplot_titles)) \
                    else rec['label']
            # Use the same peak array that was plotted (smoothed if requested)
            pk_for_stats = None
            if args.oplotmax:
                pk_for_stats = peak_altitude(rec['ne'], rec['alt'])
                if smoothing:
                    pk_for_stats = savgol_smooth_circular(pk_for_stats, args.smooth)
            s = file_stats(rec, pk=pk_for_stats)
            block = format_stats_block(label, s)
            print(block)
            stat_lines.append(block)

        txt_path = os.path.splitext(args.output)[0] + '_stats.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(stat_lines) + '\n')
        print(f"\nStatistics saved → {txt_path}")

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {args.output}")


if __name__ == '__main__':
    main()
