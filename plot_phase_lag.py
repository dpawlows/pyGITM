#!/usr/bin/env python3
"""
Compute the thermal phase lag between perihelion/aphelion and the subsequent
variable maximum/minimum.

Algorithm:
  1. Compute Mars-Sun distance at every time step to locate perihelion and aphelion.
  2. Apply a running average over a window of -window deg Ls (default 15 deg).
  3. Evaluate the smoothed variable at perihelion and aphelion.
  4. Find the maximum of the smoothed variable after perihelion and the minimum
     after aphelion.
  5. Report the Ls phase lag and absolute difference for each event.

Usage:
    plot_phase_lag.py case1/case1_reduced.nc -var Temperature -mode subsolar -alt 200
    plot_phase_lag.py case1/case1_reduced.nc -var Temperature Rho -mode global -alt 200 135 -window 20
"""

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from marstiming import getMarsSolarGeometry
from gitm_routines import get_units

plt.rcParams.update({'lines.linewidth': 1.0})
cm = 1 / 2.54
COL_WIDTH_CM = {"single": 8, "double": 16}
BASE_FONT = {"label": 9, "tick": 8, "legend": 8, "text": 8}

# Mars orbital elements
MARS_A = 1.524   # semi-major axis (AU)
MARS_E = 0.0934  # eccentricity

VAR_LABELS = {
    "CO2":         r"[CO$_2$]",
    "O":           r"[O]",
    "Rho":         r"$\rho$",
    "Temperature": r"T",
    "Vneast":      r"u",
    "Vnnorth":     r"v",
    "Vnup":        r"w",
    "e":           r"[e-]",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def mars_sun_distance(M, vMinusM):
    """Compute Mars-Sun distance in AU from mean anomaly and equation of centre."""
    v_rad = np.radians(M + vMinusM)
    return MARS_A * (1.0 - MARS_E**2) / (1.0 + MARS_E * np.cos(v_rad))


def decode_char_var(var):
    """Convert a netCDF4 string/char variable to a list of stripped strings."""
    data = var[:]
    if data.dtype.kind in ("S", "U"):
        return [str(s).strip() for s in data]
    if data.ndim == 2:
        return ["".join(row.astype(str)).strip() for row in data]
    return [str(s).strip() for s in data]


def var_label(varname):
    """Return 'display_name (units)' label for axis annotations."""
    display = VAR_LABELS.get(varname, varname)
    units = get_units(varname)
    return f"{display} ({units})" if units else display


def running_average_ls(data, Ls, window):
    """
    Apply a centred running average in Ls-space.

    For each point i, averages all points j where
        |Ls[j] - Ls[i]| <= window / 2.

    Parameters
    ----------
    data   : 1-D array of shape (n,)
    Ls     : 1-D unwrapped Ls array of shape (n,)
    window : total window width in degrees Ls

    Returns
    -------
    smoothed : 1-D array of the same shape
    """
    half = window / 2.0
    smoothed = np.full(len(data), np.nan, dtype=float)
    for i in range(len(data)):
        dist = np.abs(Ls - Ls[i])
        mask = dist <= half
        vals = data[mask]
        if np.any(np.isfinite(vals)):
            smoothed[i] = np.nanmean(vals)
    return smoothed


def find_extremum_after(smoothed, Ls, i_ref, find_max):
    """
    Find the index and Ls phase lag of the maximum (find_max=True) or
    minimum (find_max=False) in *smoothed* at indices > i_ref.

    Returns (i_ext, delta_ls) or (None, None) when no data follow i_ref.
    """
    if i_ref >= len(smoothed) - 1:
        return None, None
    after = smoothed[i_ref + 1:]
    if len(after) == 0 or not np.any(np.isfinite(after)):
        return None, None
    if find_max:
        i_ext = int(np.nanargmax(after)) + i_ref + 1
    else:
        i_ext = int(np.nanargmin(after)) + i_ref + 1
    delta_ls = Ls[i_ext] - Ls[i_ref]
    return i_ext, delta_ls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute perihelion/aphelion phase lag for one or more variables"
    )
    parser.add_argument("filename",
                        help="Reduced NetCDF file (e.g. case1/case1_reduced.nc)")
    parser.add_argument("-var", nargs="+", required=True,
                        help="Variable name(s) to analyse")
    parser.add_argument("-mode", required=True,
                        help="Mode string (e.g. global, lt14, subsolar)")
    parser.add_argument("-alt", type=float, nargs="+", required=True,
                        help="Altitude(s) in km")
    parser.add_argument("-window", type=float, default=15.0,
                        help="Smoothing window in degrees Ls (default 15; centred ±7.5 deg)")
    parser.add_argument("-data", action="store_true",
                        help="Write phase-lag results to a text file")
    parser.add_argument("-show", action="store_true",
                        help="Display plots interactively")
    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        sys.exit(f"Error: file not found: {args.filename}")

    # ------------------------------------------------------------------
    # Open NetCDF and read coordinates
    # ------------------------------------------------------------------
    ds = nc.Dataset(args.filename)
    time_var  = ds.variables["time"]
    raw_times = nc.num2date(time_var[:], units=time_var.units,
                            calendar=getattr(time_var, "calendar", "standard"))
    dates = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
             for t in raw_times]
    Ls_raw     = np.array(ds.variables["Ls"][:])
    alt_values = np.array(ds.variables["altitude"][:])
    lat_modes   = decode_char_var(ds.variables["mode_lat"])
    point_modes = decode_char_var(ds.variables["mode_point"])

    coord_names = {"time", "altitude", "latitude", "mode_lat", "mode_point",
                   "Ls", "nfiles", "average", "year", "sol"}
    available_vars = sorted({
        name[:-6] if name.endswith("_point") else name
        for name in ds.variables
        if name not in coord_names
    })

    # Validate inputs
    if args.mode not in lat_modes and args.mode not in point_modes:
        ds.close()
        sys.exit(f"Error: mode '{args.mode}' not found. "
                 f"Available: {lat_modes + point_modes}")

    bad_alts = [a for a in args.alt if a not in alt_values]
    if bad_alts:
        ds.close()
        sys.exit(f"Error: altitude(s) not found: {bad_alts}. Available: {list(alt_values)}")

    bad_vars = [v for v in args.var if v not in available_vars]
    if bad_vars:
        ds.close()
        sys.exit(f"Error: variable(s) not found: {bad_vars}. "
                 f"Available: {', '.join(available_vars)}")

    # Unwrap Ls across year boundaries
    Ls = Ls_raw.copy()
    for w in np.where(np.diff(Ls) < -180)[0]:
        Ls[w + 1:] += 360

    # ------------------------------------------------------------------
    # Compute Mars-Sun distance → perihelion and aphelion
    # ------------------------------------------------------------------
    print(f"Computing Mars-Sun distance for {len(dates)} time steps …")
    distances = []
    for dt in dates:
        geo = getMarsSolarGeometry(dt)
        distances.append(mars_sun_distance(geo.M, geo.vMinusM))
    distances = np.array(distances)

    i_peri = int(np.argmin(distances))
    i_aph  = int(np.argmax(distances))

    ls_peri_str = f"{Ls_raw[i_peri]:.1f}"
    ls_aph_str  = f"{Ls_raw[i_aph]:.1f}"

    print(f"\nPerihelion (minimum distance within dataset):")
    print(f"  Date : {dates[i_peri].strftime('%Y-%m-%d')}")
    print(f"  Ls   : {ls_peri_str} deg")
    print(f"  r    : {distances[i_peri]:.4f} AU")

    print(f"\nAphelion (maximum distance within dataset):")
    print(f"  Date : {dates[i_aph].strftime('%Y-%m-%d')}")
    print(f"  Ls   : {ls_aph_str} deg")
    print(f"  r    : {distances[i_aph]:.4f} AU")

    half_win = args.window / 2.0
    print(f"\nSmoothing window: {args.window:.1f} deg Ls (±{half_win:.1f} deg)")

    # ------------------------------------------------------------------
    # Process each variable × altitude
    # ------------------------------------------------------------------
    casetag = os.path.splitext(os.path.basename(args.filename))[0].replace("_reduced", "")

    # Storage for optional text output:  results[varname][alt] = dict
    results = {v: {} for v in args.var}

    for varname in args.var:
        units    = get_units(varname)
        unit_str = f" [{units}]" if units else ""

        print(f"\n{'='*62}")
        print(f"  Variable : {varname}{unit_str}")
        print(f"  Mode     : {args.mode}")
        print(f"{'='*62}")

        for alt in args.alt:
            alt_idx = int(np.where(alt_values == alt)[0][0])

            # Extract 1-D time series at this mode and altitude
            if args.mode in lat_modes:
                mode_idx  = lat_modes.index(args.mode)
                data_2d   = np.array(ds.variables[varname][:, :, alt_idx, mode_idx])
                data      = np.nanmean(data_2d, axis=1)   # zonal (lat) mean
            else:
                mode_idx  = point_modes.index(args.mode)
                data      = np.array(ds.variables[varname + "_point"][:, alt_idx, mode_idx])

            # Running average
            smoothed = running_average_ls(data, Ls, args.window)

            # Smoothed values at peri / aph
            val_peri = float(smoothed[i_peri])
            val_aph  = float(smoothed[i_aph])

            # Post-perihelion maximum
            i_max, dls_peri = find_extremum_after(smoothed, Ls, i_peri, find_max=True)
            # Post-aphelion minimum
            i_min, dls_aph  = find_extremum_after(smoothed, Ls, i_aph,  find_max=False)

            print(f"\n  Altitude : {int(alt)} km")
            print(f"  {'Event':<30}  {'Ls (deg)':>8}  {'Value':>14}  {'ΔLs (deg)':>10}  {'Δval':>14}")
            print(f"  {'-'*30}  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*14}")
            print(f"  {'Perihelion':<30}  {Ls_raw[i_peri]:>8.1f}  {val_peri:>14.4g}")

            if i_max is not None:
                val_max = float(smoothed[i_max])
                ls_max  = Ls[i_max] % 360
                print(f"  {'Post-peri maximum':<30}  {ls_max:>8.1f}  {val_max:>14.4g}"
                      f"  {dls_peri:>10.1f}  {val_max - val_peri:>14.4g}")
            else:
                print(f"  {'Post-peri maximum':<30}  {'—':>8}  {'—':>14}  {'—':>10}  {'—':>14}")
                print(f"    (no data after perihelion)")

            print(f"  {'Aphelion':<30}  {Ls_raw[i_aph]:>8.1f}  {val_aph:>14.4g}")

            if i_min is not None:
                val_min = float(smoothed[i_min])
                ls_min  = Ls[i_min] % 360
                print(f"  {'Post-aph minimum':<30}  {ls_min:>8.1f}  {val_min:>14.4g}"
                      f"  {dls_aph:>10.1f}  {val_min - val_aph:>14.4g}")
            else:
                print(f"  {'Post-aph minimum':<30}  {'—':>8}  {'—':>14}  {'—':>10}  {'—':>14}")
                print(f"    (no data after aphelion)")

            # Store for text output and plotting
            results[varname][alt] = dict(
                data=data, smoothed=smoothed,
                val_peri=val_peri, val_aph=val_aph,
                i_peri=i_peri, i_aph=i_aph,
                i_max=i_max, dls_peri=dls_peri,
                i_min=i_min, dls_aph=dls_aph,
            )

    ds.close()

    # ------------------------------------------------------------------
    # Plot: one column, one subplot per (variable × altitude)
    # Variables in order, each variable's altitudes sorted highest first
    # ------------------------------------------------------------------
    alts_sorted = sorted(args.alt, reverse=True)
    n_rows = len(args.var) * len(alts_sorted)
    row_height_cm = 4.0
    font = BASE_FONT

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(COL_WIDTH_CM["single"] * cm, row_height_cm * n_rows * cm),
        sharex=True, constrained_layout=True,
        squeeze=False
    )
    axes = axes.ravel()

    row = 0
    for varname in args.var:
        for alt in alts_sorted:
            ax  = axes[row]
            res = results[varname][alt]

            ax.plot(Ls % 360, res["data"],
                    color="0.7", linewidth=0.6, label="raw")
            ax.plot(Ls % 360, res["smoothed"],
                    color="k",   linewidth=1.2, label=f"{args.window:.0f}° avg")

            ax.axvline(Ls_raw[i_peri], color="tab:red",  linestyle="--",
                       linewidth=0.8, label=f"peri Ls={ls_peri_str}°")
            ax.axvline(Ls_raw[i_aph],  color="tab:blue", linestyle="--",
                       linewidth=0.8, label=f"aph Ls={ls_aph_str}°")

            if res["i_max"] is not None:
                ax.axvline(Ls[res["i_max"]] % 360, color="tab:red", linestyle=":",
                           linewidth=0.8,
                           label=f"max Ls={Ls[res['i_max']] % 360:.1f}° (Δ{res['dls_peri']:.1f}°)")

            if res["i_min"] is not None:
                ax.axvline(Ls[res["i_min"]] % 360, color="tab:blue", linestyle=":",
                           linewidth=0.8,
                           label=f"min Ls={Ls[res['i_min']] % 360:.1f}° (Δ{res['dls_aph']:.1f}°)")

            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}")
            )
            ax.tick_params(labelsize=font["tick"])
            ax.set_ylabel(var_label(varname), fontsize=font["label"])
            ax.text(0.02, 0.95, f"{int(alt)} km",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=font["text"])
            ax.grid(True, alpha=0.3)
            row += 1

    axes[-1].set_xlabel("Solar Longitude (deg)", fontsize=font["label"])

    var_str = "_".join(args.var)
    alt_str = "_".join(str(int(a)) for a in args.alt)
    outfile = f"phase_lag_{var_str}_{casetag}_{args.mode}_alt{alt_str}.png"
    plt.savefig(outfile, dpi=150)
    print(f"\n  Plot saved: {outfile}")
    if args.show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # Optional text output
    # ------------------------------------------------------------------
    if args.data:
        alt_str  = "_".join(str(int(a)) for a in args.alt)
        var_str  = "_".join(args.var)
        datafile = f"phase_lag_{var_str}_{casetag}_{args.mode}_alt{alt_str}.txt"
        with open(datafile, "w") as fh:
            fh.write(f"# Phase lag analysis: {casetag}, mode={args.mode}\n")
            fh.write(f"# Smoothing window: {args.window:.1f} deg Ls\n")
            fh.write(f"# Perihelion: {dates[i_peri].strftime('%Y-%m-%d')}"
                     f"  Ls={Ls_raw[i_peri]:.1f} deg\n")
            fh.write(f"# Aphelion  : {dates[i_aph].strftime('%Y-%m-%d')}"
                     f"  Ls={Ls_raw[i_aph]:.1f} deg\n")
            fh.write("#\n")
            fh.write(f"# {'var':<14} {'alt_km':>7}  "
                     f"{'val_peri':>14} {'val_aph':>14}  "
                     f"{'ls_max':>8} {'dls_peri':>10} {'dval_peri':>14}  "
                     f"{'ls_min':>8} {'dls_aph':>10} {'dval_aph':>14}\n")
            for varname in args.var:
                for alt in args.alt:
                    res = results[varname][alt]
                    vp  = res["val_peri"]
                    va  = res["val_aph"]
                    if res["i_max"] is not None:
                        ls_max_val = Ls[res["i_max"]] % 360
                        dls_p      = res["dls_peri"]
                        dval_p     = float(res["smoothed"][res["i_max"]]) - vp
                    else:
                        ls_max_val = dls_p = dval_p = float("nan")
                    if res["i_min"] is not None:
                        ls_min_val = Ls[res["i_min"]] % 360
                        dls_a      = res["dls_aph"]
                        dval_a     = float(res["smoothed"][res["i_min"]]) - va
                    else:
                        ls_min_val = dls_a = dval_a = float("nan")
                    fh.write(
                        f"  {varname:<14} {int(alt):>7}  "
                        f"{vp:>14.6e} {va:>14.6e}  "
                        f"{ls_max_val:>8.2f} {dls_p:>10.2f} {dval_p:>14.6e}  "
                        f"{ls_min_val:>8.2f} {dls_a:>10.2f} {dval_a:>14.6e}\n"
                    )
        print(f"\nResults saved: {datafile}")


if __name__ == "__main__":
    main()
