#!/usr/bin/env python3
"""
Compute the Mars-Sun distance for each time step in a reduced NC file,
identify perihelion and aphelion within the dataset, report distances
and their ratio, and optionally plot a variable's perihelion/aphelion
ratio vs altitude.

Usage:
    calc_perihelion_aphelion.py case1/case1_reduced.nc
    calc_perihelion_aphelion.py case1/case1_reduced.nc -var Temperature -mode global
"""

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from marstiming import getMarsSolarGeometry
from gitm_routines import get_units

plt.rcParams.update({'lines.linewidth': 1.0})

cm = 1 / 2.54
COL_WIDTH_CM = {"single": 8, "double": 16}
BASE_FONT    = {"label": 9, "tick": 8, "legend": 8, "text": 8}

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


def mars_sun_distance(M, vMinusM):
    """
    Compute Mars-Sun distance in AU.
    M        : mean anomaly (degrees)
    vMinusM  : equation of center = true anomaly - mean anomaly (degrees)
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Find perihelion/aphelion and optionally plot variable ratio vs altitude"
    )
    parser.add_argument("filename", help="Reduced NetCDF file (e.g. case1/case1_reduced.nc)")
    parser.add_argument("-var",  default=None, nargs="+", help="Variable name(s) to plot ratio for")
    parser.add_argument("-mode", default=None, help="Mode string (e.g. global, lt14, subsolar)")
    parser.add_argument("-avg",  type=int, default=None, metavar="LS",
                        help="Average over ±LS degrees around perihelion/aphelion instead of a single time step")
    parser.add_argument("-show", action="store_true", help="Display plot interactively")
    parser.add_argument("-data", action="store_true",
                        help="Write perihelion/aphelion ratios to a text file")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        sys.exit(f"Error: file not found: {args.filename}")

    ds = nc.Dataset(args.filename)
    time_var   = ds.variables["time"]
    raw_times  = nc.num2date(time_var[:], units=time_var.units,
                             calendar=getattr(time_var, "calendar", "standard"))
    dates      = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                  for t in raw_times]
    Ls         = np.array(ds.variables["Ls"][:])
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

    # ------------------------------------------------------------------
    # Compute Mars-Sun distance for every time step
    # ------------------------------------------------------------------
    print(f"Computing Mars-Sun distance for {len(dates)} time steps...")
    distances = []
    for dt in dates:
        geo = getMarsSolarGeometry(dt)
        distances.append(mars_sun_distance(geo.M, geo.vMinusM))
    distances = np.array(distances)

    i_peri = int(np.argmin(distances))
    i_aph  = int(np.argmax(distances))
    r_peri = distances[i_peri]
    r_aph  = distances[i_aph]

    print(f"\nPerihelion (within dataset):")
    print(f"  Date : {dates[i_peri].strftime('%Y-%m-%d')}")
    print(f"  Ls   : {Ls[i_peri]:.1f} deg")
    print(f"  r    : {r_peri:.4f} AU")

    print(f"\nAphelion (within dataset):")
    print(f"  Date : {dates[i_aph].strftime('%Y-%m-%d')}")
    print(f"  Ls   : {Ls[i_aph]:.1f} deg")
    print(f"  r    : {r_aph:.4f} AU")

    print(f"\nAphelion / perihelion distance ratio : {r_aph / r_peri:.4f}")
    print(f"Perihelion / aphelion flux ratio     : {(r_aph / r_peri)**2:.4f}")

    # ------------------------------------------------------------------
    # Variable ratio plot (only if -var and -mode are given)
    # ------------------------------------------------------------------
    if args.var is None and args.mode is None:
        ds.close()
        return

    if args.var is None or args.mode is None:
        print("\nError: both -var and -mode are required for the ratio plot.")
        print(f"Available variables: {', '.join(available_vars)}")
        ds.close()
        sys.exit(1)

    mode = args.mode
    if mode not in lat_modes and mode not in point_modes:
        ds.close()
        sys.exit(f"Error: mode '{mode}' not found. Available: {lat_modes + point_modes}")

    bad_vars = [v for v in args.var if v not in available_vars]
    if bad_vars:
        print(f"Error: unrecognised variable(s): {', '.join(bad_vars)}")
        print(f"Available variables: {', '.join(available_vars)}")
        ds.close()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build index masks for perihelion and aphelion windows
    # ------------------------------------------------------------------
    avg_ls = args.avg
    if avg_ls is not None:
        ls_peri = Ls[i_peri]
        ls_aph  = Ls[i_aph]
        # Angular distance on a circle
        def ls_dist(a, b):
            return np.abs(((a - b + 180) % 360) - 180)
        mask_peri = ls_dist(Ls, ls_peri) <= avg_ls
        mask_aph  = ls_dist(Ls, ls_aph)  <= avg_ls
        n_peri = mask_peri.sum()
        n_aph  = mask_aph.sum()
        print(f"\nAveraging over ±{avg_ls}° Ls: "
              f"{n_peri} steps near perihelion, {n_aph} steps near aphelion")
    else:
        mask_peri = i_peri
        mask_aph  = i_aph

    # ------------------------------------------------------------------
    # Extract ratio profile for each variable
    # ------------------------------------------------------------------
    def extract_profile(data_arr, mask):
        """Return mean profile over mask (index or boolean array)."""
        if isinstance(mask, (int, np.integer)):
            return data_arr[mask]
        return np.nanmean(data_arr[mask], axis=0)

    ratios   = {}
    profiles = {}  # stores (profile_peri, profile_aph) per variable
    for varname in args.var:
        if mode in lat_modes:
            mode_idx = lat_modes.index(mode)
            data = np.array(ds.variables[varname][:, :, :, mode_idx])  # (time, lat, alt)
            profile_peri = np.nanmean(extract_profile(data, mask_peri), axis=0)
            profile_aph  = np.nanmean(extract_profile(data, mask_aph),  axis=0)
        else:
            mode_idx = point_modes.index(mode)
            data = np.array(ds.variables[varname + "_point"][:, :, mode_idx])  # (time, alt)
            profile_peri = extract_profile(data, mask_peri)
            profile_aph  = extract_profile(data, mask_aph)
        ratios[varname]   = profile_peri / profile_aph
        profiles[varname] = (profile_peri, profile_aph)

    ds.close()

    # ------------------------------------------------------------------
    # Write ratios to text file
    # ------------------------------------------------------------------
    casetag = os.path.splitext(os.path.basename(args.filename))[0].replace("_reduced", "")

    if args.data:
        avg_str = f"±{avg_ls}°" if avg_ls is not None else "single step"
        datafile = f"peri_aph_ratio_{casetag}_{mode}.txt"
        with open(datafile, "w") as fh:
            fh.write(f"# Perihelion/aphelion ratios: {casetag}, mode={mode}\n")
            fh.write(f"# Perihelion Ls={Ls[i_peri]:.1f} deg, "
                     f"Aphelion Ls={Ls[i_aph]:.1f} deg  ({avg_str})\n")
            # Header: alt | for each var: peri, aph, ratio
            col_heads = "".join(
                f" {v+'_peri':>14} {v+'_aph':>14} {v+'_ratio':>14} {v+'_amp':>14}"
                for v in args.var
            )
            fh.write(f"# {'alt_km':>8}{col_heads}\n")
            for j, alt in enumerate(alt_values):
                cols = "".join(
                    f" {profiles[v][0][j]:>14.6e}"
                    f" {profiles[v][1][j]:>14.6e}"
                    f" {ratios[v][j]:>14.6f}"
                    f" {profiles[v][0][j] - profiles[v][1][j]:>14.6e}"
                    for v in args.var
                )
                fh.write(f"  {alt:>8.1f}{cols}\n")
        print(f"Ratios saved: {datafile}")

    # ------------------------------------------------------------------
    # Plot — one subplot per variable
    # ------------------------------------------------------------------
    n    = len(args.var)
    font = BASE_FONT

    if n == 1:
        nrows, ncols = 1, 1
        col_type = "single"
    else:
        nrows, ncols = (n + 1) // 2, 2
        col_type = "double"

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL_WIDTH_CM[col_type] * cm, 8 * nrows * cm),
                             sharey=True, constrained_layout=True,
                             squeeze=False)
    axes_flat = axes.flatten()

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    if avg_ls is not None:
        ls_note = (f"Perihelion: Ls={Ls[i_peri]:.0f}°±{avg_ls}°\n"
                   f"Aphelion:  Ls={Ls[i_aph]:.0f}°±{avg_ls}°")
    else:
        ls_note = (f"Perihelion: Ls={Ls[i_peri]:.0f}°\n"
                   f"Aphelion:  Ls={Ls[i_aph]:.0f}°")

    for i, varname in enumerate(args.var):
        ax = axes_flat[i]
        ax.plot(ratios[varname], alt_values, color='k')
        ax.axvline(1.0, color='k', linewidth=0.5, linestyle=':')
        ax.tick_params(labelsize=font["tick"])
        ax.grid(True, alpha=0.3)

        varlabel = VAR_LABELS.get(varname, varname)
        ax.set_xlabel(rf"{varlabel} perihelion / aphelion", fontsize=font["label"])

        # Ls annotation on first subplot only
        if i == 0:
            ax.text(0.98, 0.97, ls_note, transform=ax.transAxes,
                    fontsize=font["text"], va='top', ha='right')

    # Y label on left column only
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude (km)", fontsize=font["label"])

    vartag  = "_".join(args.var) if n == 1 else "multi"
    outfile = f"{vartag}_peri_aph_ratio_{casetag}_{mode}.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved: {outfile}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
