#!/usr/bin/env python3
"""
Plot the [O]/[CO2] number density ratio vs altitude from one or more
reduced NC files. One line per file on a single plot.

Usage:
    plot_O_CO2_ratio.py case1/case1_reduced.nc -mode global
    plot_O_CO2_ratio.py case1/case1_reduced.nc case2/case2_reduced.nc -mode lt14
"""

import sys
import os
import argparse

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

plt.rcParams.update({'lines.linewidth': 1.0})

cm = 1 / 2.54
COL_WIDTH_CM = {"single": 8, "double": 16}
BASE_FONT    = {"label": 9, "tick": 8, "legend": 8, "text": 8}

LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]


def decode_char_var(var):
    """Convert a netCDF4 string/char variable to a list of stripped strings."""
    data = var[:]
    if data.dtype.kind in ("S", "U"):
        return [str(s).strip() for s in data]
    if data.ndim == 2:
        return ["".join(row.astype(str)).strip() for row in data]
    return [str(s).strip() for s in data]


def get_ratio_profile(filename, mode):
    """
    Return (alt_values, O_CO2_ratio) for the given file and mode.
    Averages over all time steps (and latitude for lat-dependent modes).
    """
    ds = nc.Dataset(filename)
    alt_values  = np.array(ds.variables["altitude"][:])
    lat_modes   = decode_char_var(ds.variables["mode_lat"])
    point_modes = decode_char_var(ds.variables["mode_point"])

    if mode in lat_modes:
        mode_idx = lat_modes.index(mode)
        # (time, latitude, altitude, mode_lat) → mean over time and latitude
        O   = np.nanmean(ds.variables["O"]  [:, :, :, mode_idx], axis=(0, 1))
        CO2 = np.nanmean(ds.variables["CO2"][:, :, :, mode_idx], axis=(0, 1))
    elif mode in point_modes:
        mode_idx = point_modes.index(mode)
        # (time, altitude, mode_point) → mean over time
        O   = np.nanmean(ds.variables["O_point"]  [:, :, mode_idx], axis=0)
        CO2 = np.nanmean(ds.variables["CO2_point"][:, :, mode_idx], axis=0)
    else:
        ds.close()
        sys.exit(f"Error: mode '{mode}' not found in {filename}.\n"
                 f"  lat modes:   {lat_modes}\n"
                 f"  point modes: {point_modes}")

    ds.close()
    return alt_values, O / CO2


def main():
    parser = argparse.ArgumentParser(
        description="Plot [O]/[CO2] ratio vs altitude"
    )
    parser.add_argument("filenames", nargs="+",
                        help="Reduced NetCDF file(s)")
    parser.add_argument("-mode", required=True,
                        help="Mode string (e.g. global, lt14, subsolar)")
    parser.add_argument("-show", action="store_true",
                        help="Display plot interactively")
    args = parser.parse_args()

    for f in args.filenames:
        if not os.path.isfile(f):
            sys.exit(f"Error: file not found: {f}")

    font = BASE_FONT
    fig, ax = plt.subplots(figsize=(COL_WIDTH_CM["single"] * cm, 8 * cm),
                           constrained_layout=True)

    for i, fname in enumerate(args.filenames):
        alt_values, ratio = get_ratio_profile(fname, args.mode)
        label = os.path.splitext(os.path.basename(fname))[0].replace("_reduced", "")
        ax.plot(ratio, alt_values,
                color='k', linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=label)

    ax.set_xlabel(r"[O] / [CO$_2$]", fontsize=font["label"])
    ax.set_ylabel("Altitude (km)", fontsize=font["label"])
    ax.tick_params(labelsize=font["tick"])
    ax.legend(fontsize=font["legend"], frameon=False)
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='k', linewidth=0.5, linestyle=':')

    if len(args.filenames) == 1:
        casetag = os.path.splitext(os.path.basename(args.filenames[0]))[0].replace("_reduced", "")
    else:
        casetag = "multi"
    outfile = f"O_CO2_ratio_{casetag}_{args.mode}.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
