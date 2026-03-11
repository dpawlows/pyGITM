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
    parser.add_argument("-var",  default=None, help="Variable name to plot ratio for")
    parser.add_argument("-mode", default=None, help="Mode string (e.g. global, lt14, subsolar)")
    parser.add_argument("-show", action="store_true", help="Display plot interactively")
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
        ds.close()
        sys.exit(1)

    varname = args.var
    mode    = args.mode

    if mode in lat_modes:
        mode_idx  = lat_modes.index(mode)
        # (time, latitude, altitude, mode_lat) → average over latitude
        data = np.array(ds.variables[varname][:, :, :, mode_idx])  # (time, lat, alt)
        profile_peri = np.nanmean(data[i_peri], axis=0)  # (alt,)
        profile_aph  = np.nanmean(data[i_aph],  axis=0)
    elif mode in point_modes:
        mode_idx  = point_modes.index(mode)
        # (time, altitude, mode_point)
        data = np.array(ds.variables[varname + "_point"][:, :, mode_idx])  # (time, alt)
        profile_peri = data[i_peri]  # (alt,)
        profile_aph  = data[i_aph]
    else:
        ds.close()
        sys.exit(f"Error: mode '{mode}' not found. Available: {lat_modes + point_modes}")

    ds.close()

    ratio = profile_peri / profile_aph

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    font = BASE_FONT
    fig, ax = plt.subplots(figsize=(COL_WIDTH_CM["single"] * cm, 8 * cm),
                           constrained_layout=True)

    ax.plot(ratio, alt_values, color='k')
    ax.axvline(1.0, color='k', linewidth=0.5, linestyle=':')

    varlabel = VAR_LABELS.get(varname, varname)
    ax.set_xlabel(rf"{varlabel} perihelion / aphelion", fontsize=font["label"])
    ax.set_ylabel("Altitude (km)", fontsize=font["label"])
    ax.tick_params(labelsize=font["tick"])
    ax.grid(True, alpha=0.3)

    # Annotate with the Ls values used
    ax.text(0.98, 0.97,
            f"Perihelion: Ls={Ls[i_peri]:.0f}°\nAphelion:  Ls={Ls[i_aph]:.0f}°",
            transform=ax.transAxes, fontsize=font["text"],
            va='top', ha='right')

    casetag = os.path.splitext(os.path.basename(args.filename))[0].replace("_reduced", "")
    outfile = f"{varname}_peri_aph_ratio_{casetag}_{mode}.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved: {outfile}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
