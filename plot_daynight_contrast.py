#!/usr/bin/env python3
"""
Compute the day-night contrast ratio for one or more variables:

    contrast(Ls) = ( X_day(Ls) - X_night(Ls) ) / X_mean(Ls)

  X_day   : sza_day  point mode
  X_night : sza_night point mode
  X_mean  : global lat mode, collapsed with a cos-latitude weighting:
              X_mean(Ls) = sum_lat[ X(lat,Ls) * cos(lat) ]
                         / sum_lat[ cos(lat) ]

Layout: one column per variable, one row per altitude.
  -absolute    : plot absolute day-night difference instead of contrast ratio.
  -percentdiff : plot percent difference (X_day - X_night) / X_night * 100.
  -shareaxis   : share y-axis range across altitudes for the same variable.
Multiple files produce multiple lines per subplot.

Usage:
    plot_daynight_contrast.py case1/case1_reduced.nc -var Temperature -alt 200
    plot_daynight_contrast.py case1/case1_reduced.nc case2/case2_reduced.nc \
        -var Temperature Rho -alt 200 135 -show
    plot_daynight_contrast.py case1/case1_reduced.nc -var Temperature -alt 200 135 -absolute
"""

import sys
import os
import argparse

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

from gitm_routines import get_units

plt.rcParams.update({'lines.linewidth': 1.0})
cm = 1 / 2.54
COL_WIDTH_CM = {"single": 8, "double": 16}
BASE_FONT = {"label": 9, "tick": 8, "legend": 8, "text": 8}

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

CR_LABELS = {
    "CO2":         r"CR$_{[CO_2]}$",
    "O":           r"CR$_{[O]}$",
    "Rho":         r"CR$_\rho$",
    "Temperature": r"CR$_T$",
    "Vneast":      r"CR$_u$",
    "Vnnorth":     r"CR$_v$",
    "Vnup":        r"CR$_w$",
    "e":           r"CR$_{[e^-]}$",
}

DELTA_LABELS = {
    "CO2":         r"$\Delta$[CO$_2$]",
    "O":           r"$\Delta$[O]",
    "Rho":         r"$\Delta\rho$",
    "Temperature": r"$\Delta T$",
    "Vneast":      r"$\Delta u$",
    "Vnnorth":     r"$\Delta v$",
    "Vnup":        r"$\Delta w$",
    "e":           r"$\Delta$[e$^-$]",
}

PCTDIFF_LABELS = {
    "CO2":         r"$\Delta$[CO$_2$]/[CO$_2$]$_{night}$ (%)",
    "O":           r"$\Delta$[O]/[O]$_{night}$ (%)",
    "Rho":         r"$\Delta\rho/\rho_{night}$ (%)",
    "Temperature": r"$\Delta T/T_{night}$ (%)",
    "Vneast":      r"$\Delta u/u_{night}$ (%)",
    "Vnnorth":     r"$\Delta v/v_{night}$ (%)",
    "Vnup":        r"$\Delta w/w_{night}$ (%)",
    "e":           r"$\Delta$[e$^-$]/[e$^-$]$_{night}$ (%)",
}

LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]


def decode_char_var(var):
    """Convert a netCDF4 string/char variable to a list of stripped strings."""
    data = var[:]
    if data.dtype.kind in ("S", "U"):
        return [str(s).strip() for s in data]
    if data.ndim == 2:
        return ["".join(row.astype(str)).strip() for row in data]
    return [str(s).strip() for s in data]


def cosweight_mean(data_lat, lat_deg):
    """
    Collapse the latitude dimension with cos-latitude weighting.

    Parameters
    ----------
    data_lat : array (time, lat)
    lat_deg  : array (lat,)  in degrees

    Returns
    -------
    mean : array (time,)
    """
    weights = np.cos(np.radians(lat_deg))          # (lat,)
    weights = weights[np.newaxis, :]               # (1, lat)
    return np.nansum(data_lat * weights, axis=1) / np.nansum(
        np.where(np.isfinite(data_lat), weights, 0.0), axis=1
    )


def load_contrasts(filename, varnames, alts_sorted):
    """
    Open one NetCDF file and return:
      contrasts[varname][alt] = 1-D contrast ratio array (time,)
      absolutes[varname][alt] = 1-D absolute difference array (time,)
      pctdiffs[varname][alt]  = 1-D percent difference array (time,)
      Ls                      = unwrapped Ls array (time,)
      label                   = short case label string
    """
    ds = nc.Dataset(filename)

    Ls_raw      = np.array(ds.variables["Ls"][:])
    alt_values  = np.array(ds.variables["altitude"][:])
    lat_values  = np.array(ds.variables["latitude"][:])
    lat_modes   = decode_char_var(ds.variables["mode_lat"])
    point_modes = decode_char_var(ds.variables["mode_point"])

    # Validate required modes
    if "global" not in lat_modes:
        ds.close()
        sys.exit(f"Error [{filename}]: 'global' lat mode not found. "
                 f"Available: {lat_modes}")
    for m in ("sza_day", "sza_night"):
        if m not in point_modes:
            ds.close()
            sys.exit(f"Error [{filename}]: '{m}' point mode not found. "
                     f"Available: {point_modes}")

    global_idx    = lat_modes.index("global")
    sza_day_idx   = point_modes.index("sza_day")
    sza_night_idx = point_modes.index("sza_night")

    # Validate altitudes
    bad_alts = [a for a in alts_sorted if a not in alt_values]
    if bad_alts:
        ds.close()
        sys.exit(f"Error [{filename}]: altitude(s) not available: {bad_alts}. "
                 f"Available: {list(alt_values)}")

    # Validate variables
    coord_names = {"time", "altitude", "latitude", "mode_lat", "mode_point",
                   "Ls", "nfiles", "average", "year", "sol"}
    available_vars = sorted({
        name[:-6] if name.endswith("_point") else name
        for name in ds.variables
        if name not in coord_names
    })
    bad_vars = [v for v in varnames if v not in available_vars]
    if bad_vars:
        ds.close()
        sys.exit(f"Error [{filename}]: variable(s) not found: {bad_vars}. "
                 f"Available: {', '.join(available_vars)}")

    # Unwrap Ls
    Ls = Ls_raw.copy()
    for w in np.where(np.diff(Ls) < -180)[0]:
        Ls[w + 1:] += 360

    contrasts = {v: {} for v in varnames}
    absolutes = {v: {} for v in varnames}
    pctdiffs  = {v: {} for v in varnames}
    for varname in varnames:
        for alt in alts_sorted:
            alt_idx = int(np.where(alt_values == alt)[0][0])

            x_day = np.array(
                ds.variables[varname + "_point"][:, alt_idx, sza_day_idx],
                dtype=float
            )
            x_night = np.array(
                ds.variables[varname + "_point"][:, alt_idx, sza_night_idx],
                dtype=float
            )
            x_global = np.array(
                ds.variables[varname][:, :, alt_idx, global_idx],
                dtype=float
            )                                               # (time, lat)
            x_mean = cosweight_mean(x_global, lat_values)  # (time,)

            diff = x_day - x_night
            contrast = diff / x_mean
            contrasts[varname][alt] = contrast
            absolutes[varname][alt] = diff
            pctdiffs[varname][alt]  = diff / x_night * 100.0
            print(f"  {os.path.basename(filename)}  {varname:>14}  {int(alt):>4} km  "
                  f"mean={np.nanmean(contrast):>8.4f}  "
                  f"min={np.nanmin(contrast):>8.4f}  "
                  f"max={np.nanmax(contrast):>8.4f}")
    ds.close()
    label = (os.path.splitext(os.path.basename(filename))[0]
             .replace("_reduced", ""))
    return contrasts, absolutes, pctdiffs, Ls, label


def main():
    parser = argparse.ArgumentParser(
        description="Plot day-night contrast ratio vs Ls"
    )
    parser.add_argument("filenames", nargs="+",
                        help="Reduced NetCDF file(s)")
    parser.add_argument("-var", nargs="+", required=True,
                        help="Variable name(s)")
    parser.add_argument("-alt", type=float, nargs="+", required=True,
                        help="Altitude(s) in km")
    parser.add_argument("-absolute", action="store_true",
                        help="Plot absolute day-night difference instead of contrast ratio")
    parser.add_argument("-percentdiff", action="store_true",
                        help="Plot percent difference (X_day - X_night) / X_night * 100")
    parser.add_argument("-filter", type=int, default=None, metavar="WINDOW",
                        help="Apply Savitzky-Golay smoothing with the given window size (must be odd)")
    parser.add_argument("-shareaxis", action="store_true",
                        help="Share y-axis range across all altitudes for the same variable")
    parser.add_argument("-show", action="store_true",
                        help="Display plot interactively")
    args = parser.parse_args()

    for f in args.filenames:
        if not os.path.isfile(f):
            sys.exit(f"Error: file not found: {f}")

    if args.filter is not None:
        if args.filter < 5:
            sys.exit("Error: -filter window must be >= 5")
        if args.filter % 2 == 0:
            args.filter += 1
            print(f"Note: -filter window rounded up to {args.filter} (must be odd)")

    alts_sorted = sorted(args.alt, reverse=True)  # highest alt at top

    # ------------------------------------------------------------------
    # Load all files
    # ------------------------------------------------------------------
    all_data = []   # list of (contrasts_dict, absolutes_dict, pctdiffs_dict, Ls_array, label)
    for fname in args.filenames:
        c, a, p, ls, label = load_contrasts(fname, args.var, alts_sorted)
        all_data.append((c, a, p, ls, label))

    multi_file = len(args.filenames) > 1

    # ------------------------------------------------------------------
    # Plot — rows = altitudes, columns = variables
    # -absolute: only absolute day-night difference; otherwise contrast ratio
    # -shareaxis: shared y-range per variable (column)
    # ------------------------------------------------------------------
    n_rows     = len(alts_sorted)
    n_cols     = len(args.var)
    row_height = 4.0
    font       = BASE_FONT

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(COL_WIDTH_CM["single"] * n_cols * cm, row_height * n_rows * cm),
        sharex=True,
        sharey="col" if args.shareaxis else "none",
        constrained_layout=True,
        squeeze=False
    )

    def plot_lines(ax, data_getter, varname, alt):
        for j, entry in enumerate(all_data):
            contrasts, absolutes, pctdiffs, Ls, label = entry
            ydata = data_getter(contrasts, absolutes, pctdiffs, varname, alt)
            if args.filter is not None:
                ydata = savgol_filter(ydata, args.filter, polyorder=3)
            ax.plot(Ls, ydata,
                    color="k",
                    linestyle=LINESTYLES[j % len(LINESTYLES)],
                    linewidth=1.0,
                    label=label if multi_file else None)
        ax.axhline(0.0, color="0.6", linewidth=0.5, linestyle="--")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}")
        )
        ax.tick_params(labelsize=font["tick"])
        ax.grid(True, alpha=0.3)

    for col, varname in enumerate(args.var):
        units = get_units(varname)
        if args.percentdiff:
            ylabel = PCTDIFF_LABELS.get(varname, f"% diff ({varname})")
            getter = lambda c, a, p, v, h: p[v][h]
        elif args.absolute:
            delta_base = DELTA_LABELS.get(varname, f"$\\Delta${varname}")
            ylabel = delta_base + (f" ({units})" if units else "")
            getter = lambda c, a, p, v, h: a[v][h]
        else:
            ylabel = CR_LABELS.get(varname, f"CR ({varname})")
            getter = lambda c, a, p, v, h: c[v][h]

        axes[0, col].set_title(VAR_LABELS.get(varname, varname),
                               fontsize=font["label"])

        for row, alt in enumerate(alts_sorted):
            ax = axes[row, col]
            plot_lines(ax, getter, varname, alt)
            ax.text(0.02, 0.95, f"{int(alt)} km",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=font["text"])
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=font["label"])
            if multi_file and row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=font["legend"],
                          loc="lower right")

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Solar Longitude (deg)", fontsize=font["label"])

    if multi_file:
        casetag = "multi"
    else:
        casetag = (os.path.splitext(os.path.basename(args.filenames[0]))[0]
                   .replace("_reduced", ""))
    var_str = "_".join(args.var)
    alt_str = "_".join(str(int(a)) for a in args.alt)
    outfile = f"daynight_contrast_{var_str}_{casetag}_alt{alt_str}.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved: {outfile}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
