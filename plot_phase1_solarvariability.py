#!/usr/bin/env python3
"""
Plot a variable as a function of F10.7 and compute d<var>/dF10.7 via linear regression.

Usage:
    python plot_phase1_solarvariability.py <file1.nc> [file2.nc ...] [-var VAR] [-mode MODE] [-alt ALT]

Up to 4 NetCDF files may be specified; each gets its own subplot with shared axes.
"""

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import stats
from gitm_routines import get_units

plt.rcParams.update({'lines.linewidth': 1.0})

cm = 1 / 2.54
COL_WIDTH_CM = {"single": 8, "double": 16}
BASE_FONT    = {"label": 9, "tick": 8, "legend": 8, "text": 8}


def scale_fonts(col_type="single", base_font=BASE_FONT,
                reference_col_type="single"):
    """Scale font sizes proportionally to figure width."""
    scale = COL_WIDTH_CM[col_type] / COL_WIDTH_CM[reference_col_type]
    return {k: v * scale for k, v in base_font.items()}

def var_label(varname):
    """Return 'varname (units)' if units are known, else just 'varname'."""
    units = get_units(varname)
    return f"{varname} ({units})" if units else varname


def decode_char_var(var):
    """Convert a netCDF4 string/char variable to a list of stripped strings."""
    data = var[:]
    if data.dtype.kind in ("S", "U"):
        return [str(s).strip() for s in data]
    # Char array stored as object or masked array — join along last axis
    if data.ndim == 2:
        return ["".join(row.astype(str)).strip() for row in data]
    return [str(s).strip() for s in data]


F107_FILE = os.path.expanduser(
    "~/UpperAtmosphere/F107/radio_flux_adjusted_observation.txt"
)


# ---------------------------------------------------------------------------
# F10.7 loader
# ---------------------------------------------------------------------------

def load_f107(filepath):
    """
    Parse radio_flux_adjusted_observation.txt.
    Columns: year month day cont_day ... f10.7 f10.7_c f10.7_p f10.7_f ...
    Uses f10.7_c (index 13): gap-filled and flare-corrected values.
    Missing value sentinel -1 is excluded.
    Returns dict mapping datetime(date) -> f10.7_c (sfu).
    """
    f107 = {}
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            try:
                dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                val = float(parts[13])  # f10.7_c
                if val >= 0:
                    f107[dt] = val
            except ValueError:
                continue
    return f107


def get_f107_for_dates(dates, f107_dict):
    """Match f10.7_c values to a list of datetimes. Returns an array."""
    values = [f107_dict.get(datetime(dt.year, dt.month, dt.day), np.nan)
              for dt in dates]
    return np.array(values, dtype=float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def get_data_vars(ds, lat_modes, point_modes):
    """Return sorted list of base variable names available in the dataset."""
    coord_names = {"time", "altitude", "latitude", "mode_lat", "mode_point",
                   "Ls", "nfiles", "average"}
    base_vars = set()
    for name in ds.variables:
        if name in coord_names:
            continue
        if name.endswith("_point"):
            base_vars.add(name[:-6])
        else:
            base_vars.add(name)
    return sorted(base_vars)


def print_available(lat_modes, point_modes, alt_values, data_vars):
    print("\nAvailable variables (-var):")
    print("  " + ", ".join(data_vars))
    print("\nAvailable modes  (-mode):")
    print("  lat-based:   " + ", ".join(lat_modes))
    print("  point-based: " + ", ".join(point_modes))
    print("Available alts   (-alt):  " + ", ".join(str(a) for a in alt_values))


def extract_data(filename, varname, mode, alt_idx, lat_modes, point_modes, f107_dict):
    """Extract (T, f107, Ls) arrays from a single NetCDF file."""
    ds = nc.Dataset(filename)
    time_var = ds.variables["time"]
    raw_times = nc.num2date(time_var[:], units=time_var.units,
                            calendar=getattr(time_var, "calendar", "standard"))
    dates = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
             for t in raw_times]

    if mode in lat_modes:
        mode_idx = lat_modes.index(mode)
        data_full = np.array(ds.variables[varname][:, :, alt_idx, mode_idx])
        T = np.nanmean(data_full, axis=1)
    else:
        mode_idx = point_modes.index(mode)
        T = np.array(ds.variables[varname + "_point"][:, alt_idx, mode_idx])

    Ls = np.array(ds.variables["Ls"][:])
    ds.close()

    f107 = get_f107_for_dates(dates, f107_dict)
    return T, f107, Ls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot a variable vs F10.7 and compute d<var>/dF10.7"
    )
    parser.add_argument("filenames", nargs="+",
                        help="NetCDF file(s) — up to 4 (e.g. case1/case1_reduced.nc)")
    parser.add_argument("-var", default=None,
                        help="Variable name (e.g. Temperature, Rho, [O])")
    parser.add_argument("-mode", default=None,
                        help="Mode string (e.g. global, lt04, subsolar, ...)")
    parser.add_argument("-alt", type=float, nargs="+", default=None,
                        help="Altitude(s) in km — multiple alts require a single file")
    parser.add_argument("-ls", action="store_true",
                        help="Color scatter points by Solar Longitude (Ls)")
    parser.add_argument("-show", action="store_true",
                        help="Display the plot interactively")
    args = parser.parse_args()

    multi_file = len(args.filenames) > 1
    multi_alt  = args.alt is not None and len(args.alt) > 1

    if multi_file and multi_alt:
        sys.exit("Error: specify either multiple files or multiple altitudes, not both.")
    if multi_file and len(args.filenames) > 4:
        sys.exit("Error: at most 4 NetCDF files may be specified.")
    if multi_alt and len(args.alt) > 4:
        sys.exit("Error: at most 4 altitudes may be specified.")

    for f in args.filenames:
        if not os.path.isfile(f):
            sys.exit(f"Error: file not found: {f}")

    # Validate var/mode/alt using the first file
    ds0 = nc.Dataset(args.filenames[0])
    alt_values  = np.array(ds0.variables["altitude"][:])
    lat_modes   = decode_char_var(ds0.variables["mode_lat"])
    point_modes = decode_char_var(ds0.variables["mode_point"])
    all_modes   = lat_modes + point_modes
    data_vars   = get_data_vars(ds0, lat_modes, point_modes)
    ds0.close()

    var_ok  = args.var in data_vars
    mode_ok = args.mode in all_modes
    alt_ok  = args.alt is not None and all(a in alt_values for a in args.alt)

    if not var_ok or not mode_ok or not alt_ok:
        if args.var is not None and not var_ok:
            print(f"Error: variable '{args.var}' not recognised.")
        if args.mode is not None and not mode_ok:
            print(f"Error: mode '{args.mode}' not recognised.")
        if args.alt is not None and not alt_ok:
            bad = [a for a in args.alt if a not in alt_values]
            print(f"Error: alt(s) not available: {bad}")
        print_available(lat_modes, point_modes, alt_values, data_vars)
        sys.exit(0)

    varname = args.var

    # ------------------------------------------------------------------
    # Load F10.7 once
    # ------------------------------------------------------------------
    f107_dict = load_f107(F107_FILE)
    print(f"Using F10.7 file: {os.path.basename(F107_FILE)}")

    # ------------------------------------------------------------------
    # Extract data — one dataset per subplot
    # ------------------------------------------------------------------
    datasets = []

    if multi_alt:
        # One subplot per altitude, single file
        fname = args.filenames[0]
        for alt in args.alt:
            alt_idx = int(np.where(alt_values == alt)[0][0])
            print(f"\n--- {fname}, {int(alt)} km ---")
            T, f107, Ls = extract_data(fname, varname, args.mode, alt_idx,
                                       lat_modes, point_modes, f107_dict)
            mask = np.isfinite(f107) & np.isfinite(T)
            n_matched = mask.sum()
            print(f"Data points used: {n_matched} / {len(mask)}")
            if n_matched < 2:
                sys.exit(f"Error: fewer than 2 valid data points at {int(alt)} km.")
            f107_c = f107[mask]
            T_c    = T[mask]
            ls_c   = Ls[mask]
            slope, intercept, r_value, p_value, std_err = stats.linregress(f107_c, T_c)
            print(f"  d{varname}/dF10.7 = {slope:.4f} /sfu,  R² = {r_value**2:.4f}")
            datasets.append(dict(f107=f107_c, T=T_c, Ls=ls_c,
                                 slope=slope, intercept=intercept,
                                 r=r_value, p_value=p_value, std_err=std_err,
                                 alt=int(alt), label=f"{int(alt)} km"))
    else:
        # One subplot per file, single altitude
        alt_idx = int(np.where(alt_values == args.alt[0])[0][0])
        for fname in args.filenames:
            print(f"\n--- {fname} ---")
            T, f107, Ls = extract_data(fname, varname, args.mode, alt_idx,
                                       lat_modes, point_modes, f107_dict)
            mask = np.isfinite(f107) & np.isfinite(T)
            n_matched = mask.sum()
            print(f"Data points used: {n_matched} / {len(mask)}")
            if n_matched < 2:
                sys.exit(f"Error: fewer than 2 valid data points in {fname}.")
            f107_c = f107[mask]
            T_c    = T[mask]
            ls_c   = Ls[mask]
            slope, intercept, r_value, p_value, std_err = stats.linregress(f107_c, T_c)
            print(f"  d{varname}/dF10.7 = {slope:.4f} /sfu,  R² = {r_value**2:.4f}")
            label = os.path.splitext(os.path.basename(fname))[0].replace("_reduced", "")
            datasets.append(dict(f107=f107_c, T=T_c, Ls=ls_c,
                                 slope=slope, intercept=intercept,
                                 r=r_value, p_value=p_value, std_err=std_err,
                                 alt=int(args.alt[0]), label=label))

    # ------------------------------------------------------------------
    # Build subplot grid
    # ------------------------------------------------------------------
    n = len(datasets)
    if n == 1:
        nrows, ncols = 1, 1
        col_type = "single"
        row_height_cm = 7.0
    else:
        nrows, ncols = (n + 1) // 2, 2
        col_type = "double"
        row_height_cm = 7.0

    font = BASE_FONT
    width_cm  = COL_WIDTH_CM[col_type]
    height_cm = row_height_cm * nrows

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_cm * cm, height_cm * cm),
                             sharex=True, sharey=True,
                             constrained_layout=True,
                             squeeze=False)
    axes_flat = axes.flatten()

    # Hide unused axes (e.g. bottom-right cell when n=3)
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    varlabel = VAR_LABELS.get(varname, varname)

    sc_last = None
    for i, (ax, d) in enumerate(zip(axes_flat[:n], datasets)):
        if args.ls:
            sc_last = ax.scatter(d["f107"], d["T"], c=d["Ls"], cmap="rainbow",
                                 vmin=0, vmax=360, alpha=0.8, s=20, zorder=3)
        else:
            ax.scatter(d["f107"], d["T"], color="#444444", alpha=0.6, s=20, zorder=3)

        f107_line = np.linspace(d["f107"].min(), d["f107"].max(), 300)
        ax.plot(f107_line, d["slope"] * f107_line + d["intercept"],
                "r-", linewidth=2,
                label=(rf"$\Delta${varlabel}/$\Delta$F10.7 = {d['slope']:.3f} /sfu") #\n"
                    #    f"$R^2$ = {d['r']**2:.3f}")
            )
        ax.set_title(d["label"], fontsize=font["text"])
        ax.tick_params(labelsize=font["tick"])
        ax.legend(fontsize=font["legend"],frameon=False)
        ax.grid(True, alpha=0.3)

    # Colorbar (Ls) drawn once against the visible axes
    if args.ls and sc_last is not None:
        fig.colorbar(sc_last, ax=axes_flat[:n].tolist(),
                     label="Ls (deg)", shrink=0.8).ax.tick_params(labelsize=font["tick"])

    # X label on the bottom-most visible subplot in each column;
    # if that subplot is not in the last row, re-enable its tick labels.
    for col in range(ncols):
        for row in range(nrows - 1, -1, -1):
            if row * ncols + col < n:
                axes[row, col].set_xlabel("F10.7 (sfu)", fontsize=font["label"])
                if row < nrows - 1:
                    axes[row, col].tick_params(labelbottom=True)
                break

    # Y label on left column only
    for ax in axes[:, 0]:
        ax.set_ylabel(var_label(varname), fontsize=font["label"])


    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if multi_alt:
        filetag = os.path.splitext(os.path.basename(args.filenames[0]))[0].replace("_reduced", "")
        alttag  = "_".join(str(int(a)) for a in args.alt)
        outfile = f"{varname}_vs_F107_{filetag}_{args.mode}_alt{alttag}.png"
    elif multi_file:
        filetag = "multi"
        alttag  = str(int(args.alt[0]))
        outfile = f"{varname}_vs_F107_multi_{args.mode}_alt{alttag}.png"
    else:
        filetag = os.path.splitext(os.path.basename(args.filenames[0]))[0].replace("_reduced", "")
        outfile = f"{varname}_vs_F107_{filetag}_{args.mode}_alt{int(args.alt[0])}.png"

    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved: {outfile}")

    # ------------------------------------------------------------------
    # Write regression results to text file
    # ------------------------------------------------------------------
    regressfile = f"{filetag}_regress.txt"
    with open(regressfile, "w") as fh:
        fh.write(f"# Regression results: {varname} vs F10.7\n")
        fh.write(f"# mode={args.mode}\n")
        fh.write(f"# {'alt_km':>8} {'slope':>12} {'intercept':>12} "
                 f"{'R2':>8} {'p_value':>12} {'std_err':>12}\n")
        for d in datasets:
            fh.write(f"  {d['alt']:>8} "
                     f"{d['slope']:>12.4f} {d['intercept']:>12.4f} "
                     f"{d['r']**2:>8.4f} {d['p_value']:>12.4e} {d['std_err']:>12.4f}\n")
    print(f"Regression results saved: {regressfile}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
