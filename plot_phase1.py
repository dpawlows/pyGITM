#!/usr/bin/env python3

import argparse
import os
import numpy as np
import sys
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import marstiming
import matplotlib as mpl
from gitm_routines import get_units

plt.rcParams.update({'lines.linewidth': 1.0})
cm = 1 / 2.54

VAR_DISPLAY = {
    "CO2": r"[CO$_2$]",
    "O":   r"[O]",
    "e":   r"[e-]",
}


def var_label(varname):
    """Return 'display_name (units)' if units are known, else just display_name."""
    display = VAR_DISPLAY.get(varname, varname)
    units = get_units(varname)
    return f"{display} ({units})" if units else display


def parse_args():

    parser = argparse.ArgumentParser(
        add_help=False,
        description="""
        Plot a reduced Phase 1 climatology variable from M-GITM.

        Usage modes:
        • script.py file.nc -h        → show dataset info
        • script.py -h                → show script help
        • script.py file.nc -var ...  → generate plot
        • script.py f1.nc f2.nc ...   → multi-file subplot grid
        """
    )

    parser.add_argument("ncfile", nargs="+", help="Reduced NetCDF file(s)")

    parser.add_argument("-var", help="Variable name")
    parser.add_argument("-mode", help="Mode (e.g., global, lt14, subsolar)")
    parser.add_argument("-alt", type=float, nargs="+", help="Altitude(s) in km")
    parser.add_argument("-show", action="store_true",
                        help="Show plot instead of saving")
    parser.add_argument(
        "-vmin",
        type=float,
        default=None,
        help="Minimum value for color scale"
    )

    parser.add_argument(
        "-vmax",
        type=float,
        default=None,
        help="Maximum value for color scale"
    )

    parser.add_argument(
        "-lsmin",
        type=float,
        default=None,
        help="Minimum Ls for x-axis"
    )

    parser.add_argument(
        "-lsmax",
        type=float,
        default=None,
        help="Maximum Ls for x-axis"
    )
    parser.add_argument("-h", "--help", action="store_true",
                        help="Show help or dataset info")

    return parser


def print_script_help():
    print("""
Plot a reduced Phase 1 climatology variable.

Examples:

  Show script help:
    script.py -h

  Show dataset info:
    script.py TEST_MY24_reduced.nc -h

  Plot single file:
    script.py TEST_MY24_reduced.nc -var Temperature -mode global -alt 200

  Plot multiple files (2x2 grid):
    script.py case1.nc case2.nc case3.nc case4.nc -var Temperature -mode global -alt 200
""")


def print_dataset_info(ds):

    print("\n=== DATASET INFORMATION ===\n")

    print("Available variables:")
    for v in ds.data_vars:
        print("  ", v)

    print("\nAvailable altitudes (km):")
    print("  ", ds.altitude.values)

    if "mode_lat" in ds.coords:
        print("\nLatitude-dependent modes (mode_lat):")
        print("  ", ds.mode_lat.values)

    if "mode_point" in ds.coords:
        print("\nPoint modes (mode_point):")
        print("  ", ds.mode_point.values)

    print("\n===========================\n")


def load_data(ncfile, varname, mode, alt):
    """Load a dataset and return (ds, da, ls, lat_dependent)."""
    ds = xr.open_dataset(ncfile)

    if ds.average is not None:
        low = ds.nfiles.values < np.median(ds.nfiles.values) * 0.5
        n = len(ds.nfiles.values)
        edge_threshold = 1

        low_indices = np.where(low)[0]
        edge_lows = [i for i in low_indices if i < edge_threshold or i > n - edge_threshold - 1]
        interior_lows = [i for i in low_indices if i not in edge_lows]

        if edge_lows:
            print(f"[{ncfile}] Low count at boundary Ls (expected): {ds.Ls.values[edge_lows]}")
        if interior_lows:
            print(f"[{ncfile}] [WARNING] Low count at interior Ls (possible gap): {ds.Ls.values[interior_lows]}")

    if "mode_lat" in ds.coords and mode in ds.mode_lat.values:
        if varname not in ds.data_vars:
            print(f"ERROR: Variable '{varname}' not found in {ncfile}.")
            print_dataset_info(ds)
            sys.exit(1)
        da = ds[varname].sel(mode_lat=mode)
        lat_dependent = True
    elif "mode_point" in ds.coords and mode in ds.mode_point.values:
        if varname + "_point" not in ds.data_vars:
            print(f"ERROR: Variable '{varname}' not found in {ncfile}.")
            print_dataset_info(ds)
            sys.exit(1)
        da = ds[varname + "_point"].sel(mode_point=mode)
        lat_dependent = False
    else:
        raise ValueError(f"[{ncfile}] Mode '{mode}' not found in dataset.")

    try:
        da = da.sel(altitude=alt)
    except KeyError:
        available = ds.altitude.values
        print(f"ERROR: Altitude {alt} km not found in {ncfile}.")
        print(f"Available altitudes (km): {available}")
        sys.exit(1)

    ls = ds.Ls.values.copy()
    wrap_indices = np.where(np.diff(ls) < -180)[0]
    for w in wrap_indices:
        ls[w+1:] += 360

    return ds, da, ls, lat_dependent


def apply_ls_limits(ax, ls, args):
    if args.lsmin is not None or args.lsmax is not None:
        lsmin = args.lsmin if args.lsmin is not None else ls.min()
        lsmax = args.lsmax if args.lsmax is not None else ls.max()
        ax.set_xlim(lsmin, lsmax)


def plot_single(ds, da, ls, lat_dependent, args):
    """Single-file plot: colormesh + line plot for lat-dependent modes."""
    varname = args.var
    mode = args.mode
    altitude = args.alt

    if lat_dependent:
        fig, axes = plt.subplots(
            2, 1,
            figsize=(10, 8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.5, 3]}
        )

        da_mean = da.mean("latitude")
        axes[0].plot(ls, da_mean.values, label='Latitude average')
        axes[0].set_ylabel(var_label(varname))
        axes[0].legend(frameon=False)

        im = axes[1].pcolormesh(
            ls,
            ds.latitude.values,
            da.T,
            shading="auto",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap='plasma'
        )
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}"))
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}"))

        apply_ls_limits(axes[0], ls, args)
        apply_ls_limits(axes[1], ls, args)

        axes[1].set_ylabel("Latitude")
        axes[1].set_xlabel("Solar Longitude (deg)")
        fig.colorbar(im, ax=axes[1], label=var_label(varname), pad=0.01)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        ax.plot(ls, da.values, label=mode)
        ax.set_ylabel(var_label(varname))
        ax.set_xlabel("Solar Longitude (deg)")
        apply_ls_limits(ax, ls, args)

    return fig


LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

# Physical column widths (match your journal)
COL_WIDTH_CM = {"single": 8, "double": 16}

# Font sizes tuned for single-column width
BASE_FONT = {"label": 9, "tick": 8, "legend": 8, "text": 8}

def scale_fonts(col_type="single", base_font=BASE_FONT,
                reference_col_type="single"):
    """Scale font sizes proportionally to figure width."""
    scale = COL_WIDTH_CM[col_type] / COL_WIDTH_CM[reference_col_type]
    return {k: v * scale for k, v in base_font.items()}

def plot_point(data_by_alt, filenames, args):
    """Point mode: one row per altitude, one line per file.

    data_by_alt: list of (alt, data_arrays, ls_arrays) tuples, one per altitude.
    """
    varname = args.var

    # Highest altitude in top subplot
    data_by_alt = sorted(data_by_alt, key=lambda x: x[0], reverse=True)
    n_alts = len(data_by_alt)
 
    col_type = 'single'
    row_height_cm=4.0
    font = scale_fonts(col_type)
    width_cm  = COL_WIDTH_CM[col_type]
    height_cm = row_height_cm * n_alts

    fig, axes = plt.subplots(
        n_alts, 1,
        figsize=(width_cm * cm, height_cm*cm),
        sharex='col',
        constrained_layout=True
    )
    axes = axes.ravel()

    for i, (alt, data_arrays, ls_arrays) in enumerate(data_by_alt):
        ax = axes[i]
        for j, (da, ls, f) in enumerate(zip(data_arrays, ls_arrays, filenames)):
            label = os.path.splitext(os.path.basename(f))[0].replace("_reduced", "")
            ax.plot(ls, da.values, label=label,
                    color='k', linestyle=LINESTYLES[j % len(LINESTYLES)])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}"))
        ax.tick_params(labelsize=font["tick"])

        if args.vmin is not None and args.vmax is not None:
            ax.set_ylim(args.vmin, args.vmax)

        ypos = 0.95
        ax.text(0.018, ypos, f"{int(alt)} km",
                transform=ax.transAxes, va='top', ha='left', fontsize=font["text"])

    if n_alts > 1:
        fig.supylabel(var_label(varname), fontsize=font["label"])
        fig.supxlabel("Solar Longitude (deg)", fontsize=font["label"])
    else:
        axes[-1].set_xlabel("Solar Longitude (deg)", fontsize=font["label"])
        axes[-1].set_ylabel(var_label(varname), fontsize=font["label"])
    apply_ls_limits(axes[-1], data_by_alt[0][2][0], args)  # sharex propagates to all

    if len(filenames) > 1:
        axes[0].legend(frameon=False, ncol=min(len(filenames), 2),
                       loc='lower left', fontsize=font["legend"])

    return fig


def plot_multi(datasets, data_arrays, ls_arrays, filenames, args, nrows=2, ncols=2):
    """Multi-file colormesh subplot grid with a shared colorbar."""
    varname = args.var
    n = len(datasets)
    fs = BASE_FONT

    # Global vmin/vmax across all datasets
    vmin = args.vmin if args.vmin is not None else min(np.nanmin(da.values) for da in data_arrays)
    vmax = args.vmax if args.vmax is not None else max(np.nanmax(da.values) for da in data_arrays)

    col_type = 'double'
    font = scale_fonts(col_type)
    row_height_cm = 7
    width_cm  = COL_WIDTH_CM[col_type]
    height_cm = row_height_cm * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_cm * cm, height_cm*cm),
        sharex='col',
        sharey='row',
        constrained_layout=True
    )
    axes_flat = axes.ravel()

    im = None
    for i, (ds, da, ls) in enumerate(zip(datasets, data_arrays, ls_arrays)):
        ax = axes_flat[i]
        im = ax.pcolormesh(
            ls, ds.latitude.values, da.T,
            shading='auto', vmin=vmin, vmax=vmax, cmap='plasma'
        )
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}"))
        ax.tick_params(labelsize=fs["tick"])
        label = os.path.splitext(os.path.basename(filenames[i]))[0].replace("_reduced", "")
        ax.set_title(f"{chr(ord('a') + i)}) {label}", fontsize=fs["text"])
        apply_ls_limits(ax, ls, args)

    # Hide unused axes
    for i in range(n, nrows * ncols):
        axes_flat[i].set_visible(False)

    # Hide interior tick labels (sharex='col' and sharey='row' link the ranges,
    # but label_outer() removes the redundant tick labels)
    for i in range(n):
        axes_flat[i].label_outer()

    # Set axis labels on the appropriate edge axes
    for col in range(ncols):
        # Bottom-most visible axis in this column gets the x-label
        for row in range(nrows - 1, -1, -1):
            idx = row * ncols + col
            if idx < n:
                axes_flat[idx].set_xlabel("Solar Longitude (deg)", fontsize=fs["label"])
                break

    for row in range(nrows):
        idx = row * ncols  # leftmost column
        if idx < n:
            axes_flat[idx].set_ylabel("Latitude", fontsize=fs["label"])

    fig.colorbar(im, ax=axes.ravel().tolist(), label=var_label(varname),
                 shrink=0.6, pad=0.01).ax.tick_params(labelsize=fs["tick"])

    return fig


def main():

    parser = parse_args()
    args = parser.parse_args()

    # -------------------------------------------------
    # Case 1: -h only → show script help
    # -------------------------------------------------
    if args.help and not args.ncfile:
        print_script_help()
        sys.exit()

    # -------------------------------------------------
    # Case 2: file.nc -h → show dataset info (first file)
    # -------------------------------------------------
    if args.help and args.ncfile:
        ds = xr.open_dataset(args.ncfile[0])
        print_dataset_info(ds)
        sys.exit()

    # -------------------------------------------------
    # Case 3: Plotting mode
    # -------------------------------------------------
    if not args.ncfile:
        print("ERROR: Must specify ncfile and -var.")
        print_script_help()
        sys.exit()

    if not args.var:
        ds = xr.open_dataset(args.ncfile[0])
        print("ERROR: -var is required.")
        print_dataset_info(ds)
        sys.exit(1)

    if args.mode is None or args.alt is None:
        ds = xr.open_dataset(args.ncfile[0])
        if args.mode is None:
            print("ERROR: -mode is required.")
            if "mode_lat" in ds.coords:
                print(f"Available lat-dependent modes: {ds.mode_lat.values}")
            if "mode_point" in ds.coords:
                print(f"Available point modes:         {ds.mode_point.values}")
        if args.alt is None:
            print("ERROR: -alt is required.")
            print(f"Available altitudes (km): {ds.altitude.values}")
        sys.exit(1)

    varname = args.var
    mode = args.mode
    alts = args.alt  # always a list

    # Load all (file, alt) combinations; detect mode type from first entry
    all_loaded = {}
    for f in args.ncfile:
        for alt in alts:
            all_loaded[(f, alt)] = load_data(f, varname, mode, alt)

    lat_dependent = all_loaded[(args.ncfile[0], alts[0])][3]

    if lat_dependent:
        # Lat-dependent: multiple alts not supported; use first alt only
        if len(alts) > 1:
            print("WARNING: Multiple altitudes not supported for lat-dependent modes. Using first.")
        alt = alts[0]
        if len(args.ncfile) == 1:
            ds, da, ls, _ = all_loaded[(args.ncfile[0], alt)]
            fig = plot_single(ds, da, ls, True, args)
        else:
            datasets    = [all_loaded[(f, alt)][0] for f in args.ncfile]
            data_arrays = [all_loaded[(f, alt)][1] for f in args.ncfile]
            ls_arrays   = [all_loaded[(f, alt)][2] for f in args.ncfile]
            lat_flags   = [all_loaded[(f, alt)][3] for f in args.ncfile]
            if not all(lat_flags):
                print("ERROR: Mixed lat-dependent and point modes across files — cannot combine.")
                sys.exit(1)
            fig = plot_multi(datasets, data_arrays, ls_arrays, args.ncfile, args)
    else:
        # Point mode: support any number of files and altitudes
        data_by_alt = []
        for alt in alts:
            data_arrays = [all_loaded[(f, alt)][1] for f in args.ncfile]
            ls_arrays   = [all_loaded[(f, alt)][2] for f in args.ncfile]
            data_by_alt.append((alt, data_arrays, ls_arrays))
        fig = plot_point(data_by_alt, args.ncfile, args)

    if args.show:
        plt.show()
    else:
        alt_str = "_".join(str(int(a)) for a in alts)
        outfile = f"{varname}_{mode}_{alt_str}km.png"
        plt.savefig(outfile, dpi=150)
        print(f"\nSaved: {outfile}")

if __name__ == "__main__":
    main()
