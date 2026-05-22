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
    """Return 'display_name (units)' if units are known, else just display_name.
    For ratio vars like 'O/CO2', returns display_num/display_denom (dimensionless)."""
    if "/" in varname:
        num, denom = varname.split("/", 1)
        return f"{VAR_DISPLAY.get(num, num)}/{VAR_DISPLAY.get(denom, denom)}"
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
    parser.add_argument("-pressure", type=float, nargs="+", help="Pressure level(s) in Pa")
    parser.add_argument("-show", action="store_true",
                        help="Show plot instead of saving")
    parser.add_argument(
        "-vmin",
        type=float,
        nargs="+",
        default=None,
        help="Minimum value(s) for axis/color scale; give 1 value or one per vertical level"
    )

    parser.add_argument(
        "-vmax",
        type=float,
        nargs="+",
        default=None,
        help="Maximum value(s) for axis/color scale; give 1 value or one per vertical level"
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
    parser.add_argument("-alog", action="store_true",
                        help="Use log scaling on the y-axis of each subplot")
    parser.add_argument("-legendloc", default="lower left",
                        help="Legend location (default: 'lower left'); accepts any pyplot loc string")
    parser.add_argument("-bw", action="store_true",
                        help="Black-and-white mode: use only black lines with varying styles")
    parser.add_argument("-lcolors", nargs="+", default=None,
                        help="Line colors for each input file (point mode)")
    parser.add_argument("-ls", nargs="+", default=None,
                        help="Line styles for each input file (point mode): s=solid, d=dashed, dd=dashdot, dt=dotted")
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

    if "altitude" in ds.coords:
        print("\nAvailable altitudes (km):")
        print("  ", ds.altitude.values)
    elif "pressure" in ds.coords:
        print("\nAvailable pressure levels (Pa):")
        print("  ", ds.pressure.values)

    if "mode_lat" in ds.coords:
        print("\nLatitude-dependent modes (mode_lat):")
        print("  ", ds.mode_lat.values)

    if "mode_point" in ds.coords:
        print("\nPoint modes (mode_point):")
        print("  ", ds.mode_point.values)

    print("\n===========================\n")


def load_data(ncfile, varname, mode, vert_val, vert_coord="altitude"):
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
        da = da.sel({vert_coord: vert_val})
    except KeyError:
        available = ds[vert_coord].values
        if vert_coord == "altitude":
            print(f"ERROR: Altitude {vert_val} km not found in {ncfile}.")
            print(f"Available altitudes (km): {available}")
        else:
            print(f"ERROR: Pressure {vert_val} Pa not found in {ncfile}.")
            print(f"Available pressure levels (Pa): {available}")
        sys.exit(1)

    ls = ds.Ls.values.copy() % 360
    sort_idx = np.argsort(ls, kind='stable')
    ls = ls[sort_idx]
    da = da.isel({da.dims[0]: sort_idx})

    return ds, da, ls, lat_dependent


def apply_ls_limits(ax, ls, args):
    lsmin = args.lsmin if args.lsmin is not None else 0
    lsmax = args.lsmax if args.lsmax is not None else 360
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
            vmin=args.vmin[0] if args.vmin is not None else None,
            vmax=args.vmax[0] if args.vmax is not None else None,
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
LS_MAP = {'s': '-', 'd': '--', 'dd': '-.', 'dt': ':'}
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

# Physical column widths (match your journal)
COL_WIDTH_CM = {"single": 8, "double": 16}

# Font sizes tuned for single-column width
BASE_FONT = {"label": 9, "tick": 8, "legend": 8, "text": 8}

def scale_fonts(col_type="single", base_font=BASE_FONT,
                reference_col_type="single"):
    """Scale font sizes proportionally to figure width."""
    scale = COL_WIDTH_CM[col_type] / COL_WIDTH_CM[reference_col_type]
    return {k: v * scale for k, v in base_font.items()}

def plot_point(data_by_vert, filenames, args, vert_coord="altitude"):
    """Point mode: one row per vertical level, one line per file.

    data_by_vert: list of (vert_val, data_arrays, ls_arrays) tuples, one per level.
    vert_coord: 'altitude' or 'pressure'
    """
    varname = args.var

    # Highest altitude (lowest pressure) in top subplot
    reverse_sort = (vert_coord == "altitude")
    data_by_vert = sorted(data_by_vert, key=lambda x: x[0], reverse=reverse_sort)
    n_levels = len(data_by_vert)

    col_type = 'single'
    row_height_cm=4.0
    font = scale_fonts(col_type)
    width_cm  = COL_WIDTH_CM[col_type]
    height_cm = row_height_cm * n_levels

    fig, axes = plt.subplots(
        n_levels, 1,
        figsize=(width_cm * cm, height_cm*cm),
        sharex='col',
        constrained_layout=True
    )
    axes = axes.ravel()

    for i, (vert_val, data_arrays, ls_arrays) in enumerate(data_by_vert):
        ax = axes[i]
        for j, (da, ls, f) in enumerate(zip(data_arrays, ls_arrays, filenames)):
            label = os.path.splitext(os.path.basename(f))[0].replace("_reduced", "")
            if args.bw:
                color = 'k'
            elif args.lcolors is not None:
                color = args.lcolors[j % len(args.lcolors)]
            else:
                color = COLORS[j % len(COLORS)]
            if args.ls is not None:
                key = args.ls[j % len(args.ls)]
                linestyle = LS_MAP.get(key, '-')
            else:
                linestyle = LINESTYLES[j % len(LINESTYLES)]
            ax.plot(ls, da.values, label=label,
                    color=color, linestyle=linestyle)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x % 360:.0f}"))
        ax.tick_params(labelsize=font["tick"])

        if args.alog:
            ax.set_yscale('log')

        vmin_i = args.vmin[i] if args.vmin is not None and len(args.vmin) > 1 else (args.vmin[0] if args.vmin is not None else None)
        vmax_i = args.vmax[i] if args.vmax is not None and len(args.vmax) > 1 else (args.vmax[0] if args.vmax is not None else None)
        if vmin_i is not None and vmax_i is not None:
            ax.set_ylim(vmin_i, vmax_i)

        if vert_coord == "altitude":
            level_label = f"{int(vert_val)} km"
        else:
            level_label = f"{vert_val:.2e} Pa"
        ypos = 0.95
        ax.text(0.018, ypos, level_label,
                transform=ax.transAxes, va='top', ha='left', fontsize=font["text"])

    if n_levels > 1:
        fig.supylabel(var_label(varname), fontsize=font["label"])
        fig.supxlabel("Solar Longitude (deg)", fontsize=font["label"])
    else:
        axes[-1].set_xlabel("Solar Longitude (deg)", fontsize=font["label"])
        axes[-1].set_ylabel(var_label(varname), fontsize=font["label"])
    apply_ls_limits(axes[-1], data_by_vert[0][2][0], args)  # sharex propagates to all

    if len(filenames) > 1:
        axes[0].legend(frameon=False, ncol=min(len(filenames), 2),
                       loc=args.legendloc, fontsize=font["legend"],
                       handlelength=1.2, handletextpad=0.4,
                       labelspacing=0.2, borderpad=0.2, columnspacing=0.8)

    return fig


def plot_multi(datasets, data_arrays, ls_arrays, filenames, args, nrows=2, ncols=2):
    """Multi-file colormesh subplot grid with a shared colorbar."""
    varname = args.var
    n = len(datasets)
    fs = BASE_FONT

    # Global vmin/vmax across all datasets
    vmin = args.vmin[0] if args.vmin is not None else min(np.nanmin(da.values) for da in data_arrays)
    vmax = args.vmax[0] if args.vmax is not None else max(np.nanmax(da.values) for da in data_arrays)

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

    # Detect vertical coordinate from the first file
    ds_peek = xr.open_dataset(args.ncfile[0])
    if "pressure" in ds_peek.coords:
        vert_coord = "pressure"
    elif "altitude" in ds_peek.coords:
        vert_coord = "altitude"
    else:
        print("ERROR: Dataset has neither 'altitude' nor 'pressure' coordinate.")
        sys.exit(1)

    vert_arg = args.pressure if vert_coord == "pressure" else args.alt

    if args.mode is None or vert_arg is None:
        if args.mode is None:
            print("ERROR: -mode is required.")
            if "mode_lat" in ds_peek.coords:
                print(f"Available lat-dependent modes: {ds_peek.mode_lat.values}")
            if "mode_point" in ds_peek.coords:
                print(f"Available point modes:         {ds_peek.mode_point.values}")
        if vert_arg is None:
            if vert_coord == "pressure":
                print("ERROR: -pressure is required (dataset uses pressure levels).")
                print(f"Available pressure levels (Pa): {ds_peek.pressure.values}")
            else:
                print("ERROR: -alt is required.")
                print(f"Available altitudes (km): {ds_peek.altitude.values}")
        ds_peek.close()
        sys.exit(1)
    ds_peek.close()

    varname = args.var
    mode = args.mode
    vert_levels = vert_arg  # always a list

    # Detect and validate ratio vars (e.g. "O/CO2")
    is_ratio = "/" in varname
    if is_ratio:
        ratio_parts = varname.split("/", 1)
        ds_check = xr.open_dataset(args.ncfile[0])
        coord_names = {"time", "altitude", "pressure", "latitude", "mode_lat", "mode_point",
                       "Ls", "nfiles", "average", "year", "sol"}
        available = sorted({
            v[:-6] if v.endswith("_point") else v
            for v in list(ds_check.data_vars) + list(ds_check.coords)
            if v not in coord_names
        })
        ds_check.close()
        bad = [p for p in ratio_parts if p not in available]
        if bad:
            print(f"ERROR: Variable(s) not valid for ratio: {bad}")
            print(f"Available variables: {available}")
            sys.exit(1)

    # Load all (file, level) combinations; detect mode type from first entry
    all_loaded = {}
    for f in args.ncfile:
        for vl in vert_levels:
            if is_ratio:
                num, denom = ratio_parts
                ds_n, da_n, ls, lat_dep = load_data(f, num,   mode, vl, vert_coord)
                _,    da_d, _,  _       = load_data(f, denom, mode, vl, vert_coord)
                all_loaded[(f, vl)] = (ds_n, da_n / da_d, ls, lat_dep)
            else:
                all_loaded[(f, vl)] = load_data(f, varname, mode, vl, vert_coord)

    lat_dependent = all_loaded[(args.ncfile[0], vert_levels[0])][3]

    if lat_dependent:
        # Lat-dependent: multiple levels not supported; use first only
        if len(vert_levels) > 1:
            print("WARNING: Multiple levels not supported for lat-dependent modes. Using first.")
        vl = vert_levels[0]
        if len(args.ncfile) == 1:
            ds, da, ls, _ = all_loaded[(args.ncfile[0], vl)]
            fig = plot_single(ds, da, ls, True, args)
        else:
            datasets    = [all_loaded[(f, vl)][0] for f in args.ncfile]
            data_arrays = [all_loaded[(f, vl)][1] for f in args.ncfile]
            ls_arrays   = [all_loaded[(f, vl)][2] for f in args.ncfile]
            lat_flags   = [all_loaded[(f, vl)][3] for f in args.ncfile]
            if not all(lat_flags):
                print("ERROR: Mixed lat-dependent and point modes across files — cannot combine.")
                sys.exit(1)
            fig = plot_multi(datasets, data_arrays, ls_arrays, args.ncfile, args)
    else:
        # Point mode: support any number of files and levels
        data_by_vert = []
        for vl in vert_levels:
            data_arrays = [all_loaded[(f, vl)][1] for f in args.ncfile]
            ls_arrays   = [all_loaded[(f, vl)][2] for f in args.ncfile]
            data_by_vert.append((vl, data_arrays, ls_arrays))
        fig = plot_point(data_by_vert, args.ncfile, args, vert_coord)

    if args.show:
        plt.show()
    else:
        safe_var = varname.replace("/", "_over_")
        if vert_coord == "altitude":
            vert_str = "_".join(str(int(v)) for v in vert_levels) + "km"
        else:
            vert_str = "_".join(f"{v:.2e}" for v in vert_levels) + "Pa"
        outfile = f"{safe_var}_{mode}_{vert_str}.png"
        plt.savefig(outfile, dpi=150)
        print(f"\nSaved: {outfile}")

if __name__ == "__main__":
    main()
