#!/usr/bin/env python3

import argparse
import sys
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import marstiming
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

def parse_args():

    parser = argparse.ArgumentParser(
        add_help=False,
        description="""
        Plot a reduced Phase 1 climatology variable from M-GITM.

        Usage modes:
        • script.py file.nc -h        → show dataset info
        • script.py -h                → show script help
        • script.py file.nc -var ...  → generate plot
        """ 
    )

    parser.add_argument("ncfile", nargs="?", help="Reduced NetCDF file")

    parser.add_argument("-var", help="Variable name")
    parser.add_argument("-mode", help="Mode (e.g., global, lt14, subsolar)")
    parser.add_argument("-alt", type=float, help="Altitude in km")
    parser.add_argument("--show", action="store_true",
                        help="Show plot instead of saving")
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for color scale"
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for color scale"
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

  Plot:
    script.py TEST_MY24_reduced.nc -var Temperature -mode global -alt 200
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
    # Case 2: file.nc -h → show dataset info
    # -------------------------------------------------
    if args.help and args.ncfile:
        ds = xr.open_dataset(args.ncfile)
        print_dataset_info(ds)
        sys.exit()

    # -------------------------------------------------
    # Case 3: Plotting mode
    # -------------------------------------------------
    if not args.ncfile or not args.var or not args.mode or args.alt is None:
        print("ERROR: Must specify ncfile, -var, -mode, and -alt.")
        print_script_help()
        sys.exit()

    ds = xr.open_dataset(args.ncfile)

    varname = args.var
    mode = args.mode
    altitude = args.alt

    if "mode_lat" in ds.coords and mode in ds.mode_lat.values:
        da = ds[varname].sel(mode_lat=mode)
        lat_dependent = True

    elif "mode_point" in ds.coords and mode in ds.mode_point.values:
        da = ds[varname + "_point"].sel(mode_point=mode)
        lat_dependent = False

    else:
        raise ValueError(f"Mode '{mode}' not found in dataset.")

    da = da.sel(altitude=args.alt)

    # ------------------------------------------------------------
    # LATITUDE-DEPENDENT MODES
    # ------------------------------------------------------------
    if lat_dependent:

        fig, axes = plt.subplots(
            2, 1,
            figsize=(10, 8),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.5, 3]}
        )

        # --- Top panel: latitude mean ---
        da_mean = da.mean("latitude")

        axes[0].plot(
            ds.Ls.values,
            da_mean.values,
            label='Latitude average'
        )

        axes[0].set_ylabel(varname)
        axes[0].legend(frameon=False)

        if args.vmin is not None or args.vmax is not None:
            vmin = args.vmin
            vmax = args.vmax
        else:
            vmin = None
            vmax = None
        # --- Bottom panel: latitude vs Ls ---
        im = axes[1].pcolormesh(
            ds.Ls.values,
            ds.latitude.values,
            da.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax
        )

        axes[1].set_ylabel("Latitude")
        axes[1].set_xlabel("Solar Longitude (Ls)")

        fig.colorbar(im,ax=axes[1],label=varname, pad=0.01)

    else:
        # ------------------------------------------------------------
        # POINT MODES
        # ------------------------------------------------------------
        ax.plot(ds.Ls.values, da.values, label=mode)
        ax.set_ylabel(varname)
        ax.set_xlabel("Solar Longitude (Ls)")
        # ax.set_title(f"{mode} at {altitude} km")



    if args.show:
        plt.show()
    else:
        outfile = f"{varname}_{mode}_{int(altitude)}km.png"
        plt.savefig(outfile, dpi=150)
        print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()