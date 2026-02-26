#!/usr/bin/env python

import numpy as np
import xarray as xr
import argparse
import time
from gitm_routines import *
from gitmconcurrent import *
from marstiming import getMarsSolarGeometry

# --------------------------------------------------
# Default configuration
# --------------------------------------------------

DEFAULT_ALTITUDES = [100, 135, 150, 200]

DEFAULT_MODES = {
    "global": dict(zonal="global"),
    "lt04": dict(zonal="4"),
    "lt14": dict(zonal="14"),
    "subsolar": dict(zonal="subsolar"),
    "sza_day": dict(zonal="sza", smin=0, smax=30),
    "sza_night": dict(zonal="sza", smin=150, smax=180),
}

REQUIRED_VARS = [
    'Temperature', 'Rho', '[O]', '[CO!D2!N]', 'V!Dn!N(east)', 'V!Dn!N(north)', 'V!Dn!N(up)', '[e-]'
]

# --------------------------------------------------
# Argument Parser
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 reducer for annual M-GITM simulations."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Input GITM bin files"
    )

    parser.add_argument(
        "-case",
        required=True,
        help="Case name (e.g., MY24_case1). Necessary so label is part of the dataset."
    )

    parser.add_argument(
        "-alts",
        nargs="+",
        type=float,
        default=DEFAULT_ALTITUDES,
        help="Altitudes (km) to extract"
    )

    parser.add_argument(
        "-workers",
        type=int,
        default=16,
        help="Number of parallel workers"
    )

    parser.add_argument(
        "-output",
        default=None,
        help="Output NetCDF filename (optional)"
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()
    files = sorted(args.files, key=parse_filename)

    altitudes_km = args.alts
    case_name = args.case
    max_workers = args.workers

    print(f"\nCase: {case_name}")
    print(f"Altitudes: {altitudes_km}")
    print(f"Workers: {max_workers}")

    header = read_gitm_header(files[:1])

    # Map required variable names to indices
    var_indices = []
    for name in REQUIRED_VARS:
        if name not in header['vars']:
            raise ValueError(f"Variable '{name}' not found in header.")
        var_indices.append(header['vars'].index(name))

    vars_for_read = [0,1,2] + var_indices

    mode_data = {}

    for mode_name, mode_config in DEFAULT_MODES.items():

        print(f"\nProcessing mode: {mode_name}")

        zonal = mode_config.get("zonal", None)
        smin = mode_config.get("smin", None)
        smax = mode_config.get("smax", None)

        data = process_batch(
            files,
            vars_for_read,
            max_workers=max_workers,
            zonal=zonal,
            smin=smin,
            smax=smax,
            verbose=False
        )

        times = np.array([entry['time'] for entry in data])
        ls_vals = np.array([getMarsSolarGeometry(t).ls for t in times])

        lat = data[0]['lat']
        alt = data[0]['alt']

        # Find altitude indices
        alt_indices = [
            np.argmin(np.abs(alt - a)) for a in altitudes_km
        ]

        var_arrays = {}

        for var_name, var_idx in zip(REQUIRED_VARS, var_indices):

            print(f"  Extracting {var_name}")

            sample_var = data[0][var_idx]

            if sample_var.ndim == 2:
                # (lat, alt)
                arr = np.array([
                    entry[var_idx][:, alt_indices]
                    for entry in data
                ])
            else:
                # (alt,) → expand to (lat, alt) with NaNs
                arr = np.array([
                    np.full((len(lat), len(alt_indices)), np.nan)
                    for _ in data
                ])

                for i, entry in enumerate(data):
                    arr[i, :, :] = np.tile(
                        entry[var_idx][alt_indices],
                        (len(lat), 1)
                    )

            var_arrays[var_name] = arr

        mode_data[mode_name] = {
            "vars": var_arrays,
            "time": times,
            "Ls": ls_vals,
            "lat": lat
        }

        del data
        gc.collect()

    # --------------------------------------------------
    # Construct Dataset
    # --------------------------------------------------

    print("\nConstructing xarray Dataset...")

    modes = list(DEFAULT_MODES.keys())
    time_coord = mode_data[modes[0]]["time"]
    latitude = mode_data[modes[0]]["lat"]
    altitude = np.array(altitudes_km)

    ds_vars = {}

    for var_name in REQUIRED_VARS:

        stacked = np.stack(
            [mode_data[m]["vars"][var_name] for m in modes],
            axis=-1
        )  # shape (time, lat, alt, mode)

        ds_vars[var_name] = (
            ["time", "latitude", "altitude", "mode"],
            stacked
        )

    ds = xr.Dataset(
        data_vars=ds_vars,
        coords=dict(
            time=("time", time_coord),
            Ls=("time", mode_data[modes[0]]["Ls"]),
            latitude=("latitude", latitude),
            altitude=("altitude", altitude),
            mode=("mode", modes)
        ),
        attrs=dict(
            case_name=case_name,
            altitudes_km=str(altitudes_km),
            modes=str(modes),
            created=time.strftime("%Y-%m-%d"),
            description="Annual M-GITM reduced dataset"
        )
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    outfile = args.output if args.output else f"{case_name}_reduced.nc"

    print(f"\nSaving {outfile}")

    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds.data_vars
    }

    ds.to_netcdf(outfile, encoding=encoding)

    print("Done.")


if __name__ == "__main__":
    main()