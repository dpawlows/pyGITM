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

    parser.add_argument(
        "-serial",
        action="store_true",
        help="Run process_batch serially (debug mode)"
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
    serial = args.serial

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
            verbose=False,
            serial=serial
        )

        times = np.array([entry['time'] for entry in data])
        ls_vals = np.array([getMarsSolarGeometry(t).ls for t in times])

        lat = data[0].get('lat', None)
        if lat is None:
            # Subsolar or SZA mode there is no latitude dimension
            lat = np.array([np.nan])
            single_location = True
        else:
            single_location = False

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
                # (alt,) → create latitude dimension of size 1
                arr = np.array([
                    entry[var_idx][alt_indices][None, :]
                    for entry in data
                ])

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

    lat_modes = []
    point_modes = []

    for m in DEFAULT_MODES.keys():
        sample = mode_data[m]["vars"][REQUIRED_VARS[0]]
        if sample.shape[1] == 1:
            point_modes.append(m)
        else:
            lat_modes.append(m)

    reference_mode = lat_modes[0] if lat_modes else point_modes[0]
    time_coord = mode_data[reference_mode]["time"]
    if lat_modes:
        latitude = mode_data[lat_modes[0]]["lat"]
    else:
        # Only point modes exist
        latitude = np.array([np.nan])
    altitude = np.array(altitudes_km)

    ds_vars_lat = {}

    for var_name in REQUIRED_VARS:
        clean_name = clean_varname(var_name, netcdf_safe=True)

        stacked = np.stack(
            [mode_data[m]["vars"][var_name] for m in lat_modes],
            axis=-1
        )

        ds_vars_lat[clean_name] = (
            ["time", "latitude", "altitude", "mode_lat"],
            stacked
        )

    for var_name in REQUIRED_VARS:
        clean_name = clean_varname(var_name, netcdf_safe=True)

        stacked = np.stack(
            [mode_data[m]["vars"][var_name] for m in lat_modes],
            axis=-1
        )

        ds_vars_lat[clean_name] = (
            ["time", "latitude", "altitude", "mode_lat"],
            stacked
        )

    ds_vars_point = {}

    for var_name in REQUIRED_VARS:
        stacked = np.stack(
            [mode_data[m]["vars"][var_name][:, 0, :] for m in point_modes],
            axis=-1
        )
        clean_name = clean_varname(var_name, netcdf_safe=True)

        ds_vars_point[clean_name + "_point"] = (
            ["time", "altitude", "mode_point"],
            stacked
        )

    ds = xr.Dataset(
    data_vars={**ds_vars_lat, **ds_vars_point},
    coords=dict(
        time=("time", time_coord),
        Ls=("time", mode_data[reference_mode]["Ls"]),
        latitude=("latitude", latitude),
        altitude=("altitude", altitude),
        mode_lat=("mode_lat", lat_modes),
        mode_point=("mode_point", point_modes)
        ),
        attrs=dict(
            case_name=case_name,
            altitudes_km=str(altitudes_km),
            modes_lat=str(lat_modes),
            modes_point=str(point_modes),
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

    ds.to_netcdf(outfile, engine="netcdf4", encoding=encoding)

    print("Done.")


if __name__ == "__main__":
    main()