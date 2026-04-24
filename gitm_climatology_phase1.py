#!/usr/bin/env python

import numpy as np
import xarray as xr
import argparse
import time
from gitm_routines import *
from gitmconcurrent import *
from marstiming import getMarsSolarGeometry

boltzmann = 1.380649e-23


def _interp_to_pressure_levels(var_profile, pressure_profile, target_levels):
    """Interpolate a 1-D profile to target pressure levels in log-pressure space.

    pressure_profile decreases with altitude index, so flip before calling np.interp.
    Returns NaN outside the range of the pressure profile.
    """
    log_p = np.log(pressure_profile[::-1])   # increasing
    var_flip = var_profile[::-1]
    log_targets = np.log(np.asarray(target_levels, dtype=float))
    return np.interp(log_targets, log_p, var_flip, left=np.nan, right=np.nan)


def _extract_at_pressure(entry, var_idx, pressure_density_indices,
                         pressure_temp_index, pressure_levels):
    """Return variable from one entry interpolated to pressure_levels.

    Shape returned:
      - (nlat, n_pres) for 2-D (lat, alt) entries
      - (1, n_pres)    for 1-D (alt,) entries  — matches the [None, :] alt convention
    """
    temp = entry[pressure_temp_index]
    number_density = np.zeros_like(temp, dtype=float)
    for idens in pressure_density_indices:
        number_density += entry[idens]
    pressure = number_density * boltzmann * temp

    var_data = entry[var_idx]
    n_pres = len(pressure_levels)

    if var_data.ndim == 2:
        n_lat = var_data.shape[0]
        result = np.empty((n_lat, n_pres))
        for ilat in range(n_lat):
            result[ilat] = _interp_to_pressure_levels(
                var_data[ilat], pressure[ilat], pressure_levels
            )
    else:
        result = _interp_to_pressure_levels(var_data, pressure, pressure_levels)[None, :]

    return result

# --------------------------------------------------
# Default configuration
# --------------------------------------------------

DEFAULT_ALTITUDES = [100, 135, 150, 200]

DEFAULT_MODES = {
    "global": dict(zonal="global", point=False),
    "lt04": dict(zonal="4",point=False),
    "lt14": dict(zonal="14",point=False),
    "subsolar": dict(zonal="subsolar",point=True),
    "antisolar": dict(zonal="antisolar",point=True),
    "sza_day": dict(zonal="sza", smin=0, smax=30,point=True),
    "sza_night": dict(zonal="sza", smin=150, smax=180,point=True),

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
        "-pressure",
        nargs="+",
        type=float,
        default=None,
        help="Pressure level(s) in Pa to extract instead of altitudes"
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

    parser.add_argument(
        "-average",
        type=str,
        default=None,
        choices=["sol", "ls", "none"],
        help="Averaging mode: 'sol' for sol averaging, 'ls' for Ls bin averaging, 'none' for no averaging."
    )

    parser.add_argument(
        "-lsBinWidth",
        type=float,
        default=None,
        help="Ls bin width for averaging. If None, average by sol."
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
    pressure_mode = args.pressure is not None
    pressure_levels = args.pressure

    print(f"\nCase: {case_name}")
    if pressure_mode:
        print(f"Pressure levels (Pa): {pressure_levels}")
    else:
        print(f"Altitudes: {altitudes_km}")
    print(f"Workers: {max_workers}")

    header = read_gitm_header(files[:1])

    # Map required variable names to indices
    var_indices = []
    for name in REQUIRED_VARS:
        if name not in header['vars']:
            raise ValueError(f"Variable '{name}' not found in header.")
        var_indices.append(header['vars'].index(name))

    # the gitm reader returns entries keyed by the original header index, not by position
    # thus we use var_indices directly
    vars_for_read = [0,1,2] + var_indices

    pressure_density_indices = []
    pressure_temp_index = None
    if pressure_mode:
        try:
            pressure_temp_index = header['vars'].index('Temperature')
            rho_index = header['vars'].index('Rho')
        except ValueError as e:
            raise ValueError(f"Pressure mode requires Temperature and Rho variables: {e}")
        pressure_density_indices = [
            i for i in range(rho_index + 1, pressure_temp_index)
            if header['vars'][i].startswith('[') and header['vars'][i].endswith(']')
        ]
        if not pressure_density_indices:
            raise ValueError("No neutral density variables found between Rho and Temperature.")
        for idx in pressure_density_indices + [pressure_temp_index]:
            if idx not in vars_for_read:
                vars_for_read.append(idx)

    mode_data = {}

    for mode_name, mode_config in DEFAULT_MODES.items():

        print(f"\nProcessing mode: {mode_name}")

        zonal = mode_config.get("zonal", None)
        smin = mode_config.get("smin", None)
        smax = mode_config.get("smax", None)

        average = mode_config.get("average",False)

        data = process_batch(
            files,
            vars_for_read,
            max_workers=max_workers,
            zonal=zonal,
            smin=smin,
            smax=smax,
            verbose=False,
            serial=serial,
            average=args.average,
            lsBinWidth=args.lsBinWidth,
        )

        times, ls_vals, years, sols, nfiles = [], [], [], [], []
        for entry in data:
            times.append(entry['time'])
            ls_vals.append(entry['Ls']) 
            years.append(entry['year'])
            sols.append(entry['sol'])
            nfiles.append(entry['nfiles'])
        times   = np.array(times)
        ls_vals = np.array(ls_vals)
        years   = np.array(years)
        sols    = np.array(sols)
        nfiles = np.array(nfiles)

        lat = data[0].get('lat', None)
        if lat is None:
            # Subsolar or SZA mode there is no latitude dimension
            lat = np.array([np.nan])
            single_location = True
        else:
            single_location = False

        var_arrays = {}

        if pressure_mode:
            for var_name, var_idx in zip(REQUIRED_VARS, var_indices):
                print(f"  Extracting {var_name} at pressure levels")
                arr = np.array([
                    _extract_at_pressure(
                        entry, var_idx, pressure_density_indices,
                        pressure_temp_index, pressure_levels
                    )
                    for entry in data
                ])
                var_arrays[var_name] = arr
        else:
            alt = data[0]['alt']
            alt_indices = [np.argmin(np.abs(alt - a)) for a in altitudes_km]

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
            "year": years,
            "sol": sols,
            "lat": lat,
            "nfiles": nfiles,
        }
        
        del data
        gc.collect()

    # --------------------------------------------------
    # Construct Dataset
    # --------------------------------------------------

    print("\nConstructing xarray Dataset...")

    lat_modes   = [m for m, cfg in DEFAULT_MODES.items() if not cfg["point"]]
    point_modes = [m for m, cfg in DEFAULT_MODES.items() if cfg["point"]]


    reference_mode = lat_modes[0] if lat_modes else point_modes[0]
    expected = np.median(mode_data[reference_mode]["nfiles"])
    
    # Quick check to make sure there aren't any data that were averaged with low counts
    if args.average in ("sol", "ls"):
        low_count_threshold = expected * 0.5
        low = mode_data[reference_mode]["nfiles"] < low_count_threshold
        if np.any(low):
            print(f"[WARNING] {np.sum(low)} time steps have fewer than {low_count_threshold:.0f} files")
            print(f"  Min count: {mode_data[reference_mode]['nfiles'].min()}")

    time_coord = mode_data[reference_mode]["time"]
    if lat_modes:
        latitude = mode_data[lat_modes[0]]["lat"]
    else:
        # Only point modes exist
        latitude = np.array([np.nan])
    if pressure_mode:
        vert_coord_name = "pressure"
        vert_coord_vals = np.array(pressure_levels)
    else:
        vert_coord_name = "altitude"
        vert_coord_vals = np.array(altitudes_km)

    ds_vars_lat = {}

    for var_name in REQUIRED_VARS:
        clean_name = clean_varname(var_name, netcdf_safe=True)

        stacked = np.stack(
            [mode_data[m]["vars"][var_name] for m in lat_modes],
            axis=-1
        )

        ds_vars_lat[clean_name] = (
            ["time", "latitude", vert_coord_name, "mode_lat"],
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
            ["time", vert_coord_name, "mode_point"],
            stacked
        )


    ds_lat = xr.Dataset(
        data_vars=ds_vars_lat,
        coords={
            "time": ("time", time_coord),
            "Ls": ("time", mode_data[reference_mode]["Ls"]),
            "year": ("time", mode_data[reference_mode]["year"]),
            "sol": ("time", mode_data[reference_mode]["sol"]),
            "nfiles": ("time", mode_data[reference_mode]["nfiles"]),
            "latitude": ("latitude", latitude),
            vert_coord_name: (vert_coord_name, vert_coord_vals),
            "mode_lat": ("mode_lat", lat_modes),
        }
    )

    ds_point = xr.Dataset(
        data_vars=ds_vars_point,
        coords={
            "time": ("time", time_coord),
            vert_coord_name: (vert_coord_name, vert_coord_vals),
            "mode_point": ("mode_point", point_modes),
        }
    )

    ds = xr.merge([ds_lat, ds_point])
    ds.attrs = dict(case_name=case_name,
            altitudes_km=str(altitudes_km) if not pressure_mode else "N/A",
            pressure_levels_pa=str(pressure_levels) if pressure_mode else "N/A",
            modes_lat=str(lat_modes),
            modes_point=str(point_modes),
            average=str(args.average),
            lsBinWidth=str(args.lsBinWidth),
            created=time.strftime("%Y-%m-%d"),
            description="Annual M-GITM reduced dataset"
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
