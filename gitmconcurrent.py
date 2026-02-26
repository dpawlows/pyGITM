import re 
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm 
import os 
from datetime import datetime
import marstiming 
import gitm_routines as gr
import numpy as np 
from functools import partial
import multiprocessing
import gc

rtod = 180/np.pi
deg_to_hours = 1/15.0
def _time_from_filename(file):
    fname = os.path.basename(file)
    return datetime.strptime(fname[-17:-4], "%y%m%d_%H%M%S")

def _strip_ghost_cells(data):
    for k, v in list(data.items()):
        if isinstance(k, int):
            data[k] = v[2:-2, 2:-2, 2:-2]
    return data

def _select_mode(smin, smax, zonal):
    if smin is not None and smax is not None:
        return "sza_average", None
    if isinstance(zonal, str):
        zl = zonal.lower()

        if zl == "subsolar":
            return "subsolar", None

        if zl == "global":
            return "zonal_mean", None   # <-- NEW MODE

        try:
            return "local_time_average", int(zonal)
        except ValueError:
            pass
    if isinstance(zonal, int) and zonal:
        return "local_time_average", zonal
    return "full_3D", None

def _clean_varnames(data):
    data['vars'] = [
        gr.clean_varname(v.decode('ascii').strip()) if isinstance(v, bytes)
        else gr.clean_varname(v.strip())
        for v in data['vars']
    ]
    return data

def readMarsGITM(
    file, vars,
    smin=None, smax=None,
    loc=None, zonal=False,
    lsBinWidth=None,
    oco2=False,
    verbose=False
    ):

    try:
        time = _time_from_filename(file)

        # ---- Read data --------------------------------------------------
        if file.endswith("bin"):
            data = gr.read_gitm_one_file(file, vars_to_read=vars)

            alt = data[2][0, 0, :] / 1000.0
            data[0] *= rtod
            data[1] *= rtod

        else:
            data = gr.read_gitm_ascii_onefile(file, vars_to_read=vars)
            alt = data[2][0, 0, :]

        data = _clean_varnames(data)
        lon = data[0][:, 0, 0]
        lat = data[1][0, :, 0]

        # ---- Geometry ---------------------------------------------------
        timedata = marstiming.getMarsSolarGeometry(time)
        Lon, Lat = np.meshgrid(lon, lat, indexing="ij")
        sza = marstiming.getSZAfromTime(timedata, Lon, Lat)
        local_time = (timedata.MTC + lon * deg_to_hours) % 24

        # ---- Result container ------------------------------------------
        result = {
            "time": time,
            "alt": alt,
            "sol": timedata.sol,
            "year": timedata.year,
            "MTC": timedata.MTC,
            "Ls": timedata.ls,
        }

        if lsBinWidth is not None:
            result["ls_bin"] = int(timedata.ls // lsBinWidth) * lsBinWidth

        # ---- 1D case ----------------------------------------------------
        if "3D" not in file:
            result.update({
                "lon": lon,
                "lat": lat,
                "sza": sza,
            })
            for v in vars[3:]:
                result[v] = data[v]
            return result

        # ---- 3D case  --------------------------------------------
        data = _strip_ghost_cells(data)
        mode, lt_target = _select_mode(smin, smax, zonal)
        half_width = 0.5

        # ---- Optional O/CO2 --------------------------------------------
        local_vars = vars.copy()
        if oco2:
            names = data["vars"]
            try:
                o_idx = names.index("[O]")
                co2_idx = names.index("[CO_2]")
            except ValueError:
                raise ValueError("O and CO2 must be selected to compute O/CO2")

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = data[o_idx] / data[co2_idx]
                ratio[data[co2_idx] == 0] = np.nan

            new_idx = max(local_vars) + 1
            data[new_idx] = ratio
            data["vars"].append("O/CO$_2$")
            local_vars.append(new_idx)

        # ---- Variable processing ---------------------------------------
        for v in local_vars[3:]:
            var = data[v]

            if mode == "sza_average":
                prof = [
                    np.nanmean(var[:, :, k][(sza >= smin) & (sza <= smax)])
                    if np.any((sza >= smin) & (sza <= smax))
                    else np.nan
                    for k in range(len(alt))
                ]
                result[v] = np.array(prof)

            elif mode == "local_time_average":
                dlt = np.abs((local_time - lt_target + 12) % 24 - 12)
                mask = dlt <= half_width
                result[v] = (
                    np.nanmean(var[mask, :, :], axis=0)
                    if np.any(mask)
                    else np.full((len(lat), len(alt)), np.nan)
                )

            elif mode == "subsolar":
                ilon = np.argmin(np.abs(lon - timedata.subSolarLon))
                ilat = np.argmin(np.abs(lat - timedata.solarDec))
                result[v] = var[ilon, ilat, :]

            elif mode == "zonal_mean":
                # Average over longitude axis (axis=0)
                # Preserve latitude and altitude
                result[v] = np.nanmean(var, axis=0)

            else:
                result.update({
                    "lon": lon,
                    "lat": lat,
                    "sza": sza,
                    v: var,
                })

        if verbose:
            print(
                f"[readMarsGITM] {os.path.basename(file)}: "
                f"Sol={result['sol']:.1f}, "
                f"Ls={result['Ls']:.1f}, "
                f"MTC={result['MTC']:.2f}"
            )

        return result

    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def group_by_sol_average(raw_results,zonal,lsBinWidth = None):
    '''Perform average at a fixed Mars location.'''

    if not raw_results:
        return []

    if lsBinWidth is not None:
        # Bin by Ls
        binned = defaultdict(lambda: {'count': 0, 'times':[]})
        for result in raw_results:
            if result is None: continue
            ls_bin = result['ls_bin']
            binned[ls_bin]['times'].append(result['time'])

            if 'alt' not in binned[ls_bin]:
                binned[ls_bin]['alt'] = result['alt']
            if 'lat' not in binned[ls_bin] and 'lat' in result:
                binned[ls_bin]['lat'] = result['lat']

            for k, v in result.items():
                if isinstance(k, int):
                    if k not in binned[ls_bin]:
                        binned[ls_bin][k] = np.array(v, dtype=float)
                    else:
                        binned[ls_bin][k] += (v - binned[ls_bin][k]) / (binned[ls_bin]['count'] + 1)
            binned[ls_bin]['count'] += 1

        # Sort and convert to list
        final_data = []
        for ls_bin, bin_data in sorted(binned.items()):
            mean_time = bin_data['times'][0]
            if len(bin_data['times']) > 1:
                timestamps = [t.timestamp() for t in bin_data['times']]
                mean_time = datetime.fromtimestamp(np.mean(timestamps))

            entry = {'ls': ls_bin, 'alt': bin_data['alt'],'time':mean_time}
            if 'lat' in bin_data:
                entry['lat'] = bin_data['lat']
            
            for k in bin_data:
                if isinstance(k, int):
                    entry[k] = bin_data[k]
            final_data.append(entry)
        return final_data

    else:
            
        # Group entries by Mars Sol number
        grouped_by_yearsol = defaultdict(list)
        prev_sol = None
        prev_year = None

        for entry in raw_results:
            if entry is not None:
                my = int(entry['year'])
                sol = int(entry['sol'])
                 # Check for wrap-around in sol without year increment
                if prev_sol is not None and prev_year is not None:
                    if sol < 1 and prev_sol > 660:
                        if my <= prev_year:
                            print(f"[WARNING] Sol reset detected without Mars Year increment:")
                            print(f"  Previous: MY={prev_year}, Sol={prev_sol}, time={entry['time']}")
                            print(f"  Current:  MY={my}, Sol={sol}, time={entry['time']}")
                            breakpoint()

            grouped_by_yearsol[(my, sol)].append(entry)
            prev_sol = sol
            prev_year = my


        results = []

        for (my, sol) in sorted(grouped_by_yearsol.keys()):
            entries = grouped_by_yearsol[(my,sol)]
            timestamps = np.array([entry['time'].timestamp() for entry in entries])
            mean_time = datetime.fromtimestamp(np.mean(timestamps))

            # Assume all entries have the same lat/lon/alt grids
            sample = entries[0]
            lat = sample['lat']
            alt = sample['alt']

            nalt = alt.size

            vars_to_average = [k for k in sample.keys() if isinstance(k, int)]

            # Handle if lat is an array or a single value
            if isinstance(lat, np.ndarray):
                nlat = lat.size
            else:
                nlat = None  # single location case

            # Determine if the data is 1D or 2D
            var_sample = sample[vars_to_average[0]]
            if var_sample.ndim == 2:
                two_d = True
            else:
                two_d = False

            # Preallocate sums and counts
            sums = {}
            counts = {}
            for var_index in vars_to_average:
                if two_d:
                    sums[var_index] = np.zeros((nlat, nalt))
                    counts[var_index] = np.zeros((nlat, nalt))
                else:
                    sums[var_index] = np.zeros((nalt,))
                    counts[var_index] = np.zeros((nalt,))
                    
            for entry in entries:
                for var_index in vars_to_average:
                    var = entry[var_index]  # (nlon, nlat, nalt)
        
                    valid = np.isfinite(var)
                    sums[var_index][valid] += var[valid]
                    counts[var_index][valid] += 1  

            # Build result for this Sol
            sol_result = {'year':my,'sol': sol, 'lat': lat, 'alt': alt,'time': mean_time}
            if isinstance(lat, np.ndarray):
                sol_result['lat'] = lat

            for var_index in vars_to_average:
                with np.errstate(divide='ignore', invalid='ignore'):
                    avg = sums[var_index] / counts[var_index]
                    avg[counts[var_index] == 0] = np.nan

                sol_result[var_index] = avg

            results.append(sol_result)


    return results


def process_batch(files, vars,smin=None,smax=None,zonal=False,lsBinWidth=None, oco2=False,max_workers=None,
    verbose=False,serial=False):
    """Function to process a batch of files in parallel."""
    
    if len(vars) < 4:
        raise ValueError("Expected at least 4 variable indices: lon, lat, alt, and 1+ data var")

    reader = partial(readMarsGITM, vars=vars, smin=smin, smax=smax,zonal=zonal,lsBinWidth=lsBinWidth,oco2=oco2,
        verbose=verbose)

    # ------------------------------------------------------------
    # SERIAL PATH (debug-friendly)
    # ------------------------------------------------------------
    if serial:
        print(f"[process_batch] Running serially on {len(files)} files...")
        raw_results = [
            reader(f)
            for f in tqdm(files, desc="Processing files", unit="file")
        ]

    # ------------------------------------------------------------
    # PARALLEL PATH
    # ------------------------------------------------------------
    else:
        # --- Auto-tune max_workers ---
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()

            # Guess if on SSD 
            is_ssd = False
            try:
                diskstat = os.statvfs('.')
                if hasattr(diskstat, 'f_frsize') and hasattr(diskstat, 'f_bavail'):
                    # crude guess: SSDs tend to have much faster filesystem response
                    is_ssd = True
            except:
                pass  # can't detect, assume HDD

            if is_ssd:
                max_workers = min(cpu_count, 32)  # if SSD, use up to 32 workers
            else:
                max_workers = min(cpu_count // 2, 16)  # if HDD, be more conservative

        print(f"[process_batch] Using {max_workers} workers for {len(files)} files...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            raw_results = list(tqdm(executor.map(reader,files,chunksize=8),
                total=len(files),desc="Processing files",unit="file"
                    ))
    if zonal:
        return group_by_sol_average(raw_results,zonal,lsBinWidth=lsBinWidth)


    return [r for r in raw_results if r is not None]


    

