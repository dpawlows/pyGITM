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

def readMarsGITM(file, vars,smin=None,smax=None,loc=None,zonal=False,lsBinWidth=None,oco2=False,verbose=False):

    try: 
        filename = os.path.basename(file)
        time = datetime.strptime(filename[-17:-4],"%y%m%d_%H%M%S")  

        if file.endswith('bin'):
            data = gr.read_gitm_one_file(file, vars_to_read=vars)
            
            # We are going to get rid of the ghost cells here:
            for key in list(data.keys()):
                if isinstance(key, int):  # only process numerical keys, skip 'vars', 'time', etc
                    arr = data[key]
                    # Remove ghost cells: slice 2:-2 in each dimension
                    data[key] = arr[2:-2, 2:-2, 2:-2]
            
            alt = data[2][0, 0, :]/1000.
            data[0] = data[0]*rtod
            data[1] = data[1]*rtod

        else:
            data = gr.read_gitm_ascii_onefile(file, vars_to_read=vars)
            alt = data[2][0, 0, :]
        
        data['vars'] = [gr.clean_varname(v.decode('ascii').strip()) if isinstance(v, bytes) \
                 else gr.clean_varname(v.strip()) for v in data['vars']]

        lon = data[0][:, 0, 0]  
        lat = data[1][0, :, 0]

        # Calculate various geometry parameters
        timedata = marstiming.getMarsSolarGeometry(time)
        Lon,Lat = np.meshgrid(lon,lat,indexing='ij') #to conform to typical GITM (lon,lat) structure
        sza = marstiming.getSZAfromTime(timedata,Lon,Lat)
        local_time_1d =  (timedata.MTC + lon * deg_to_hours) % 24 
        
        # Determine processing mode
        mode = 'full_3D'
        lt_target = None
        lt_width = 1
        half_width = lt_width / 2.0

        if smin is not None and smax is not None:
            mode = 'sza_average'
        elif isinstance(zonal, str):
            if zonal.lower() == 'subsolar':
                mode = 'subsolar'

            else:
                try:
                    lt_target = int(zonal)
                    mode = 'local_time_average'
                except ValueError:
                    pass  # ignore invalid strings
        elif isinstance(zonal, int) and zonal != False:
            mode = 'local_time_average'
            lt_target = zonal

        #Prepare result container
        result = {
            'time':time,
            'alt': alt,
            'sol':timedata.sol,
            'year':timedata.year,
            'MTC':timedata.MTC,
            'Ls':timedata.ls
        }
        if lsBinWidth is not None:
            ls_bin = int(timedata.ls // lsBinWidth) * lsBinWidth
            result['ls_bin'] = ls_bin

        # Add O/CO2 if requested        
        local_vars = vars.copy() #vars is modified to accomodate oco2
        if oco2:
            # Calculate O/CO2
            # We do this before averaging as the O/CO2 is not linear
            varnames = data["vars"]
            try:
                #map from header['vars'] index to the corresponding position in vars
                o_index = varnames.index('[O]')  
                co2_index = varnames.index('[CO_2]')

            except ValueError:
                raise ValueError("O or CO2 was not selected in -var and is needed to calculate O/CO2.")

            O = np.asarray(data[o_index])
            CO2 = np.asarray(data[co2_index])

            # Append it to reshaped along new last dimension
            with np.errstate(divide='ignore', invalid='ignore'):
                oco2_ratio = O / CO2
                oco2_ratio[CO2 == 0] = np.nan

            new_index = max(local_vars) + 1
            data[new_index] = oco2_ratio
            local_vars.append(new_index)
            data["vars"].append('O/CO$_2$')

        # Process variables based on input mode
        for var_index in local_vars[3:]:  # skipping lon, lat, alt
            var_data = data[var_index]  # shape (lon, lat, alt)

            if mode == 'sza_average':
                result['lat'] = lat  # Save lat array for consistency if needed
                sza_mask = (sza >= smin) & (sza <= smax)
                profile = []

                for k in range(len(alt)):
                    values = var_data[:, :, k][sza_mask]
                    profile.append(np.nan if values.size == 0 else np.nanmean(values))

                result[var_index] = np.array(profile)
            
            elif mode == 'local_time_average':
                result['lat'] = lat
                dlt = np.abs((local_time_1d - lt_target + 12) % 24 - 12)
                lt_mask = dlt <= half_width
                good_lon_idx = np.where(lt_mask)[0]
                if good_lon_idx.size > 0:
                    selected = var_data[good_lon_idx, :, :]
                    result[var_index] = np.nanmean(selected, axis=0)
                else:
                    result[var_index] = np.full((len(lat), len(alt)), np.nan)

            elif mode == "subsolar":
                sslon = timedata.subSolarLon
                sslat = timedata.solarDec

                # Find the closest lon/lat grid points
                ilon = np.argmin(np.abs(lon - sslon))
                ilat = np.argmin(np.abs(lat - sslat))

                profile = var_data[ilon, ilat, :]
                result[var_index] = profile
                result['lat'] = lat  # Save lat array for consistency if needed

            else:
                # Include grid info and full variable arrays
                result['lon'] = lon
                result['lat'] = lat
                result['sza'] = sza
                result[var_index] = var_data

        if verbose:
            print(f"[readMarsGITM] {filename}: Sol={result['sol']:.1f}, Ls={result['Ls']:.1f}, MTC={result['MTC']:.2f}")

        del data
        gc.collect()

        return result 
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def zonal_fixed_ave(raw_results,zonal,lsBinWidth = None):
    '''Perform zonal average at a fixed Mars location.'''

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
    verbose=False):
    """Function to process a batch of files in parallel."""
    
    if len(vars) < 4:
        raise ValueError("Expected at least 4 variable indices: lon, lat, alt, and 1+ data var")

    reader = partial(readMarsGITM, vars=vars, smin=smin, smax=smax,zonal=zonal,lsBinWidth=lsBinWidth,oco2=oco2,
        verbose=verbose)

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
        return zonal_fixed_ave(raw_results,zonal,lsBinWidth=lsBinWidth)

    else:
        # No binning — return raw list
        final_data = [r for r in raw_results if r is not None]

    return final_data


    

