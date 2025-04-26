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


def readMarsGITM(file, vars,smin=None,smax=None,loc=None,zonal=False,lsBinWidth=None,oco2=False):
        
    try: 
        filename = os.path.basename(file)
        time = datetime.strptime(filename[-17:-4],"%y%m%d_%H%M%S")  

        if file.endswith('bin'):
            data = gr.read_gitm_one_file(file, vars=-1)
        else:
            data = gr.read_gitm_ascii_onefile(file, vars)
        
        lon = data[0][:, 0, 0]  
        lat = data[1][0, :, 0]
        alt = data[2][0, 0, :]

        # Calculate SZA
        timedata = marstiming.getMarsSolarGeometry(time)
        Lon,Lat = np.meshgrid(lon,lat,indexing='ij') #to conform to typical GITM (lon,lat) structure
        sza = marstiming.getSZAfromTime(timedata,Lon,Lat)

        result = {
            'time':time,
            'alt': alt,
        }

        if lsBinWidth is not None:
            ls_bin = int(timedata.ls // lsBinWidth) * lsBinWidth
            result['ls_bin'] = ls_bin

        local_vars = vars.copy() #vars is modified to accomodate oco2

        if oco2:
            # Calculate O/CO2
            # We do this before averaging as the O/CO2 is not linear
            varnames = data["vars"]
            try:
                #map from header['vars'] index to the corresponding position in vars
                o_index = varnames.index('[O]')  
                co2_index = varnames.index('[CO$_2$]')
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
       
        for var_index in local_vars[3:]:  # skipping lon, lat, alt
            var_data = data[var_index]  # shape (lon, lat, alt)

            # Only average if SZA bounds provided
            if smin is not None and smax is not None:
                sza_mask = (sza >= smin) & (sza <= smax)
                profile = []

                for k in range(len(alt)):
                    values = var_data[:, :, k][sza_mask]
                    profile.append(np.nan if values.size == 0 else np.nanmean(values))

                result[var_index] = np.array(profile)
            elif zonal:

                result['lat'] = lat
                # Compute mean over longitude axis (axis 0)
                result[var_index] = np.mean(var_data, axis=0)  # shape (lat, alt)
            else:
                # Include grid info and full variable arrays
                result['lon'] = lon
                result['lat'] = lat
                result['sza'] = sza
                
                result[var_index] = var_data

        return result 
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_batch(files, vars,smin=None,smax=None,zonal=False,lsBinWidth=None, oco2=False,max_workers=None):
    """Function to process a batch of files in parallel."""
    
    if len(vars) < 4:
        raise ValueError("Expected at least 4 variable indices: lon, lat, alt, and 1+ data var")

    reader = partial(readMarsGITM, vars=vars, smin=smin, smax=smax,zonal=zonal,lsBinWidth=lsBinWidth,oco2=oco2,)
    
    # --- Auto-tune max_workers ---
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()

        # Guess if on SSD (simple, fast test)
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
        raw_results = list(tqdm(executor.map(reader,files,chunksize=1),
            total=len(files),desc="Processing files",unit="file"
                ))

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
    else:
        # No binning — return raw list
        final_data = [r for r in raw_results if r is not None]

    return final_data


    

