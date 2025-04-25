import re 
from collections import defaultdict
import pandas as pd 
import concurrent.futures
from tqdm import tqdm 
import os 
from datetime import datetime
import marstiming 
import numpy as np 
from functools import partial
import multiprocessing

class FileParsingError(Exception):
    """Custom exception for header parsing errors."""
    pass

def readMarsGRAM(file,vars,smin=None,smax=None,loc=None,zonal=False,lsBinWidth=None,oco2=False):
    try: 
        local_vars = vars.copy() #vars is modified to accomodate oco2

        header = readASCIIheader(file)
        filename = os.path.basename(file)

        temp_df = pd.read_csv(file,
                    delim_whitespace=True,  # matches loadtxt's default behavior
                    skiprows=header['skiprows'],
                    header=None,            # since your file has no header row for data
                    usecols=local_vars)
                
        data_array = temp_df.values  # convert entire DataFrame to NumPy array
        reshaped = data_array.reshape((header['nlons'], header['nlats'], header['nalts'], -1))

        if oco2:
            #We do this before averaging as the O/CO2 is not linear
            nvars = reshaped.shape[-1]
            expanded = np.zeros((*reshaped.shape[:-1], nvars + 1))
            expanded[..., :nvars] = reshaped
            reshaped = expanded
            #Add oco2 ratio to the reshaped array
            o_header_index = header['vars'].index('[O]')
            co2_header_index = header['vars'].index('[CO$_2$]')
            try:
                #map from header['vars'] index to the corresponding position in vars
                o_reshaped_index = local_vars.index(o_header_index)  
                co2_reshaped_index = local_vars.index(co2_header_index)
            except ValueError:
                raise ValueError("O or CO2 was not selected in -var and is needed to calculate O/CO2.")

            O = reshaped[:, :, :, o_reshaped_index]
            CO2 = reshaped[:, :, :, co2_reshaped_index]

            # Safely calculate the ratio
            oco2_ratio = np.where(CO2 != 0, O / CO2, np.nan)

            # Append it to reshaped along new last dimension
            with np.errstate(divide='ignore', invalid='ignore'):
                reshaped[..., -1] = O / CO2
            reshaped[..., -1][CO2 == 0] = np.nan

            new_index = max(local_vars) + 1
            local_vars.append(new_index)
            header['vars'].append('O/CO$_2$')

        lon = reshaped[:, 0, 0, 0]  
        lat = reshaped[0, :, 0, 1]
        alt = reshaped[0, 0, :, 2]
        time = datetime.strptime(filename[-17:-4],"%y%m%d_%H%M%S")  

        # Need to calculate sza at all points on the grid and save to the df
        timedata = marstiming.getMarsSolarGeometry(time)
        Lon,Lat = np.meshgrid(lon,lat,indexing='ij') #to conform to typical GITM (lon,lat) structure
        sza = marstiming.getSZAfromTime(timedata,Lon,Lat)
        
        result = {'time':time,
        'alt': alt,
        }

        ls_bin = None 
        if lsBinWidth is not None:
            ls_bin = int(timedata.ls // lsBinWidth) * lsBinWidth
            result['ls_bin'] = ls_bin

        for i, var_index in enumerate(local_vars[3:]):
            var_data = reshaped[:, :, :, i + 3]  # (lon, lat, alt)

            # Only average if SZA bounds provided
            if smin is not None and smax is not None:
                sza_mask = (sza >= smin) & (sza <= smax)
                profile = []

                for k in range(header['nalts']):
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
                
                result[var_index] = reshaped[:, :, :, i + 3]  # shape: (lon, lat, alt)

 
        del temp_df, data_array, reshaped  # help GC free memory early

        return result 
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_batch(files, vars,smin=None,smax=None,zonal=False,lsBinWidth=None, oco2=False,max_workers=None):
    """Function to process a batch of files in parallel."""
    
    if len(vars) < 4:
        raise ValueError("Expected at least 4 variable indices: lon, lat, alt, and 1+ data var")

    reader = partial(readMarsGRAM, vars=vars, smin=smin, smax=smax,zonal=zonal,lsBinWidth=lsBinWidth,oco2=oco2,)
    
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
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

def readASCIIheader(file):
    nlats = nlons = nalts = None
    linecount = 0
    started = False
    vars = None 
    with open(file,'r') as f:

        for linecount, line in enumerate(f,start=1):
            if not line.strip():
                continue

            if "Number of Latitude points:" in line:
                nlats = int(line.split(":")[1].strip())
            if "Number of Longitude points:" in line:
                nlons = int(line.split(":")[1].strip())
            if "Number of Altitude points:" in line:
                nalts = int(line.split(":")[1].strip())

            if 'Longitude' in line and 'Latitude' in line and \
                'Altitude'in line:
                vars = re.split(r'\s{2,}', line.strip())

            if '#START' in line:
                started = True 
                break 

    if not started:
        raise FileParsingError("No '#START' line found in file.")
    if not (nlats and nlons and nalts):
        raise FileParsingError("Missing grid size information in file header")

    if vars is None:
        raise FileParsingError("Missing variable names in the header")

    return {'nlons':nlons,
        'nlats':nlats,
        'nalts':nalts,
        'vars':vars,
        'skiprows': linecount,
        } 
    

