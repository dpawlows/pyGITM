#!/usr/bin/env python

import numpy as np 
from matplotlib import pyplot as pp 
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import sys 
import re 
from gitm_routines import * 
from gitmconcurrent import *
import pandas as pd 
import time 
import pickle

startTime = time.time()
def get_args(argv):

    filelist = []
    var = None
    verbose = False
    help = False
    alog = False
    mini = None 
    maxi = None 
    alt = None
    lon = None
    lat = None
    cut = None
    smin = None
    smax = None
    binwidth = None 
    oco2 = None
    serial = None
    zonal = None

    help = 0

    for arg in argv:

        IsFound = 0

        if (not IsFound):
            m = re.match(r'-var=(.*)',arg)
            if m:
                var = m.group(1)
                IsFound = 1

            m = re.match(r'-min=(.*)',arg)
            if m:
                mini = m.group(1)
                IsFound = 1

            m = re.match(r'-max=(.*)',arg)
            if m:
                maxi = m.group(1)
                IsFound = 1
            
            m = re.match(r'-h',arg)
            if m:
                help = True
                IsFound = 1
            
            m = re.match(r'-serial',arg)
            if m:
                serial = True
                IsFound = 1
            
            m = re.match(r'-subsolar',arg)
            if m:
                subsola = True
                IsFound = 1

            m = re.match(r'-oco2',arg)
            if m:
                oco2 = True
                IsFound = 1

            m = re.match(r'-alog',arg)
            if m:
                alog = True
                IsFound = 1

            m = re.match(r'-verbose',arg)
            if m:
                verbose = True
                IsFound = 1

            m = re.match(r'-cut=(.*)',arg)
            if m:
                cut = m.group(1)
                IsFound = 1

            m = re.match(r'-lat=(.*)',arg)
            if m:
                lat = int(m.group(1))
                IsFound = 1

            m = re.match(r'-lon=(.*)',arg)
            if m:
                lon = int(m.group(1))
                IsFound = 1

            m = re.match(r'-alt=(.*)',arg)
            if m:
                alt = int(m.group(1))
                IsFound = 1

            m = re.match(r'-smin=(.*)',arg)
            if m:
                smin = int(m.group(1))
                IsFound = 1   
            
            m = re.match(r'-smax=(.*)',arg)
            if m:
                smax = int(m.group(1))
                IsFound = 1

            m = re.match(r'-binls=(.*)',arg)
            if m:
                binwidth = int(m.group(1))
                IsFound = 1
            
            m = re.match(r'-zonal=(.*)',arg)
            if m:
                zonal = m.group(1)
                IsFound = 1

            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)
            
    args = {'filelist':filelist,
        'var':var,
        'help':help,
        'alog':alog,
        'cut':cut,
        'lat':lat,
        'lon':lon,
        'alt':alt,
        'smin':smin,
        'smax':smax,
        'binls':binwidth,
        'oco2':oco2,
        'serial':serial,
        'verbose':verbose,
        'zonal':zonal,
        'mini':mini,
        'maxi':maxi,
    }

    return args

def plotSZA(data):
    '''Plot sza as a function of lat and lon given a data from a single ascii'''

    lon = np.array(data['lon'])
    lat = np.array(data['lat'])
    sza = data['sza']
    # 1. Shift lon from [0,360] to [-180,180]
    lon_shifted = (lon + 180) % 360 - 180
    # 2. Sort longitudes and get sorting indices
    sorted_indices = np.argsort(lon_shifted)
    lon_sorted = lon_shifted[sorted_indices]
    # 3. Reorder SZA accordingly along the longitude axis (axis=1)
    sza_sorted = sza[:, sorted_indices]
    Lon, Lat = np.meshgrid(lon_sorted, lat)

    pp.figure(figsize=(10, 6))
    contour = pp.contourf(Lon, Lat, sza_sorted, levels=30, cmap='gist_rainbow')
    contour2 = pp.contour(Lon, Lat, sza_sorted, levels=[0, 90, 180], colors='black', linewidths=1.5, linestyles='--')

    pp.xlabel("Longitude (°E)")
    pp.ylabel("Latitude (°N)")

    pp.clabel(contour2, fmt='%2.0f', colors='black', fontsize=11)
    cb = pp.colorbar(contour)
    cb.set_label('Solar Zenith Angle (degrees)')
    pp.savefig('sza_map.png',dpi=150)

    min_idx = np.unravel_index(np.abs(sza).argmin(), sza.shape)
    subsolar_lat = lat[min_idx[0]]
    subsolar_lon = lon_shifted[min_idx[1]]
    print(f"Approx Subsolar point: {subsolar_lon:.1f}°E, {subsolar_lat:.1f}°N")
    print(f"Grid resolution: {len(lon)} x {len(lat)}")


args = get_args(sys.argv)
verbose = args['verbose']

if args['filelist'][0].endswith('bin'):
    header = read_gitm_header(args['filelist'][:1])
else:
    header = read_ascii_header(args['filelist'][0])

zonal = False 
lsBinWidth = None
if args['var'] is None:
    print('-var is required!')
    exit(1)

smin = None
smax = None

# if len(args['filelist']) == 1:
#     if args['alt'] is None:
#         print("To plot a single file, specify the altitude")
#         args['help'] = '-h'

if args['cut'] == 'alt':
    palt = args['alt']

elif args['cut'] == 'loc':
    if args.get('lon') is None or args.get('lat') is None:
        print("For location-based cut, both -lon and -lat are required.")
        args['help'] = '-h'
    else:
        plon = args['lon'] if args['lon'] >= 0 else 360 + args['lon']
        plat = args['lat']

elif args['cut'] == 'sza':
    if args.get('smin') is None or args.get('smax') is None:
        print("For SZA cut, both -smin and -smax must be specified.")
        args['help'] = '-h'
    else:
        smin = args['smin']
        smax = args['smax']

elif args['cut'] == 'zonal':
    if args.get('alt') is None:
        print("Zonal averaging requires an altitude (-alt)")
        args['help'] = '-h'
    else:
        palt = args['alt']
        if args['zonal'] == 'sza':
            # Zonal average using per-file SZA averaging
            if args.get('smin') is None or args.get('smax') is None:
                print("Zonal SZA averaging requires -smin and -smax")
                args['help'] = '-h'
            else:
                smin = args['smin']
                smax = args['smax']
                zonal = args['zonal']
        elif args['zonal']:
            zonal = args['zonal']
        else:
            try: 
                lsBinWidth = args['binls']
                zonal = True
            except:
                print("zonal averaging requires either -zonal=lt [or subsolar] or -binls=binwidth")
                args['help'] = '-h'            

else:
    print(f"Invalid cut type: {args['cut']}")
    args['help'] = '-h'  

vars = [0,1,2]
vars.extend([int(v) for v in args["var"].split(',')])
varnames = [header['vars'][i] for i in vars]

oco2 = False
if args['oco2']:
    # Normalize varnames (remove LaTeX for comparison)
    clean_varnames = [clean_varname(v) for v in varnames]
    clean_header_vars = [clean_varname(v) for v in header['vars']]

    # Check if [O] and [CO2] already included
    o_present =  '[O]' in clean_varnames
    co2_present = '[CO_2]' in clean_varnames

    if o_present and co2_present:
        oco2 = True
    else:
        # Try to find [O] and [CO2] in header and add them
        try:
            o_index = clean_header_vars.index('[O]')
            co2_index = clean_header_vars.index('[CO_2]')

            # Add if not already in vars
            if o_index not in vars:
                vars.append(o_index)
                varnames.append(header['vars'][o_index])
            if co2_index not in vars:
                vars.append(co2_index)
                varnames.append(header['vars'][co2_index])

            oco2 = True

        except ValueError:
            print("O or CO2 not available to calculate O/CO2")
            args["help"] = '-h'


if (args["help"]):

    print('Usage : ')
    print('plotMarsGRAM.py -var=N1[,N2,N3,...] -cut=type [-lat=lat -lon=lon -alog')
    print('              -smin=smin -smax=smax  -help [file]')
    print('   -help : print this message')
    print('   -var=number[,num2,num3,...] : number is variable to plot')
    print('   -cut=loc,sza,zonal: Plot type ')
    print('   -lat=latitude : latitude in degrees (closest) (cut=loc) ')
    print('   -lon=longitude: longitude in degrees (closest) (cut=loc)')
    print('   -smin=minsza: minimum solar zenith angle (cut=sza)')
    print('   -smax=maxsza: maximum solar zenigh angle (cut=sza)')
    print('   -min=min: minimum value to plot (can be a list, like -var)')
    print('   -max=max: maximum value to plot (can be a list, like -var)')
    print('   -alt=alt: altitude to plot (if a single file or cut=zonal)')
    print('   -alog: plot the log of the variable')
    print('   -binls=binwidth: running average over binwidth degrees ls')
    print('   -oco2: calculate and plot O/CO2 ratio')
    print('   -serial: run the code in serial (good for testing)')
    print('   -zonal=[lt, subsolar, sza]: take zonal average at a fixed lt, subsolar point , or using sza average')
    print('   -verbose: verbose')
    print('   Non-KW arg: file(s)')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


filelist = args["filelist"]
filelist.sort(key=parse_filename)
nfiles = len(filelist)

# Allows specification of the plot range
vmin_map = {}
vmax_map = {}

if args['mini']:
    minv = args['mini'].split(',')
    for i, val in enumerate(minv):
        if i < len(vars): # first 3 don't count
            vmin_map[vars[i+3]] = val

if args['maxi']:
    maxv = args['maxi'].split(',')
    for i, val in enumerate(maxv):
        if i < len(vars): # first 3 don't count
            vmax_map[vars[i+3]] = val


if filelist[0].endswith('bin'):
    #fix GITM variable names, but only if we aren't reading an ascii.
    varnames = [name_dict[var] for var in varnames]

# ---------------------------------------------------------------------- 
# Get the data
# ---------------------------------------------------------------------- 
if args['serial']:
    ### For serial testing only:
    lsBinWidth = None
    data = []
    for file in filelist[:]:
        data.append(readMarsGITM(file, vars, smin=smin, smax=smax, zonal=zonal, lsBinWidth=lsBinWidth, 
        oco2=args['oco2'],verbose=verbose))

    if args['cut'] == 'zonal':
        data = zonal_fixed_ave(data, args['zonal'])
else:
    #process the files in parallel
    vars_working = vars.copy()
    data = process_batch(filelist, vars_working,max_workers=16,smin=smin,
    smax=smax,zonal=zonal,lsBinWidth=lsBinWidth,oco2=oco2,verbose=verbose) 

    if args['zonal'] == 'subsolar' or args['zonal'] == 'sza':
        picklefile = f'1Ddata_{data[0]["time"].strftime("%Y%m%d")}_{data[-1]["time"].strftime("%Y%m%d")}.pkl'
        with open(picklefile, 'wb') as f:
            pickle.dump(data, f)
    
        print(f"Zonally processed data saved to {picklefile}")

if oco2:
    #vars is modified in readMarsGram if we add O/CO2 so we need to also update varnames
    vars.append(max(vars)+1)
    varnames.append('O/CO$_2$')

# ---------------------------------------------------------------------- 
### Plotting
# ---------------------------------------------------------------------- 
# cmap = 'turbo'
cmap = 'plasma'

times = [entry['time'] for entry in data]
alts = data[0]['alt']
plotdim = ""


for var_index, varname in zip(vars[3:], varnames[3:]):
    fig, ax = pp.subplots(figsize=(10,5))

    safe_varname = clean_varname(varname)
    vmin = vmin_map.get(var_index, None)
    vmax = vmax_map.get(var_index, None)

    if 'oco_2' in safe_varname:
        my_norm = LogNorm(vmin=0.1, vmax=10)  # adjust limits if needed
    else:
        my_norm = None  # no special scaling

    if args['cut'] != 'sza' and args['cut'] != 'zonal':
        for entry in data:
            
            lon = entry['lon']
            lat = entry['lat']
            alt_index = np.argmin(np.abs(data[0]['alt'] - palt))

            Z = entry[var_index][..., alt_index].T   # shape (lat, lon)
            Lon, Lat = np.meshgrid(lon, lat, indexing='xy')
            cmap = 'plasma'
            # pcm = ax.contourf(Lon, Lat, Z, cmap=cmap,levels=30)
            pcm = pp.pcolormesh(Lon, Lat, Z,
                                    shading='auto', cmap=cmap,vmin=vmin,vmax=vmax)
    
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(varname)
            
            if (args['cut'] == 'alt'):
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")
                title = time.strftime('%b %d, %Y %H:%M:%S')+'; Alt : '+"%.2f" % alts[alt_index] + ' km'
            ax.set_title(title)

            sTime = entry['time'].strftime('%y%m%d_%H%M%S')
            pp.savefig(f'{safe_varname}_{args["cut"]}_{sTime}+.png')

        # plotSZA(data[0])


    else:
        if args['cut'] == 'sza':
            values = np.array([entry[var_index] for entry in data])
            T,A = np.meshgrid(times,alts)

            mask = A > 250  
            masked_values = np.ma.masked_where(mask, values.T)
            # pcm = pp.contourf(times,alts,values.T,levels=30)
            pcm = pp.pcolormesh(times, alts, masked_values,
                                shading='auto', cmap=cmap,
                                norm=my_norm,vmin=vmin,vmax=vmax)
            if 'oco_2' in safe_varname:
                contour_levels = [0.5, 1.0, 2.0]  # Example: highlight O/CO2 near 1
                pp.contour(times, alts, masked_values, 
                        levels=contour_levels,vmin=vmin,vmax=vmax,
                        colors='white', linewidths=1.0,linestyles='dashed')

            pp.colorbar(pcm, label=f"{varname} (SZA-filtered avg)")
            ax.set_ylim(np.min(A),250)
            ax.set_xlabel("Time")
            ax.set_ylabel("Altitude (km)")
            ax.set_title(f"SZA-filtered Horizontal Average of {varname}")

        elif args['cut'] == 'zonal':
            alt_index = np.argmin(np.abs(data[0]['alt'] - palt)) # time-alt plot
            times = [entry['time'] for entry in data]
            lat = data[0]['lat']
            
            first_var = data[0][var_index]

            if args['zonal'] != 'subsolar' and args['zonal'] != 'sza':
                values = np.array([entry[var_index][:, alt_index] for entry in data])
                pcm = pp.pcolormesh(times, lat, values.T, shading='auto', cmap=cmap,vmin=vmin,vmax=vmax)
                pp.colorbar(pcm, label=f"{varname} (Zonal Avg) \n{int(palt)} km")

                ax.set_ylabel("Latitude (°N)")
                plottype = ""
                plotend = ""

            else:
                # Subsolar or sza tracking (only 1D: alt)
                values = np.array([entry[var_index][alt_index] for entry in data])

                ax.plot(times, values, linestyle='-')
                if vmin is not None or vmax is not None:
                    ax.set_ylim(bottom=vmin, top=vmax)
                ax.set_ylabel(f"{varname}")

                if smin:
                    plottype = 'SZA avg '
                else:
                    plottype = "Subsolar "
                plotend = f"at {int(palt)} km"
                plotdim = f"-1D_{args['zonal']}"


            ax.set_xlabel("Time")

            if args['binls']:
                avetype = f"Ls Bin: {args['binls']}"
            elif args['zonal'] is not None:
                try:
                    lt = int(args['zonal'])
                    avetype = f"({lt} LT)"
                except ValueError: 
                    avetype = f"({args['zonal']}) "
                    
            else:
                avetype = ""
            ax.set_title(f"{plottype}{varname}: Zonal Avg {avetype}{plotend}")

        pp.xticks(rotation=45) 
        pp.tight_layout() 

        #Add Ls on top axis
        xticks = ax.get_xticks()
        xtick_dates = mdates.num2date(xticks)
        ls_vals = [marstiming.getMarsSolarGeometry(t).ls for t in xtick_dates]

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f"{ls:.0f}°" for ls in ls_vals])
        ax2.set_xlabel("Solar Longitude (Ls)")
        pp.subplots_adjust(top=0.85) #make room for the top axis
        
        ax = autoscale_axis(ax,axis='y')
        
        pp.savefig(f"{safe_varname}-{args['cut']}{plotdim}.png",dpi=150)
# ----------------------------------------------------------------------    

endTime = time.time()
print(f"Execution time: {endTime - startTime:.2f} seconds")





