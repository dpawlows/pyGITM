#!/usr/bin/env python

import numpy as np 
from matplotlib import pyplot as pp 
import matplotlib.dates as mdates
import sys 
import re 
from gitm_routines import * 
from gitmconcurrent import *
import pandas as pd 
import time 

startTime = time.time()
def get_args(argv):

    filelist = []
    var = -1
    help = False
    alog = False
    min = None 
    max = None 
    alt = None
    lon = None
    lat = None
    cut = None
    smin = None
    smax = None
    binwidth = None 
    oco2 = None

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
                min = float(m.group(1))
                IsFound = 1

            m = re.match(r'-max=(.*)',arg)
            if m:
                max = float(m.group(1))
                IsFound = 1
            
            m = re.match(r'-h',arg)
            if m:
                help = True
                IsFound = 1
            
            m = re.match(r'-oco2',arg)
            if m:
                oco2 = True
                IsFound = 1

            m = re.match(r'-alog',arg)
            if m:
                alog = True
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

            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)
            
    args = {'filelist':filelist,
        'var':var,
        'help':help,
        'alog':alog,
        'min':min,
        'max':max,
        'cut':cut,
        'lat':lat,
        'lon':lon,
        'alt':alt,
        'smin':smin,
        'smax':smax,
        'binls':binwidth,
        'oco2':oco2
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
header = read_ascii_header(args['filelist'][0])
zonal = False 

if not args['var']:
    print('-var is required!')
    args["help"] = '-h'  

smin = None
smax = None

if len(args['filelist']) == 1:
    if args['alt'] is None:
        print("To plot a single file, specify the altitude")
        args['help'] = '-h'
    else:
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
        zonal = True

else:
    print(f"Invalid cut type: {args['cut']}")
    args['help'] = '-h'  

vars = [0,1,2]
vars.extend([int(v) for v in args["var"].split(',')])
varnames = [header['vars'][i] for i in vars]
oco2 = False
if args['oco2']:
    # Normalize varnames (remove LaTeX for comparison)
    clean_varnames = [v.replace('$', '').replace('{', '').replace('}', '') for v in varnames]
    clean_header_vars = [v.replace('$', '').replace('{', '').replace('}', '') for v in header['vars']]

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
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -alt=alt: altitude to plot (if a single file or cut=zonal)')
    print('   -alog: plot the log of the variable')
    print('   -binls=binwidth: running average over binwidth degrees ls')
    print('   -oco2: calculate and plot O/CO2 ratio')
    print('   Non-KW arg: file(s)')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


filelist = args["filelist"]
nfiles = len(filelist)

#process the files in parallel
vars_working = vars.copy()
data = process_batch(filelist, vars_working,max_workers=16,smin=smin,
   smax=smax,zonal=zonal,lsBinWidth=args['binls'],oco2=oco2) 

#####################################
#For serial testing only:
# lsBinWidth = None
# data = []
# for file in filelist[:1]:
#    data.append(readMarsGITM(file, vars, smin=smin, smax=smax, zonal=zonal, lsBinWidth=lsBinWidth, oco2=args['oco2']))
#####################################
if oco2:
    #vars is modified in readMarsGram if we add O/CO2 so we need to also update varnames
    vars.append(max(vars)+1)
    varnames.append('O/CO$_2$')

endTime = time.time()
print(f"Execution time: {endTime - startTime:.2f} seconds")

# ---------------------------------------------------------------------- 
### Plotting

cmap = 'turbo'

times = [entry['time'] for entry in data]
alts = data[0]['alt']

for var_index, varname in zip(vars[3:], varnames[3:]):
    fig, ax = pp.subplots(figsize=(10,6))

    safe_varname = (varname.replace('$', '')
                             .replace('{', '')
                             .replace('}', '')
                             .replace('/', '')
                             .replace('[', '')
                             .replace(']', '')
                             .lower())

    if len(filelist) == 1:
        entry = data[0]
        lon = entry['lon']
        lat = entry['lat']
        alt_index = np.argmin(np.abs(data[0]['alt'] - palt))

        Z = entry[var_index][..., alt_index].T   # shape (lat, lon)
        Lon, Lat = np.meshgrid(lon, lat, indexing='xy')

        pcm = ax.contourf(Lon, Lat, Z, cmap=cmap,levels=30)
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(varname)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{varname} at {int(alts[alt_index])} km")
        pp.savefig(f'{safe_varname}_altcut.png')

    else:
        if args['cut'] == 'sza':
            values = np.array([entry[var_index] for entry in data])
            T,A = np.meshgrid(times,alts)
            # pcm = pp.contourf(times,alts,values.T,levels=30)
            pcm = pp.pcolormesh(times, alts, values.T,
                                shading='auto', cmap=cmap)

            pp.colorbar(pcm, label=f"{varname} (SZA-filtered avg)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Altitude (km)")
            ax.set_title(f"SZA-filtered Horizontal Average of {varname}")
        elif args['cut'] == 'zonal' and args['alt'] is not None:

            alt_index = np.argmin(np.abs(data[0]['alt'] - palt))

            times = [entry['time'] for entry in data]
            lat = data[0]['lat']
            values = np.array([entry[var_index][:, alt_index] for entry in data])

            pcm = pp.pcolormesh(times, lat, values.T, shading='auto', cmap=cmap)

            pp.colorbar(pcm, label=f"{varname} (Zonal Avg) \n{int(palt)} km")
            ax.set_xlabel("Time")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"Time–Latitude of {varname} (Zonal Avg)")

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
        
        pp.savefig(f"{safe_varname}-{args['cut']}.png",dpi=150)
# ----------------------------------------------------------------------    


# plotSZA(data[0])




