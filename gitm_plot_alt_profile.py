#!/usr/bin/env python
#Plots alt profile for a single location 

from glob import glob
from datetime import datetime
from datetime import timedelta
import argparse
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from gitm_routines import *
import sys

rtod = 180.0/3.141592
boltzmann = 1.380649e-23

def get_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('filelist', nargs='*')
    parser.add_argument('-var', default='15')
    parser.add_argument('-diff', default='0')
    parser.add_argument('-cut', default='loc', choices=['loc', 'sza'])
    parser.add_argument('-lat', type=int, default=-100)
    parser.add_argument('-lon', type=int, default=-100)
    parser.add_argument('-smin', type=int, default=-100)
    parser.add_argument('-smax', type=int, default=-100)
    parser.add_argument('-min', dest='minv', type=float, default=None)
    parser.add_argument('-max', dest='maxv', type=float, default=None)
    parser.add_argument('-minalt', type=float, default=90.0)
    parser.add_argument('-maxalt', type=float, default=280.0)
    parser.add_argument('-alog', dest='IsLog', action='store_true')
    parser.add_argument('-pressure', dest='pressure', action='store_true')
    parser.add_argument('-h', '--help', dest='help', action='store_true')

    parsed_args = parser.parse_args(argv[1:])

    return vars(parsed_args)

args = get_args(sys.argv)
header = read_gitm_header(args["filelist"])

if args['cut'] == 'loc' and args['lon'] > -50:
    plon = args['lon']
    plat = args['lat']
elif args['cut'] == 'sza' and args['smin'] > -50:
    smin = args['smin']
    smax = args['smax']
else:
    args["help"] = '-h'    

if (args["help"]):

    print('Usage : ')
    print('gitm_plot_alt_profile.py -var=N1[,N2,N3,...] -lat=lat -lon=lon -alog')
    print('                     -help [file]')
    print('   -help : print this message')
    print('   -var=number[,num2,num3,...] : number is variable to plot')
    print('   -cut=loc,sza: Plot type ')
    print('   -lat=latitude : latitude in degrees (closest) (cut=loc) ')
    print('   -lon=longitude: longitude in degrees (closest) (cut=loc)')
    print('   -smin=minsza: minimum solar zenith angle (cut=sza)')
    print('   -smax=maxsza: maximum solar zenigh angle (cut=sza)')
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -minalt=min: minimum altitude (km) to plot')
    print('   -maxalt=max: maximum altitude (km) to plot')
    print('   -alog: plot the log of the variable')
    print('   -pressure: calculate and plot pressure from neutral densities and temperature')
    print('   -diff=backgroundFiles: plot the difference between 2 sets of files')
    print('   Non-KW arg: files.')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


filelist = args["filelist"]
nFiles = len(filelist)

if nFiles != 1:
    print('Only 1 file should be specified')
    exit(1)

file = filelist[0]

try:
    iSZA = header["vars"].index('SolarZenithAngle')
    vars = [0,1,2,iSZA]
except: 
    vars = [0,1,2]

diff = False
if args['diff'] != '0':
    diff = True
    backgroundFilelist = sorted(glob(args["diff"]))
    nBackFiles = len(backgroundFilelist)
    if nBackFiles != nFiles:
        print('Difference between sizes of perturbation and background filelists:')
        print('Lengths: {}   {}'.format(nFiles,nBackFiles))
        print('Only 1 file should be specified')
        exit(1)
    bFile = backgroundFilelist[0]

plot_keys = args['var'].split(',')
pressure_density_indices = []
pressure_temp_index = None
if args['pressure']:
    try:
        pressure_temp_index = header['vars'].index('Temperature')
    except ValueError:
        print('Pressure plotting requires an ALL file that includes Temperature.')
        exit(1)

    try:
        rho_index = header['vars'].index('Rho')
    except ValueError:
        print('Pressure plotting requires an ALL file that includes Rho and neutral densities.')
        exit(1)

    pressure_density_indices = [
        i for i in range(rho_index + 1, pressure_temp_index)
        if header['vars'][i].startswith('[') and header['vars'][i].endswith(']')
    ]
    if len(pressure_density_indices) == 0:
        print('Pressure plotting requires neutral density variables between Rho and Temperature in the ALL file.')
        exit(1)

    plot_keys = ['pressure']

requested_vars = []
if not args['pressure']:
    requested_vars.extend([int(v) for v in plot_keys])
else:
    requested_vars.extend(pressure_density_indices + [pressure_temp_index])

vars.extend(requested_vars)
vars = list(dict.fromkeys(vars))

Var = []
for key in plot_keys:
    if key == 'pressure':
        Var.append('Pressure (Pa)')
    else:
        Var.append(header['vars'][int(key)])

nvars = len(plot_keys)
AllData = {a:[] for a in plot_keys}
AllData2D = []
AllAlts = []
AllSZA = []
j = 0

data = read_gitm_one_file(file, vars)
    
[nLons, nLats, nAlts] = data[0].shape
Alts = data[2][0][0]/1000.0
Lons = data[0][:,0,0]*rtod
Lats = data[1][0,:,0]*rtod

if args['maxalt'] <= args['minalt']:
    print('maxalt must be greater than minalt')
    exit(1)

ialt1 = find_nearest_index(Alts,args['minalt'])
ialt2 = find_nearest_index(Alts,args['maxalt'])

time = data["time"]

if diff:
    if bFile == '':
        #It is possible that we don't have an output file at the same time.
        print('Missing background file corresponding to: {}'.format(file))
        exit(1)
    background = read_gitm_one_file(bFile,vars)

if args['cut'] == 'loc':
    ilon = find_nearest_index(Lons,plon)
    ilat = find_nearest_index(Lats,plat)

    for ivar in plot_keys:
        if ivar == 'pressure':
            number_density = np.zeros_like(data[pressure_temp_index][ilon,ilat,ialt1:ialt2+1])
            for idens in pressure_density_indices:
                number_density = number_density + data[idens][ilon,ilat,ialt1:ialt2+1]
            temp = number_density*boltzmann*data[pressure_temp_index][ilon,ilat,ialt1:ialt2+1]
            if diff:
                b_number_density = np.zeros_like(background[pressure_temp_index][ilon,ilat,ialt1:ialt2+1])
                for idens in pressure_density_indices:
                    b_number_density = b_number_density + background[idens][ilon,ilat,ialt1:ialt2+1]
                btemp = b_number_density*boltzmann*background[pressure_temp_index][ilon,ilat,ialt1:ialt2+1]
                temp = (temp-btemp)/btemp*100.0

        elif diff:
            temp = (data[int(ivar)][ilon,ilat,ialt1:ialt2+1]-background[int(ivar)][ilon,ilat,ialt1:ialt2+1])/ \
                background[int(ivar)][ilon,ilat,ialt1:ialt2+1]*100.0
        else:
            temp = data[int(ivar)][ilon,ilat,ialt1:ialt2+1]

        AllData[ivar].append(temp)


if args['cut'] == 'sza':        
    AllSZA.append(data[iSZA][:,:,0])
    mask = (AllSZA[-1] >= smin) & (AllSZA[-1] <= smax ) 
    for ivar in plot_keys:
        if ivar == 'pressure':
            number_density = np.zeros_like(data[pressure_temp_index][:,:,ialt1:ialt2+1])
            for idens in pressure_density_indices:
                number_density = number_density + data[idens][:,:,ialt1:ialt2+1]
            pressure = number_density*boltzmann*data[pressure_temp_index][:,:,ialt1:ialt2+1]
            temp = pressure[mask].mean(axis=0)
            if diff:
                b_number_density = np.zeros_like(background[pressure_temp_index][:,:,ialt1:ialt2+1])
                for idens in pressure_density_indices:
                    b_number_density = b_number_density + background[idens][:,:,ialt1:ialt2+1]
                bpressure = b_number_density*boltzmann*background[pressure_temp_index][:,:,ialt1:ialt2+1]
                mean2 = bpressure[mask].mean(axis=0)
                temp = (temp-mean2)/mean2*100.

        elif diff:
            #Calculate the mean of both sets of data and then calculate the percent difference.
            mean1 = data[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
            mean2 = background[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
            temp = (mean1-mean2)/mean2*100.

        else:
            temp = data[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)

        AllData[ivar].append(temp)

for ivar in plot_keys:
    AllData[ivar] = np.array(AllData[ivar])


if args['cut']  == 'sza':
    AllSZA = np.array(AllSZA)

fig = pp.figure()

Alts = Alts[ialt1:ialt2+1]

cmap = 'plasma'
i=0
ax = pp.subplot(121)
if len(Var) == 1:
    marker = '+'
else:
    marker = '.'

for ivar in plot_keys:
    AllData1D = AllData[ivar][0]
    if (ivar == '3' and (not diff)) or args['IsLog']:
        mask = (AllData1D != 0.0) 
        AllData1D = np.log10(AllData1D[mask])
        Alts = Alts[mask]
        Var[i] = Var[i].replace('!U','^')
        Var[i] = Var[i].replace('!D','_')
        Var[i] = Var[i].replace('!N','')
        Var[i] = '$'+Var[i]+'$'

    
    if len(Var) > 1: 
        label = Var[i]
    else:
        label = None
    plot = ax.plot(AllData1D,Alts,marker,label=label)  
    i+=1

if len(Var) == 1:
    if plot_keys[0] == 'pressure':
        svar = 98
    else:
        svar = int(plot_keys[0])
    if diff:
        xlabel = '{}\n% Diff'.format(Var[0])
    else:
        xlabel = Var[0]
else:
    xlabel = 'Density'
    svar = 99
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1.5, box.height])
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5),frameon=False)
   
# if i < len(Var)-1:
#     ax.get_xaxis().set_ticklabels([])

pp.ylabel('Alt (km)')
if args['IsLog']:
    xlabel = 'Log '+xlabel
pp.xlabel(xlabel)

if args['minv'] == None:
    minv = min(AllData1D)
else:
    minv = args['minv']
if args['maxv'] == None:
    maxv = max(AllData1D)
else:
    maxv = args['maxv']

pp.xlim([minv,maxv])
pp.ylim([args['minalt'],args['maxalt']])

   

outfile = 'altprofile_var{:02d}_{}.png'.format(svar,time.strftime('%y%m%d_%H%M%S'))
print("Writing to file: {}".format(outfile))
pp.savefig(outfile)
