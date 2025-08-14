#!/usr/bin/env python
#Plots alt profile for a single location 

from glob import glob
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from gitm_routines import *
import argparse
import sys
import os

rtod = 180.0/3.141592


def compute_ratio(numer, denom):
    """Safely compute a ratio avoiding divide by zero."""
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)


def percent_diff(data, baseline):
    """Return percent difference between two arrays."""
    return (data - baseline) / baseline * 100.0


def get_args(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot GITM altitude profile at a location or SZA range"
    )
    parser.add_argument("filelist", nargs="+", help="Input GITM file(s)")
    parser.add_argument("-var", type=str, help="Variable(s) to plot, comma separated")
    parser.add_argument("-diff", default="0", help="Background files glob for differencing")
    parser.add_argument("-cut", default="loc", choices=["loc", "sza"], help="Plot type")
    parser.add_argument("-lat", type=float, default=-100.0, help="Latitude in degrees")
    parser.add_argument("-lon", type=float, default=-100.0, help="Longitude in degrees")
    parser.add_argument("-smin", type=float, default=-100.0, help="Minimum solar zenith angle")
    parser.add_argument("-smax", type=float, default=-100.0, help="Maximum solar zenith angle")
    parser.add_argument("-min", dest="minv", type=float, help="Minimum value to plot")
    parser.add_argument("-max", dest="maxv", type=float, help="Maximum value to plot")
    parser.add_argument("-alog", dest="IsLog", action="store_true", help="Plot log of variable")
    parser.add_argument("-oco2", action="store_true", help="Calculate and plot O/CO2 ratio")
    parser.add_argument("-press", dest="pressure", action="store_true", help="Use pressure as vertical coordinate")
    parser.add_argument("-grid", action="store_true", help="Enable grid on plot")
    parser.add_argument("--list-vars", action="store_true", help="List available variables and exit")
    parser.add_argument("-mix", dest="mixing", action="store_true", help="Plot mixing ratio for neutral species")

    args = parser.parse_args(argv[1:])

    if args.cut == "loc":
        if args.lon <= -50:
            parser.error("cut=loc requires -lon and -lat")
    elif args.cut == "sza":
        if args.smin <= -50 or args.smax <= -50:
            parser.error("cut=sza requires -smin and -smax")

    if args.mixing and (args.var is not None or args.oco2):
        parser.error("-mix cannot be combined with -var or -oco2")

    return vars(args)

args = get_args(sys.argv)
header = read_gitm_header(args["filelist"])

if args["list_vars"]:
    for i, var in enumerate(header["vars"]):
        print(i, var)
    sys.exit()

if args['cut'] == 'loc':
    plon = args['lon']
    plat = args['lat']
else:  # cut == 'sza'
    smin = args['smin']
    smax = args['smax']


filelist = args["filelist"]
nFiles = len(filelist)
if nFiles > 1:
    print('Only 1 file should be specified')
    exit(1)

file = filelist[0]

try:
    iSZA = header["vars"].index('SolarZenithAngle')
    vars = [0,1,2,iSZA]
except:
    vars = [0,1,2]

# Only include neutral species (ions have a '+' and electrons are '[e-]')
species_inds = [
    i for i, name in enumerate(header['vars'])
    if name.startswith('[') and '+' not in name and 'e-' not in name.lower()
]
def _norm_species(name):
    return (
        name.replace('[', '').replace(']', '')
        .replace('!D', '').replace('!N', '').replace('!U', '')
        .lower()
    )
norm_names = [_norm_species(n) for n in header['vars']]
n2_index = norm_names.index('n2') if 'n2' in norm_names else None
ar_index = norm_names.index('ar') if 'ar' in norm_names else None
temp_index = None
if args['pressure']:
    # Identify temperature variable
    for tname in ['Temperature', 'Tn']:
        if tname in header['vars']:
            temp_index = header['vars'].index(tname)
            break
    if temp_index is None:
        print('Temperature not available to calculate pressure')
        exit(1)
    if temp_index not in vars:
        vars.append(temp_index)
    for idx in species_inds:
        if idx not in vars:
            vars.append(idx)

if args.get('mixing'):
    for idx in species_inds:
        if idx not in vars:
            vars.append(idx)

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
if args.get('mixing'):
    var_list = [str(i) for i in species_inds]
    Var = [header['vars'][i] for i in species_inds]
elif args['oco2']:
    if args['var'] is not None:
        print('Cannot specify both -var and -oco2')
        exit(1)
    if not os.path.basename(file).startswith('3DALL'):
        print('O/CO2 ratio can only be calculated from 3DALL files')
        exit(1)
    try:
        o_index = header['vars'].index('[O]')
        co2_index = header['vars'].index('[CO!D2!N]')
    except ValueError:
        print('O or CO2 not available to calculate O/CO2')
        exit(1)
    for idx in [o_index, co2_index]:
        if idx not in vars:
            vars.append(idx)
    var_list = ['O/CO2']
    Var = ['O/CO2']
else:
    if args['var'] is None:
        print('Either -var, -oco2, or -mix must be specified')
        exit(1)
    var_list = args['var'].split(',')
    Var = []
    for v in var_list:
        if v == 'O/CO2':
            Var.append('O/CO2')
        else:
            Var.append(header['vars'][int(v)])
vars.extend([int(v) for v in var_list if v.isdigit()])
nvars = len(var_list)
AllData = {a:[] for a in var_list}
AllData2D = []
AllAlts = []
AllSZA = []

data = read_gitm_one_file(file, vars)

[nLons, nLats, nAlts] = data[0].shape
AltKm = data[2][0][0]/1000.0
Lons = data[0][:,0,0]*rtod
Lats = data[1][0,:,0]*rtod

if args['pressure']:
    kb = 1.380649e-23
    temp = data[temp_index]
    dens = np.zeros_like(temp)
    for idx in species_inds:
        dens += data[idx]
    pressure = dens * kb * temp

ialt1 = find_nearest_index(AltKm,90)
ialt2 = find_nearest_index(AltKm,300)

time = data["time"]

if diff:
    if bFile == '':
        #It is possible that we don't have an output file at the same time.
        print('Missing background file corresponding to: {}'.format(file))
        exit(1)
    background = read_gitm_one_file(bFile,vars)

Press = None
mask = None
if args['cut'] == 'loc':
    ilon = find_nearest_index(Lons,plon)
    ilat = find_nearest_index(Lats,plat)

    if args['pressure']:
        Press = pressure[ilon, ilat, ialt1:ialt2+1]
    if args.get('mixing'):
        total = np.zeros_like(data[species_inds[0]][ilon, ilat, ialt1:ialt2+1])
        for idx in species_inds:
            total += data[idx][ilon, ilat, ialt1:ialt2+1]
        if diff:
            base_total = np.zeros_like(background[species_inds[0]][ilon, ilat, ialt1:ialt2+1])
            for idx in species_inds:
                base_total += background[idx][ilon, ilat, ialt1:ialt2+1]

    for ivar in var_list:
        if ivar == 'O/CO2':
            o = data[o_index][ilon, ilat, ialt1:ialt2+1]
            co2 = data[co2_index][ilon, ilat, ialt1:ialt2+1]
            ratio = compute_ratio(o, co2)
            if diff:
                o_b = background[o_index][ilon, ilat, ialt1:ialt2+1]
                co2_b = background[co2_index][ilon, ilat, ialt1:ialt2+1]
                base_ratio = compute_ratio(o_b, co2_b)
                temp = percent_diff(ratio, base_ratio)
            else:
                temp = ratio
        elif args.get('mixing'):
            prof = data[int(ivar)][ilon, ilat, ialt1:ialt2+1]
            ratio = compute_ratio(prof, total)
            if diff:
                base = background[int(ivar)][ilon, ilat, ialt1:ialt2+1]
                base_ratio = compute_ratio(base, base_total)
                temp = percent_diff(ratio, base_ratio)
            else:
                temp = ratio
        else:
            prof = data[int(ivar)][ilon, ilat, ialt1:ialt2+1]
            if diff:
                base = background[int(ivar)][ilon, ilat, ialt1:ialt2+1]
                temp = percent_diff(prof, base)
            else:
                temp = prof

        AllData[ivar].append(temp)


if args['cut'] == 'sza':
    AllSZA.append(data[iSZA][:,:,0])
    mask = (AllSZA[-1] >= smin) & (AllSZA[-1] <= smax )
    if args['pressure']:
        Press = pressure[:,:,ialt1:ialt2+1][mask].mean(axis=0)
    if args.get('mixing'):
        total = np.zeros(ialt2 - ialt1 + 1)
        for idx in species_inds:
            total += data[idx][:,:,ialt1:ialt2+1][mask].mean(axis=0)
        if diff:
            base_total = np.zeros(ialt2 - ialt1 + 1)
            for idx in species_inds:
                base_total += background[idx][:,:,ialt1:ialt2+1][mask].mean(axis=0)

    for ivar in var_list:
        if ivar == 'O/CO2':
            o = data[o_index][:,:,ialt1:ialt2+1][mask]
            co2 = data[co2_index][:,:,ialt1:ialt2+1][mask]
            ratio = compute_ratio(o, co2).mean(axis=0)
            if diff:
                o_b = background[o_index][:,:,ialt1:ialt2+1][mask]
                co2_b = background[co2_index][:,:,ialt1:ialt2+1][mask]
                base_ratio = compute_ratio(o_b, co2_b).mean(axis=0)
                temp = percent_diff(ratio, base_ratio)
            else:
                temp = ratio
        elif args.get('mixing'):
            prof = data[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
            ratio = compute_ratio(prof, total)
            if diff:
                base = background[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
                base_ratio = compute_ratio(base, base_total)
                temp = percent_diff(ratio, base_ratio)
            else:
                temp = ratio
        else:
            prof = data[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
            if diff:
                base = background[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
                temp = percent_diff(prof, base)
            else:
                temp = prof

        AllData[ivar].append(temp)

homopause_alt = None
if args.get('mixing') and n2_index is not None and ar_index is not None:
    if args['cut'] == 'loc':
        n2_prof = data[n2_index][ilon, ilat, ialt1:ialt2+1]
        ar_prof = data[ar_index][ilon, ilat, ialt1:ialt2+1]
    else:
        n2_prof = data[n2_index][:,:,ialt1:ialt2+1][mask].mean(axis=0)
        ar_prof = data[ar_index][:,:,ialt1:ialt2+1][mask].mean(axis=0)
    ratio = compute_ratio(n2_prof, ar_prof)
    surf_ratio = ratio[0]
    idxs = np.where(ratio < 1.5 * surf_ratio)[0]
    if len(idxs) > 0:
        homopause_alt = AltKm[ialt1:ialt2+1][idxs[-1]]

for ivar in var_list:
    AllData[ivar] = np.array(AllData[ivar])


if args['cut']  == 'sza':
    AllSZA = np.array(AllSZA)

fig = pp.figure()
if args['pressure']:
    Alts = Press
else:
    Alts = AltKm[ialt1:ialt2+1]

cmap = 'plasma'
i=0
ax = pp.subplot(111)
if len(Var) == 1:
    marker = '+'
else:
    marker = '.'

for ivar in var_list:
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
    svar = var_list[0]
    if svar.isdigit():
        svar = f'{int(svar):02d}'
    else:
        svar = svar.replace('/', '_')
    if diff:
        xlabel = '{}\n% Diff'.format(Var[0])
    else:
        xlabel = Var[0]
else:
    if args.get('mixing'):
        xlabel = 'Mixing Ratio'
    else:
        xlabel = 'Density'
    svar = 'multi'
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),frameon=False)
   
# if i < len(Var)-1:
#     ax.get_xaxis().set_ticklabels([])

if args['pressure']:
    pp.ylabel('Pressure (Pa)')
else:
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

if 'O/CO2' in var_list:
    pp.axvline(x=1, linestyle='--')

if args['grid']:
    pp.grid(True)  # Turn the grid on

pp.xlim([minv,maxv])
if args['pressure']:
    ax.set_yscale('log')
    ax.set_ylim([Alts.max(), Alts.min()])
else:
    pp.ylim([90,250])

if homopause_alt is not None:
    ax.text(0.95, 0.95, f'Homopause: {homopause_alt:.1f} km',
            transform=ax.transAxes, ha='right', va='top')

   

prefix = 'pressprofile' if args['pressure'] else 'altprofile'
outfile = '{}_var{}_{}.png'.format(prefix, svar, time.strftime('%y%m%d_%H%M%S'))
print("Writing to file: {}".format(outfile))
pp.savefig(outfile)
