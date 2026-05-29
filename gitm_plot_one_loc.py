#!/usr/bin/env python
#Plots a altitude and time for a single location
from glob import glob
from datetime import datetime
from datetime import timedelta
import argparse
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib import  ticker
from gitm_routines import *
import sys
import marstiming as mt
import netCDF4 as nc
from gitmconcurrent import process_batch


rtod = 180.0/3.141592
marsDay = 1.02749125 * 86400
minalt = 100
maxalt = 250

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_args(argv):

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-var', default='15')
    parser.add_argument('-diff', default='0')
    parser.add_argument('-cut', default='loc')
    parser.add_argument('-lat', type=int, default=-100)
    parser.add_argument('-lon', type=int, default=-100)
    parser.add_argument('-alt', type=int, default=-100)
    parser.add_argument('-lt', type=int, default=-100)
    parser.add_argument('-average', type=int, default=-100)
    parser.add_argument('-smin', type=int, default=-100)
    parser.add_argument('-smax', type=int, default=-100)
    parser.add_argument('-mini', type=float)
    parser.add_argument('-maxi', type=float)
    parser.add_argument('-alog', action='store_true')
    parser.add_argument('-oplotmax', action='store_true')
    parser.add_argument('-lsavg', type=float, default=-1)
    parser.add_argument('-savedata', action='store_true')
    parser.add_argument('-pressure', type=float, default=0.0)
    parser.add_argument('-ymin', type=float, default=None)
    parser.add_argument('-ymax', type=float, default=None)
    parser.add_argument('-h', '-help', action='store_true', dest='help')
    parser.add_argument('filelist', nargs='*')

    parsed = parser.parse_args(argv[1:])

    args = {'filelist':parsed.filelist,
            'var':parsed.var,
            'help':parsed.help,
            'lat':parsed.lat,
            'lon':parsed.lon,
            'alt':parsed.alt,
            'IsLog':int(parsed.alog),
            'cut':parsed.cut,
            'smin':parsed.smin,
            'smax':parsed.smax,
            'diff':parsed.diff,
            'lt':parsed.lt,
            'average':parsed.average,
            'mini':parsed.mini,
            'maxi':parsed.maxi,
            'oplotmax':parsed.oplotmax,
            'lsavg':parsed.lsavg,
            'savedata':parsed.savedata,
            'pressure':parsed.pressure,
            'ymin':parsed.ymin,
            'ymax':parsed.ymax}

    return args

args = get_args(sys.argv)
if args['filelist']:
    header = read_gitm_header(args["filelist"])
else:
    header = {'vars': []}

boltzmann = 1.380649e-23
pressure_mode = args['pressure'] > 0
pressure_density_indices = []
pressure_temp_index = None

if pressure_mode:
    try:
        pressure_temp_index = header['vars'].index('Temperature')
    except ValueError:
        print('Pressure mode requires an ALL file that includes Temperature.')
        exit(1)
    pressure_density_indices = [
        i for i, v in enumerate(header['vars'])
        if v.startswith('[') and v.endswith(']')
        and '+' not in v and '-' not in v and '00' not in v
    ]
    if not pressure_density_indices:
        print('Pressure mode requires neutral species number density variables in the ALL file.')
        exit(1)

averaging = False
if args['average'] > 0:
    averaging = True
    tAverage = args['average']

if args['cut'] == 'loc' and args['lon'] > -50:
    plon = args['lon']
    plat = args['lat']
elif args['cut'] == 'sza' and args['smin'] > -50:
    smin = args['smin']
    smax = args['smax']
elif args['cut'] == 'lt' and args['lat'] > -91:
    plat = args['lat']
else:
    args["help"] = '-h'

if args['cut'] == 'lt' and args['alt'] > 0:
    lineplot = True
    palt = args['alt']
else:
    lineplot = False

if averaging and args['lt'] < 0:
    print('Time averaging can only be performed on lt plot type')
    args["help"] = '-h'

if (args["help"]):

    print('Usage : ')
    print('gitm_plot_one_loc.py -var=N1[,N2,N3,...] -lat=lat -lon=lon -alog')
    print('                     -help [file]')
    print('   -help : print this message')
    print('   -var=number[,num2,num3,...] : number is variable to plot')
    print('   -cut=loc,sza,lt: Plot type ')
    print('   -lat=latitude : latitude in degrees (closest) (cut=loc) ')
    print('   -lon=longitude: longitude in degrees (closest) (cut=loc)')
    print('   -alt=altitude: altitude in km (closest) (cut=lt) will create a line plot')
    print('   -smin=minsza: minimum solar zenith angle (cut=sza)')
    print('   -smax=maxsza: maximum solar zenigh angle (cut=sza)')
    print('   -lt=localtime: nearest localtime to plot')
    print('   -average=time: average a local time plot across time seconds')
    print('   -alog: plot the log of the variable')
    print('   -oplotmax: overplot the altitude of the maximum value')
    print('   -lsavg=degrees: average data into Ls bins of the given width')
    print('   -savedata: save plotted data to a NetCDF (.nc) file')
    print('   -diff=backgroundFiles: plot the difference between 2 sets of files')
    print('   -pressure=Pa: plot altitude of the given pressure level vs Ls (cut=loc only, requires Temperature and neutral species in ALL file)')
    print('   Non-KW args: files.')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()

filelist = args["filelist"]
nFiles = len(filelist)
if nFiles < 2:
    print('Please enter multiple files')
    exit(1)

# Sort filenames based on year
# Regular expression pattern to match 3D???_t', and a date and time stamp
pattern = r'(.*)?3D..._t(\d{6})_(\d{6})'

fl = sorted(filelist, key=lambda x: extract_timestamp(x,pattern))
filelist = fl

diff = False
if args['diff'] != '0':
    diff = True
    backgroundFilelist = sorted(glob(args["diff"]))
    fl = sorted(backgroundFilelist, key=lambda x: extract_timestamp(x,pattern))
    backgroundFilelist = fl
    nBackFiles = len(backgroundFilelist)
    if nBackFiles != nFiles:
        print('Difference between sizes of perturbation and background filelists:')
        print('Lengths: {}   {}'.format(nFiles,nBackFiles))
        exit(1)

if pressure_mode:
    vars = [0, 1, 2] + pressure_density_indices + [pressure_temp_index]
    PressureAlts = []
else:
    vars = [0, 1, 2] + [int(v) for v in args["var"].split(',')]
    Var = [header['vars'][int(i)] for i in args['var'].split(',')]
    nvars = len(args['var'].split(','))
    #We want to store data for multiple variables, so we use a dict where var indices are the keys
    AllData = {a:[] for a in args['var'].split(',')}
sum = []
AllTimes = []
j = 0
indexDayStart = []
newday = True

# Read all files in parallel

if pressure_mode:
    results = process_batch(filelist, vars,
                            pressure_density_indices=pressure_density_indices,
                            pressure_temp_index=pressure_temp_index)
else:
    results = process_batch(filelist, vars)

AllTimes = [r['time'] for r in results]
AllLs = np.array([mt.getMarsSolarGeometry([t.year,t.month,t.day,t.hour,t.minute,t.second]).ls
                  for t in AllTimes])

if diff:
    bg_results = process_batch(backgroundFilelist, vars)

# Geometry from first result (ghost cells stripped, alt already in km)
first = results[0]
lon = first['lon']
lat = first['lat']
alt = first['alt']

ialt1 = find_nearest_index(alt, minalt)
ialt2 = find_nearest_index(alt, maxalt)

for j, result in enumerate(results):
    bg = bg_results[j] if diff else None

    if args['cut'] == 'loc':
        ilon = find_nearest_index(lon, plon)
        ilat = find_nearest_index(lat, plat)
        if pressure_mode:
            pressure_profile = result['pressure'][ilon, ilat, ialt1:ialt2+1]
            PressureAlts.append(np.interp(args['pressure'], pressure_profile[::-1], alt[ialt1:ialt2+1][::-1]))
        else:
            for ivar in args['var'].split(','):
                v = int(ivar)
                if diff:
                    temp = (result[v][ilon,ilat,ialt1:ialt2+1] - bg[v][ilon,ilat,ialt1:ialt2+1]) / \
                        bg[v][ilon,ilat,ialt1:ialt2+1] * 100.0
                else:
                    temp = result[v][ilon,ilat,ialt1:ialt2+1]

                AllData[ivar].append(temp)


    if args['cut'] == 'sza':
        sza = result['sza']  # 2D (nlon x nlat), computed by readMarsGITM
        mask = (sza >= smin) & (sza <= smax)
        if pressure_mode:
            pressure_profile = result['pressure'][:,:,ialt1:ialt2+1][mask].mean(axis=0)
            PressureAlts.append(np.interp(args['pressure'], pressure_profile[::-1], alt[ialt1:ialt2+1][::-1]))

        for ivar in args['var'].split(',') if not pressure_mode else []:
            v = int(ivar)
            if diff:

                #Calculate the mean of both sets of data and then calculate the percent difference.
                if ivar == '2':
                    mean1 = (result[6][:,:,ialt1:ialt2+1][mask].mean(axis=0)/\
                             result[4][:,:,ialt1:ialt2+1][mask].mean(axis=0))
                    mean2 = (bg[6][:,:,ialt1:ialt2+1][mask].mean(axis=0)/\
                             bg[4][:,:,ialt1:ialt2+1][mask].mean(axis=0))
                else:
                    mean1 = result[v][:,:,ialt1:ialt2+1][mask].mean(axis=0)
                    mean2 = bg[v][:,:,ialt1:ialt2+1][mask].mean(axis=0)

                temp = (mean1-mean2)/mean2*100.

            else:
                temp = result[v][:,:,ialt1:ialt2+1][mask].mean(axis=0)

            AllData[ivar].append(temp)

    if args['cut'] == 'lt':
        marstime = mt.getMarsSolarGeometry([result['time'].year,result['time'].month,result['time'].day,\
            result['time'].hour,result['time'].minute,result['time'].second])

        #subsolarlon is in degrees west so convert to east first
        subsolarlon = marstime.subSolarLon
        ltdiff = args['lt'] - 12  #subsolar is at 12:00 LT
        plon = (subsolarlon + ltdiff*360/24) % 360
        if lineplot:
            ialt = find_nearest_index(alt, palt)

        ilon = find_nearest_index(lon, plon)
        ilat = find_nearest_index(lat, plat)

        for ivar in args['var'].split(','):
            v = int(ivar)
            if lineplot:

                if diff:
                    AllData[ivar].append((result[v][ilon,ilat,ialt] -
                                         bg[v][ilon,ilat,ialt])/\
                                         bg[v][ilon,ilat,ialt]*100.0)
                else:
                    AllData[ivar].append(result[v][ilon,ilat,ialt])
            else:

                if diff:
                    temp = (result[v][ilon,ilat,ialt1:ialt2+1] - bg[v][ilon,ilat,ialt1:ialt2+1]) / \
                        bg[v][ilon,ilat,ialt1:ialt2+1]*100.0
                else:
                    temp = result[v][ilon,ilat,ialt1:ialt2+1]

                if averaging:
                    if newday:
                        sum = temp
                        newday = False
                        indexDayStart.append(j)
                        aveTStart = AllTimes[j]
                    else:
                        sum = sum + temp
                    if (result['time'] - aveTStart).total_seconds() > marsDay:
                        newday = True
                        AllData[ivar].append(sum/(j-indexDayStart[-1]+1))
                        sum = []

                else:
                    AllData[ivar].append(temp)


if pressure_mode:
    PressureAlts = np.array(PressureAlts)
    # Sort by Ls so the filter and plot are in order
    sort_idx = np.argsort(AllLs)
    AllLs = AllLs[sort_idx]
    PressureAlts = PressureAlts[sort_idx]
else:
    for ivar in args['var'].split(','):
        AllData[ivar] = np.array(AllData[ivar])

lsAveraging = args['lsavg'] > 0
if lsAveraging:
    lsavg_deg = args['lsavg']
    if pressure_mode:
        from scipy.signal import savgol_filter
        ls_spacing = np.median(np.diff(AllLs))
        window_samples = int(round(lsavg_deg / ls_spacing))
        if window_samples % 2 == 0:
            window_samples += 1
        window_samples = max(window_samples, 3)
        PressureAlts = savgol_filter(PressureAlts, window_samples, 3)
    else:
        bin_edges = np.arange(0, 360 + lsavg_deg, lsavg_deg)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ls_avg_data = {ivar: [] for ivar in args['var'].split(',')}
        valid_centers = []
        for lo, hi, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
            mask = (AllLs >= lo) & (AllLs < hi)
            if mask.sum() > 0:
                valid_centers.append(center)
                for ivar in args['var'].split(','):
                    ls_avg_data[ivar].append(AllData[ivar][mask].mean(axis=0))
        for ivar in args['var'].split(','):
            AllData[ivar] = np.array(ls_avg_data[ivar])
        LsTimes = np.array(valid_centers)

Alts = alt[ialt1:ialt2+1]

if pressure_mode:
    xdata = AllLs  # always Ls-sorted; savgol applied in-place above if lsAveraging
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(xdata, PressureAlts)
    ax.set_xlabel('Ls (degrees)')
    ax.set_ylabel('Altitude (km)')
    if args['cut'] == 'loc':
        subtitle = 'Lat={}, Lon={}'.format(plat, plon)
    elif args['cut'] == 'sza':
        subtitle = 'SZA={}-{}'.format(smin, smax)
    else:
        subtitle = 'LT={}, Lat={}'.format(args['lt'], plat)
    ax.set_title('Altitude of {:.2e} Pa pressure level  ({})'.format(args['pressure'], subtitle))
    if args['ymin'] is not None or args['ymax'] is not None:
        ax.set_ylim(args['ymin'], args['ymax'])
    pp.tight_layout()
    stime = AllTimes[0].strftime('%Y%m%d_%H%M%S')
    outfile = 'gitm_{}_pressure{}_{}.png'.format(args['cut'], args['pressure'], stime)
    print('Writing file: {}'.format(outfile))
    pp.savefig(outfile)
    exit()

cmap = 'plasma'
i=0

if lsAveraging:
    Times = LsTimes
elif averaging:
    averageDayStart = ((np.asarray(indexDayStart[0:-1])+np.asarray(indexDayStart[1:]))/2).astype(int)
    Times = [AllTimes[i] for i in averageDayStart]
else:
    Times = AllTimes

if lineplot and args['cut'] != 'lt':
    extraplot = 1
else:
    extraplot = 0

if args['cut'] == 'lt':
    figsize = (8,4.5)
else:
    figsize = (8,6)

fig, ax = plt.subplots(np.max([1,len(Var)])+extraplot, 1, sharex=True,figsize=figsize)
nlevels = 30

mini = args['mini']
maxi = args['maxi']

userrange = False
if mini != None or maxi != None:
    userrange = True

hcmaxi = np.asarray([60,30,55])
hcmini = hcmaxi*-1
for ivar in args['var'].split(','):

    AllData2D = AllData[ivar]
    if ivar == '3' and (not diff) and (not lineplot):
        AllData2D = np.log10(AllData2D)
        Var[i] = "Log "+ name_dict[Var[i]]
    else:
        Var[i] = name_dict[Var[i]]

    if ivar == '2':
        Var[i] = "O/CO$_2$"

    if nvars > 1:
        thisax = ax[i]
    else:
        thisax = ax
    if not userrange:
        mini = np.min(AllData2D)
        maxi = np.max(AllData2D)

    if diff:
        absmax = np.max([np.abs(mini),np.abs(maxi)])
        mini = -absmax
        maxi = absmax
        cmap = pp.get_cmap('twilight_shifted')#.reversed()

    if diff:
        Var[i] = '{}\n% Diff'.format(Var[i])
    else:
        Var[i] = Var[i]

    if diff:
        mini = hcmini[i]
        maxi = hcmaxi[i]
    if lineplot:
         thisax.plot(Times,AllData2D)
         thisax.set_ylim([mini,maxi])
         thisax.set_ylabel(Var[i])
         pp.tight_layout()
    else:
        levels = np.linspace(mini,maxi,30)
        cont = thisax.contourf(Times,Alts,np.transpose(AllData2D),levels=levels,cmap=cmap)

        if int(ivar) == 3:

            i250 = np.argmin(np.abs(Alts-250))
            imax = np.argmax(AllData2D[:,i250])
            print('Max at time {}'.format(Times[imax].strftime('%Y %m %d:%H %M %S')))


        pp.colorbar(cont,ax=thisax,label=Var[i])

        if args['oplotmax']:
            def parabolic_peak_alt(profile, alts):
                i = np.argmax(profile)
                if i == 0 or i == len(profile) - 1:
                    return alts[i]
                y0, y1, y2 = profile[i-1], profile[i], profile[i+1]
                x0, x1, x2 = alts[i-1], alts[i], alts[i+1]
                denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
                a = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
                b = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom
                return -b / (2*a)
            alt_of_max = np.array([parabolic_peak_alt(AllData2D[t, :], Alts)
                                   for t in range(AllData2D.shape[0])])
            thisax.plot(Times, alt_of_max, color='black', linewidth=1.5)


    i += 1


    if args['cut'] != 'lt':
        pp.ylabel('Alt (km)')



if lsAveraging:
    pp.xlabel('Ls (degrees)')
else:
    pp.xlabel('Time (UT)')
    time_span = (AllTimes[-1] - AllTimes[0]).total_seconds() / 86400.0
    if time_span > 1.0:
        myFmt = mdates.DateFormatter("%m/%d")
        thisax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate(rotation=45, ha='center')
    else:
        myFmt = mdates.DateFormatter("%H:%M:%S")
        thisax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()

var_str = args['var'].replace(',', '_')
stime = AllTimes[0].strftime('%Y%m%d_%H%M%S')
if args['cut'] == 'loc':
    outfile = 'gitm_loc_lat{}_lon{}_var{}_{}.png'.format(plat, plon, var_str, stime)
elif args['cut'] == 'sza':
    outfile = 'gitm_sza_{}_{}_var{}_{}.png'.format(smin, smax, var_str, stime)
elif args['cut'] == 'lt':
    outfile = 'gitm_lt{}_lat{}_var{}_{}.png'.format(args['lt'], plat, var_str, stime)
else:
    outfile = 'gitm_{}_var{}_{}.png'.format(args['cut'], var_str, stime)
print('Writing file: {}'.format(outfile))
pp.savefig(outfile)

if args['savedata']:
    ncfile = outfile.replace('.png', '.nc')
    print('Writing data file: {}'.format(ncfile))
    reference_time = 'seconds since 1970-01-01 00:00:00 UTC'
    with nc.Dataset(ncfile, 'w', format='NETCDF4') as ds:
        ds.createDimension('alt', len(Alts))
        alt_var = ds.createVariable('alt', 'f4', ('alt',))
        alt_var.units = 'km'
        alt_var.long_name = 'Altitude'
        alt_var[:] = Alts

        if lsAveraging:
            ds.createDimension('ls', len(Times))
            xdim = 'ls'
            ls_var = ds.createVariable('ls', 'f4', ('ls',))
            ls_var.units = 'degrees'
            ls_var.long_name = 'Solar Longitude'
            ls_var[:] = Times
        else:
            ds.createDimension('time', len(Times))
            xdim = 'time'
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = reference_time
            time_var.calendar = 'standard'
            time_var[:] = nc.date2num(list(Times), units=reference_time, calendar='standard')
            ls_var = ds.createVariable('ls', 'f4', ('time',))
            ls_var.units = 'degrees'
            ls_var.long_name = 'Solar Longitude'
            ls_var[:] = AllLs

        for idx, ivar in enumerate(args['var'].split(',')):
            data = AllData[ivar]
            varname = 'var_{}'.format(ivar)
            if lineplot:
                v = ds.createVariable(varname, 'f8', (xdim,))
            else:
                v = ds.createVariable(varname, 'f8', (xdim, 'alt'))
            v.long_name = Var[idx]
            v[:] = data
