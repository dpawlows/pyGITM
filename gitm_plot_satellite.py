#!/usr/bin/env python

### Plot a GITM satellite file

import sys
import numpy as np
import re
import os
import datetime as dt
from matplotlib import pyplot as pp
from gitm_routines import *
import pandas as pd
import ngims
import rose
import marstiming as mt
from concurrent.futures import ThreadPoolExecutor

minaltplot = 50
maxaltplot = 250
minalt = minaltplot
def find_homopause(n2, ar, alts):
    ratio = n2 / ar
    idx = np.where(ratio <= ratio[0]*1.5)[0]

    if len(idx) == 0:
        return None
    i_last = idx[-1]
    if i_last < len(alts) - 1:
        r0, r1 = ratio[i_last], ratio[i_last + 1]
        a0, a1 = alts[i_last], alts[i_last + 1]
        if r1 != r0:
            return a0 + (1.25 - r0) * (a1 - a0) / (r1 - r0)
        return a0
    return alts[i_last]


def compute_solar_geom(time, lon_rad, lat_rad):
    """Return local time and solar zenith angle for a given time and location.

    Parameters
    ----------
    time : datetime
        Observation time.
    lon_rad : float
        Longitude in radians.
    lat_rad : float
        Latitude in radians.

    Returns
    -------
    tuple of floats
        Local time (hours) and solar zenith angle (degrees).
    """

    lon = np.degrees(lon_rad)
    lat = np.degrees(lat_rad)
    msd = mt.getMarsSolarGeometry(time)
    lt = mt.getLTfromTime(time, lon)
    sza = mt.getSZAfromTime(msd, lon, lat)
    return lt, sza

def get_args(argv):

    filelist = []
    var = -1
    help = False
    alog = False
    min = None
    max = None
    minalt = None
    maxalt = None
    average = False
    stddev = False
    sats = None
    reactions = False
    mix = False
    press = False
    single = False
    grid = False
    oplot = False
    ps = False

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

            m = re.match(r'-minalt=(.*)',arg)
            if m:
                minalt = float(m.group(1))
                IsFound = 1

            m = re.match(r'-maxalt=(.*)',arg)
            if m:
                maxalt = float(m.group(1))
                IsFound = 1
            
            m = re.match(r'-h',arg)
            if m:
                help = True
                IsFound = 1

            m = re.match(r'-alog',arg)
            if m:
                alog = True
                IsFound = 1
            
            m = re.match(r'-average',arg)
            if m:
                average = True
                IsFound = 1

            m = re.match(r'-reactions',arg)
            if m:
                reactions = True
                IsFound = 1

            m = re.match(r'-stddev',arg)
            if m:
                stddev = True
                IsFound = 1

            m = re.match(r'-sats=(.*)',arg)
            if m:
                sats = m.group(1)
                IsFound = 1

            m = re.match(r'-mix',arg)
            if m:
                mix = True
                IsFound = 1

            m = re.match(r'-press',arg)
            if m:
                press = True
                IsFound = 1

            m = re.match(r'-single',arg)
            if m:
                single = True
                IsFound = 1

            m = re.match(r'-grid',arg)
            if m:
                grid = True
                IsFound = 1

            m = re.match(r'-oplot',arg)
            if m:
                oplot = True
                IsFound = 1

            m = re.match(r'-ps',arg)
            if m:
                ps = True
                IsFound = 1

            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)


    args = {'filelist':filelist,
        'var':var,
        'help':help,
        'alog':alog,
        'min':min,
        'max':max,
        'minalt':minalt,
        'maxalt':maxalt,
        'average':average,
        'stddev':stddev,
        'sats':sats,
        'reactions':reactions,
        'mix':mix,
        'press':press,
        'single':single,
        'grid':grid,
        'oplot':oplot,
        'ps':ps,
    }

    return args

plotmaxden = False
args = get_args(sys.argv)
if args['minalt'] is not None:
    minaltplot = args['minalt']
    minalt = args['minalt']
if args['maxalt'] is not None:
    maxaltplot = args['maxalt']
sats = args['sats']
currentsats = ['ngims','rose']
if sats and not sats in currentsats:
    print('Only available sats are: '+",".join(currentsats))
    args['help'] = True

if args['mix'] and sats:
    print('Satellite comparison not available with -mix option. Ignoring -sats.')
    sats = None

if args['press'] and sats:
    print('Satellite comparison not available with -press option. Ignoring -sats.')
    sats = None

if args['stddev'] and not args['average']:
    print('You must specify -average if you are using -stddev')
    args['help'] = True

header = read_gitm_header(args["filelist"])

if args['var'] == -1 and not args['mix']:
    args['help'] = True

if (args["help"]):

    print('Plot a 1D GITM file (1D??? or a satellite file)')
    print('Usage : ')
    print('gitm_plot_satellite.py -var=N -alog ')
    print('       -help [*.bin or a file]')
    print('   -help : print this message')
    print('   -var=var1,var2,var3... : variable(s) to plot')
    print('   -alog : plot the log of the variable')
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -minalt=alt: minimum altitude (km) to plot')
    print('   -maxalt=alt: maximum altitude (km) to plot')
    print('   -average: plot orbit averages')
    print('   -stddev: if using average, also plot stddev')
    print('   -sats=sats: overplot sat files. Current sat options')
    print('      are: ngims/rose')
    print('   -reactions: you are plotting reactions and you might want to plot the associated text')
    print('   -mix : plot mixing ratio of all neutral species')
    print('   -press : use pressure as vertical coordinate')
    print('   -single : force one plot per file even with a single variable')
    print('   -grid : overlay a grid on the plots')
    print('   -oplot : overplot multiple files and variables on a single figure')
    print('   -ps : save plot as a PostScript file instead of PNG')
    print('   At end, list the files you want to plot')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()

filelist = args['filelist']

if args['mix']:
    for f in filelist:
        if 'ALL' not in f.upper():
            print(f'Warning: {f} may not contain neutral species; ALL output required.')

vars = [0,1,2]
kb = 1.380649e-23
temp_index = None
species_inds = []
if args['press']:
    for tname in ['Temperature', 'Tn']:
        if tname in header['vars']:
            temp_index = header['vars'].index(tname)
            break
    if temp_index is None:
        print('Temperature not available to calculate pressure')
        exit(1)
    if temp_index not in vars:
        vars.append(temp_index)
    species_inds = [i for i, v in enumerate(header['vars']) if v.startswith('[')]
    for idx in species_inds:
        if idx not in vars:
            vars.append(idx)

if args['mix']:
    neutral_vars = []
    for i, v in enumerate(header["vars"]):
        if v.startswith('[') and '!U+!N' not in v and v != '[e-]':
            neutral_vars.append(i)
    if len(neutral_vars) == 0:
        print('No neutral species found. Ensure ALL output is used.')
        exit()
    vars.extend(neutral_vars)
    plot_vars = neutral_vars
    n2_idx = header['vars'].index('[N!D2!N]') if '[N!D2!N]' in header['vars'] else None
    ar_idx = header['vars'].index('[Ar]') if '[Ar]' in header['vars'] else None
    homopause_alt = None
    if n2_idx is None or ar_idx is None:
        print('N2 or Ar not found; homopause will not be computed.')
else:
    plot_vars = [int(v) for v in args["var"].split(',')]
    vars.extend(plot_vars)
    homopause_alt = None

Var = [header['vars'][i] for i in plot_vars]

varmap = {30:44,29:32,28:16,31:14,32:30,33:28,34:12,4:'CO2',6:'O',
7:'N2',9:'Ar',5:'CO',
}

varcmap = {28:'O+',29:'O$_2$+',30:'CO$_2$+',31:'N$_2$+',32:'NO+',33:'CO+',34:'C+',
4:'CO$_2$',6:'O',7:'N$_2$',9:'Ar',5:'CO'}

fig,ax = pp.subplots()

minv = 9.e20
maxv = -9.e20


# Parallel loading of files
alldata = []
pressures = []
directories = []

def _load_file(fname):
    match = re.match(r'(.*?)/', fname)
    directory = match.group(1) if match else None
    data = read_gitm_one_file(fname, vars)
    pressure = None
    if args['press']:
        temp = data[temp_index]
        dens = np.zeros_like(temp)
        for idx in species_inds:
            dens += data[idx]
        pressure = (dens * kb * temp)[0,0]
    return fname, data, directory, pressure

with ThreadPoolExecutor() as executor:
    results = list(executor.map(_load_file, filelist))

for fname, data, directory, pressure in results:
    if directory:
        directories.append(directory)
    if args['press']:
        pressures.append(pressure)
    if not alldata:
        alts = data[2][0,0]/1000. #Assumes the altitude grid doesn't change with file
        iminalt = find_nearest_index(alts,minalt)
    alldata.append(data)

df = pd.DataFrame(alldata)
#plot options depending on the dataset
linestyles = ['-','--','-.',':']
file_linestyles = []
ndirs = 0
if directories:
    dirmap = {}
    i = 0
    for key in set(directories):
        dirmap[key] = linestyles[i]
        i+=1
    ndirs = len(dirmap)

if ndirs <= 1:
    linestyle = '-'

cmap = pp.get_cmap('tab20')
colors = [cmap(i) for i in range(20)]

if args['press']:
    yarrays = pressures
else:
    yarrays = [alts for _ in alldata]

if not args['average']:
    if len(alldata) > 1 and (len(plot_vars) > 1 or args['single']) and not args['oplot']:
        for ifile, data in enumerate(alldata):
            fig, ax = pp.subplots()
            minv = np.inf
            maxv = -np.inf
            if args['press']:
                yvals = yarrays[ifile][iminalt:]
                ax.set_yscale('log')
                ax.set_ylim([yvals.max(), yvals.min()])
            else:
                yvals = yarrays[ifile][iminalt:]
                ax.set_ylim([minaltplot, maxaltplot])
            ivar = 0
            if args['mix']:
                total = np.zeros_like(data[plot_vars[0]][0,0,iminalt:])
                for idx in plot_vars:
                    total += data[idx][0,0,iminalt:]
                for pvar in plot_vars:
                    pdata = data[pvar][0,0,iminalt:] / total
                    if args['alog']:
                        pdata = np.where(pdata > 0, np.log10(pdata), 0)
                    minv = min(minv, np.min(pdata))
                    maxv = max(maxv, np.max(pdata))
                    line, = ax.plot(pdata, yvals, color=colors[ivar])
                    label = name_dict.get(header['vars'][pvar], header['vars'][pvar])
                    line.set_label(label)
                    ivar += 1
            else:
                for pvar in plot_vars:
                    pdata = data[pvar][0,0,iminalt:]
                    if args['alog']:
                        pdata = np.where(pdata > 0, np.log10(pdata), 0)
                    minv = min(minv, np.min(pdata))
                    maxv = max(maxv, np.max(pdata))
                    line, = ax.plot(pdata, yvals, color=colors[ivar])
                    if args['reactions']:
                        line.set_label(marsreactions[int(header['vars'][pvar])])
                    else:
                        label = name_dict.get(header['vars'][pvar], header['vars'][pvar])
                        line.set_label(label)
                    ivar += 1
            mini = args['min'] if args['min'] is not None else minv
            maxi = args['max'] if args['max'] is not None else maxv
            ax.set_xlim([mini, maxi])
            if args['mix']:
                ax.set_xlabel('Mixing Ratio')
            else:
                ax.set_xlabel('Density')
            ax.set_ylabel('Pressure (Pa)' if args['press'] else 'Altitude (km)')
            if args['grid']:
                ax.grid(True)
            if args['mix'] and n2_idx is not None and ar_idx is not None:
                n2_data = data[n2_idx][0,0,iminalt:]
                ar_data = data[ar_idx][0,0,iminalt:]
                homopause_alt = find_homopause(n2_data, ar_data, alts[iminalt:])
                if homopause_alt is not None and not args['press']:
                    ax.text(0.95,0.95,f"homopause: {homopause_alt:.1f} km", transform=ax.transAxes, ha='right', va='top')
            LT, SZA = compute_solar_geom(data['time'], data[0][0,0,0], data[1][0,0,0])
            ax.set_title(f"{data['time'].strftime('%Y%m%d-%H:%M UT')}\n{LT:.1f} LT, {SZA:.1f} SZA")
            ax.legend(loc='upper left', frameon=False)
            var_list = args['var'].split(',') if isinstance(args['var'], str) and args['var'] != -1 else []
            if args['mix']:
                svar = 'mix'
            elif len(var_list) > 1:
                svar = 'multi'
            elif len(var_list) == 1:
                svar = var_list[0]
                if svar.isdigit():
                    svar = f"{int(svar):02d}"
                else:
                    svar = svar.replace('/', '_')
            else:
                svar = 'novar'
            prefix = 'presssatellite' if args['press'] else 'satellite'
            ext = 'ps' if args['ps'] else 'png'
            outfile = f"{prefix}_var{svar}_{data['time'].strftime('%y%m%d_%H%M%S')}.{ext}"
            print(f"Writing to file: {outfile}")
            pp.savefig(outfile)
            pp.close(fig)
        sys.exit(0)
    var_colors = {pvar: colors[i % len(colors)] for i, pvar in enumerate(plot_vars)}
    file_linestyles = [linestyles[i % len(linestyles)] for i in range(len(alldata))]

    for ifile, data in enumerate(alldata):
        linestyle = file_linestyles[ifile]
        if args['mix']:
            total = np.zeros_like(data[plot_vars[0]][0,0,iminalt:])
            for idx in plot_vars:
                total += data[idx][0,0,iminalt:]
            for pvar in plot_vars:
                pdata = data[pvar][0,0,iminalt:] / total
                if args['alog']:
                    pdata = np.where(pdata > 0, np.log10(pdata), 0)
                if min(pdata) < minv:
                    minv = min(pdata)
                if max(pdata) > maxv:
                    maxv = max(pdata)

                line, = pp.plot(pdata, yarrays[ifile][iminalt:], color=var_colors[pvar], ls=linestyle)
                if ifile == 0 and ndirs <= 1:
                    label = name_dict.get(header["vars"][pvar], header["vars"][pvar])
                    line.set_label(label)
        else:
            for pvar in plot_vars:
                pdata = data[pvar][0,0,iminalt:]
                if args['alog']:
                    pdata= np.where(pdata > 0, np.log10(pdata), 0)
                if min(pdata) < minv:
                    minv = min(pdata)
                if max(pdata) > maxv:
                    maxv = max(pdata)

                line, = pp.plot(pdata, yarrays[ifile][iminalt:], color=var_colors[pvar], ls=linestyle)
                if ifile == 0 and ndirs <= 1:
                    if args['reactions']:
                        line.set_label(marsreactions[int(header['vars'][pvar])])
                    else:
                        try:
                            line.set_label(name_dict[header["vars"][pvar]])
                        except:
                            line.set_label(header["vars"][pvar])
    if ndirs <= 1 and sats:
        line.set_label('MGITM')

    if args['mix'] and n2_idx is not None and ar_idx is not None:
        n2_data = alldata[0][n2_idx][0,0,iminalt:]
        ar_data = alldata[0][ar_idx][0,0,iminalt:]
        homopause_alt = find_homopause(n2_data, ar_data, alts[iminalt:])

else:
    meandata = df.mean()
    if args['press']:
        ymean = np.mean(np.array(yarrays), axis=0)
    else:
        ymean = yarrays[0]
    if args['mix'] and n2_idx is not None and ar_idx is not None:
        n2_data = meandata[n2_idx][0,0,iminalt:]
        ar_data = meandata[ar_idx][0,0,iminalt:]
        homopause_alt = find_homopause(n2_data, ar_data, alts[iminalt:])
    if args['mix']:
        total = np.zeros_like(meandata[plot_vars[0]][0,0,iminalt:])
        for pvar in plot_vars:
            total += meandata[pvar][0,0,iminalt:]
        ivar = 0
        for pvar in plot_vars:
            pdata = meandata[pvar][0,0,iminalt:] / total
            if args['alog']:
                pdata = np.where(pdata > 0, np.log10(pdata), 0)
            if min(pdata) < minv:
                minv = min(pdata)
            if max(pdata) > maxv:
                maxv = max(pdata)
            ax.plot(pdata,ymean[iminalt:],color=colors[ivar],linewidth=2,
                    label=name_dict.get(header["vars"][pvar], header["vars"][pvar]))
            if args['stddev']:
                tempdata = df[pvar].to_numpy()
                newdata = np.zeros((len(df),np.shape(tempdata[0])[2]))
                for i in range(len(newdata)):
                    newdata[i,:] = tempdata[i][0,0,iminalt:]
                stddata = np.std(newdata,0)
                if args['alog']:
                    stddata = np.log10(stddata)
                pp.fill_betweenx(ymean[iminalt:],pdata-stddata,pdata+stddata)
            ivar +=1
    else:
        for pvar in plot_vars:
            pdata = meandata[pvar][0,0]
            if args['alog']:
                pdata = np.log10(pdata)
            if min(pdata) < minv:
                minv = min(pdata)
            if max(pdata) > maxv:
                maxv = max(pdata)

            ax.plot(pdata,ymean[iminalt:],'k',linewidth=2,label='MGITM')

            if args['stddev']:
                tempdata = df[pvar].to_numpy()
                newdata = np.zeros((len(df),np.shape(tempdata[0])[2]))
                for i in range(len(newdata)):
                    newdata[i,:] = tempdata[i][0,0]
                stddata = np.std(newdata,0)
                if args['alog']:

                    stddata = np.log10(stddata)
                pp.fill_betweenx(ymean[iminalt:],pdata-stddata,pdata+stddata)
if args['min'] is not None:
    mini = args['min']
else:
    mini = minv

if args['max'] is not None:
    maxi = args['max']
else:
    maxi = maxv

if args['press']:
    if args['average']:
        yvals = ymean[iminalt:]
    else:
        yvals = yarrays[0][iminalt:]
    ax.set_yscale('log')
    pp.ylim([yvals.max(), yvals.min()])
else:
    imaxden = np.argmax(pdata)
    inearest = find_nearest_index(alts[iminalt:],270)
    maxden = pdata[inearest]
    if plotmaxden:
        pp.ax([-999,1e30],[alts[imaxden],alts[imaxden]],'r--')
    # pp.plot([maxden,maxden],[0,300],'r--',alpha=.7)
    pp.ylim([minaltplot,maxaltplot])
pp.xlim([mini,maxi])

### Test the average 
def testave():
    sum = np.zeros(len(alldata[0][0][0,0]))
    for data in alldata:
        sum = sum + data[29][0,0]
        
    sum = sum/len(alldata)
    print('difference between averages is:')
    print(sum - meandata[29][0,0])

    vars = plot_vars

if sats:
    satsdir = '/home/dpawlows/Docs/Research/MGITM-MAVENcomparison2023/DD2/NGIMS/'
    #satsdir = '/media/dpawlows/Mars/NGIMS/2015 /'
    start = alldata[0]['time'].strftime('%Y%m%d')
    end = alldata[-1]['time'].strftime('%Y%m%d')

    averageAltBin = 3.5 #km
    minalt = minaltplot
    maxalt = maxaltplot
    altbins = np.arange(minalt,maxalt,averageAltBin)
    nbins = len(altbins)
    totaldata = np.zeros(nbins-1)
    counts = np.zeros(nbins-1)

    if sats=='ngims':  

        speciesColumn = 'species'
        qualityFlag = ['OV','OU']
        version = 'v08'
        dentype = 'csn'
        inboundonly = True 
        if '+' in varcmap.get(plot_vars[0], ''):
            speciesColumn = 'ion_mass'
            qualityFlag = ['SCP','SC0']
            version = 'v08'
            dentype = 'ion'

        varlist = [varmap[v] for v in plot_vars if v in varmap]
        files = ngims.getfiles(start,end,dentype=dentype,version=version,dir=satsdir)
        if len(files) == 0:
            print('No NGIMS files found!')
        else:
            model_time = alldata[0]['time']
            def _file_time(fname):
                m = re.search(r'_(\d{8}T\d{6})_', os.path.basename(fname))
                if m:
                    return dt.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')
                return None
            file_times = [_file_time(f) for f in files]
            if any(t is not None for t in file_times):
                diffs = [abs((t - model_time).total_seconds()) if t else float('inf') for t in file_times]
                files = [files[int(np.argmin(diffs))]]

        if not args['average']:

            for fi in files:
                satdata = ngims.readNGIMS(fi)
                satdata = satdata[(satdata["alt"] < 350)]
                satdata = satdata[satdata["quality"].isin(qualityFlag)]

                for pvar in varlist:
                    searchVar = int(pvar) if dentype == 'ion' else pvar
                    newdf = satdata[(satdata[speciesColumn] == searchVar)]
                    if newdf.shape[0] == 0:
                        print("Error in ngims_plot_profile: Empty data frame from {}".format(fi))
                        exit(1)

                    if inboundonly:
                        minalt = newdf['alt'].idxmin()
                        indices = list(newdf.index.values)
                        imin = indices.index(minalt)+1
                        newdf = newdf.loc[indices[0:imin]] #update the df with only inbound data
                    if args['alog']:
                        density = np.log10(newdf.loc[newdf["alt"] < maxalt,"abundance"]*1e6)
                    else:
                        density = newdf.loc[newdf["alt"] < maxalt,"abundance"]*1e6

                    starred = ''
                    temp = newdf['quality'].isin(qualityFlag)
                    if temp.values.sum() / newdf.shape[0] > .75:
                        starred = '*'
                    
                    altitude = newdf.loc[newdf["alt"] < maxalt,'alt'].values
                    line, = ax.plot(density,altitude,'.',markersize = 5,color='dimgrey')
                    # if allions:
                    #     line.set_label(varmap[pvar])
                    # else:
                    #     line.set_label(str(data.orbit.values[0])+starred)
            line.set_label('NGIMS')

        else:
            orbitavedensity = np.zeros((len(files),nbins-1))
            for pvar in varlist:
                ifile = 0

                for fi in files:


                    satdata = ngims.readNGIMS(fi)
                    satdata = satdata[(satdata["alt"] < 350)]
                    satdata = satdata[satdata["quality"].isin(qualityFlag)]
                    newdf = satdata[(satdata[speciesColumn] == int(pvar))]

                    if inboundonly:
                            minalt = newdf['alt'].idxmin()
                            indices = list(newdf.index.values)
                            imin = indices.index(minalt)+1
                            newdf = newdf.loc[indices[0:imin]]

                    for ibin in range(len(altbins)-1):
                        lower = altbins[ibin] 
                        upper = altbins[ibin+1] 
                        tempdata = newdf.loc[(newdf["alt"] <= upper) & \
                            (newdf["alt"] > lower)]
                        orbitavedensity[ifile,ibin] = np.nanmean(tempdata['abundance'].to_numpy())*1.e6
                        count = tempdata['abundance'].count()

                        if count > 0:
                            counts[ibin] = counts[ibin] + count 
                            totaldata[ibin] = totaldata[ibin] + tempdata['abundance'].sum()*1.e6
                    
                    ifile += 1


                totaldata = totaldata/counts
                density2 = np.nanmean(orbitavedensity,axis=0)
                stddevdata = np.std(orbitavedensity,axis=0)
                averagebins = (altbins[0:-1] + altbins[1:])/2.
                
                pp.plot(totaldata,averagebins,'k--',linewidth=1,label='NGIMS')
                pp.fill_betweenx(averagebins,totaldata-stddevdata,totaldata+stddevdata,\
                    color='lightgrey',alpha=.8)

    if sats=='rose':
        satsdir = '/home/dpawlows/Docs/Research/MGITM-MAVENcomparison2023/ROSE/'
        files = rose.getRoseFiles(start,end,dir=satsdir)

        if not args['average']:
            for f in files:
                satdata = rose.readRoseTab(f)
                newdf = satdata[(satdata['altitude'] >= minalt) & (satdata['altitude'] <= maxalt)]
                pp.plot(newdf['nelec'],newdf['altitude'],'.',markersize = 5,color='dimgrey')

        else:
            orbitavedensity = np.zeros((len(files),nbins-1))
            ifile = 0
            for f in files:
                satdata = rose.readRoseTab(f)
                newdf = satdata[(satdata['altitude'] >= minalt) & (satdata['altitude'] <= maxalt)]

                for ibin in range(len(altbins)-1):
                    lower = altbins[ibin] 
                    upper = altbins[ibin+1] 
                    tempdata = newdf.loc[(newdf["altitude"] <= upper) & \
                        (newdf["altitude"] > lower)]
                    orbitavedensity[ifile,ibin] = np.nanmean(tempdata['nelec'].to_numpy())

                ifile += 1

            density2 = np.nanmean(orbitavedensity,axis=0)
            stddevdata = np.std(orbitavedensity,axis=0)
            averagebins = (altbins[0:-1] + altbins[1:])/2.
            
            pp.plot(density2,averagebins,'k--',linewidth=1,label='ROSE')
            pp.fill_betweenx(averagebins,density2-stddevdata,density2+stddevdata,\
                color='lightgrey',alpha=.8)

if homopause_alt is not None:
    ax.text(0.95,0.95,f"homopause: {homopause_alt:.1f} km", transform=ax.transAxes,
            ha='right', va='top')

if args['oplot'] and len(alldata) > 1:
    # Shrink the plot area to make room for the variable legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    file_handles = [pp.Line2D([], [], color='k', linestyle=file_linestyles[i])
                    for i in range(len(alldata))]
    file_labels = [os.path.splitext(os.path.basename(f))[0] for f in filelist]
    file_legend = ax.legend(file_handles, file_labels, loc='upper right',
                            frameon=False)

    var_handles = []
    var_labels = []
    for pvar in plot_vars:
        if args['reactions']:
            label = marsreactions[int(header['vars'][pvar])]
        else:
            label = name_dict.get(header['vars'][pvar], header['vars'][pvar])
        var_handles.append(pp.Line2D([], [], color=var_colors[pvar]))
        var_labels.append(label)

    ax.add_artist(file_legend)
    ax.legend(var_handles, var_labels, loc='lower left',
              bbox_to_anchor=(1, 0.5), frameon=False)
elif ndirs > 1:
    handles = [pp.Line2D([], [], linestyle=value) for value in dirmap.values()]
    pp.legend(handles, dirmap.keys(),loc='upper right',frameon=False)
else:
    if args['reactions']:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        pp.legend(bbox_to_anchor=[1.04,1],loc='upper left',frameon=False,prop={'size': 10})
    else:
        pp.legend(loc='upper left',frameon=False)
# pp.xlabel(name_dict[header['vars'][vars[3]]])
if args['mix']:
    pp.xlabel('Mixing Ratio')
else:
    pp.xlabel('Density')
# pp.xlabel('Production Rate (m$^{-3}s^{-1}$)')
# pp.xlabel('[e-] [m$^{-3}$]')
if args['press']:
    pp.ylabel('Pressure (Pa)')
else:
    pp.ylabel('Altitude (km)')
if args['grid']:
    pp.grid(True)

LT, SZA = compute_solar_geom(data['time'], data[0][0,0,0], data[1][0,0,0])
pp.title(f"{data['time'].strftime('%Y%m%d-%H:%M UT')}\n{LT:.1f} LT, {SZA:.1f} SZA")
# Build an informative filename similar to gitm_plot_alt_profile
var_list = args['var'].split(',') if isinstance(args['var'], str) and args['var'] != -1 else []
if args['mix']:
    svar = 'mix'
elif var_list:
    svar = var_list[0]
    if svar.isdigit():
        svar = f"{int(svar):02d}"
    else:
        svar = svar.replace('/', '_')
    if len(var_list) > 1:
        svar = 'multi'
else:
    svar = 'novar'
prefix = 'presssatellite' if args['press'] else 'satellite'
ext = 'ps' if args['ps'] else 'png'
outfile = f"{prefix}_var{svar}_{data['time'].strftime('%y%m%d_%H%M%S')}.{ext}"
print(f"Writing to file: {outfile}")
pp.savefig(outfile)
# breakpoint()
