#!/usr/bin/env python

### Plot a GITM satellite file

import sys 
import numpy as np 
import re 
from matplotlib import pyplot as pp
from gitm_routines import *
import pandas as pd 
import ngims 
import rose
import marstiming as mt 

minalt = 80 

def get_args(argv):

    filelist = []
    var = -1
    help = False
    alog = False
    min = None 
    max = None 
    average = False
    stddev = False
    sats = None
    reactions = False

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

            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)


    args = {'filelist':filelist,
        'var':var,
        'help':help,
        'alog':alog,
        'min':min,
        'max':max,
        'average':average,
        'stddev':stddev,
        'sats':sats,
        'reactions':reactions,
    }

    return args

plotmaxden = False
args = get_args(sys.argv)
sats = args['sats']
currentsats = ['ngims','rose']
if sats and not sats in currentsats:
    print('Only available sats are: '+",".join(currentsats))
    args['help'] = True 

if args['stddev'] and not args['average']:
    print('You must specify -average if you are using -stddev')
    args['help'] = True

header = read_gitm_header(args["filelist"])

if args['var'] == -1:
    args['help'] = True

if (args["help"]):

    print('Plot a 1D GITM file (1D??? or a satellite file)')
    print('Usage : ')
    print('gitm_plot_satellites.py -var=N -alog ')
    print('       -help [*.bin or a file]')
    print('   -help : print this message')
    print('   -var=var1,var2,var3... : variable(s) to plot')
    print('   -alog : plot the log of the variable')
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -average: plot orbit averages')
    print('   -stddev: if using average, also plot stddev')
    print('   -sats=sats: overplot sat files. Current sat options')
    print('      are: ngims/rose')
    print('   -reactions: you are plotting reactions and you might want to plot the associated text')
    print('   At end, list the files you want to plot')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()

filelist = args['filelist']


vars = [0,1,2]

vars.extend([int(v) for v in args["var"].split(',')])
Var = [header['vars'][int(i)] for i in args['var'].split(',')]

varmap = {29:44,28:32,27:16,4:'CO2',6:'O',
7:'N2',9:'Ar',5:'CO',
}

varcmap = {27:'O$^+$',28:'O$_2^+$',29:'CO$_2^+$',
4:'CO$_2$',6:'O',7:'N$_2$',9:'Ar',5:'CO'}

fig,ax = pp.subplots()

minv = 9.e20
maxv = -9.e20

alldata = [] 
directories = [] 
i = 0


for file in filelist:
    match = re.match(r'(.*?)\/',filelist[i])
    if match:
        directories.append(match.group(1))
    data = read_gitm_one_file(file,vars)

    if i == 0:
        alts = data[2][0,0]/1000. #Assumes the altitude grid doesn't change with file
        iminalt = find_nearest_index(alts,minalt)
    alldata.append(data) #an array with all sat files data
    i+=1

df = pd.DataFrame(alldata)
#plot options depending on the dataset
linestyles = ['-','--','-.',':']
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

if not args['average']:
    for ifile in range(len(alldata)):
        ivar = 0
        for pvar in args["var"].split(','):
            pdata = alldata[ifile][int(pvar)][0,0,iminalt:]
            if args['alog']: 
                pdata= np.where(pdata > 0, np.log10(pdata), 0)
            if min(pdata) < minv:
                minv = min(pdata)
            if max(pdata) > maxv:
                maxv = max(pdata)

            if ndirs > 1:
                linestyle = dirmap[directories[ifile]]
            line, = pp.plot(pdata,alts[iminalt:],color=colors[ivar],ls=linestyle)
            if ndirs <= 1:
                if args['reactions']:
                    line.set_label(marsreactions[int(header['vars'][int(pvar)])])
                else:
                    line.set_label(name_dict[header["vars"][int(pvar)]])

            ivar +=1
    if ndirs <= 1 and sats:
        line.set_label('MGITM')
 
else: 
    meandata = df.mean()
    for pvar in args["var"].split(','):
        pdata = meandata[int(pvar)][0,0]
        if args['alog']: 
            pdata = np.log10(pdata)
        if min(pdata) < minv:
            minv = min(pdata)
        if max(pdata) > maxv:
            maxv = max(pdata)

        ax.plot(pdata,alts[iminalt:],'k',linewidth=2,label='MGITM') 

        if args['stddev']:
            tempdata = df[int(pvar)].to_numpy()
            newdata = np.zeros((len(df),np.shape(tempdata[0])[2]))
            for i in range(len(newdata)):
                newdata[i,:] = tempdata[i][0,0]
            stddata = np.std(newdata,0)
            if args['alog']: 

                stddata = np.log10(stddata)
            pp.fill_betweenx(alts[iminalt:],pdata-stddata,pdata+stddata)
if args['min'] is not None:
    mini = args['min']
else:
    mini = minv

if args['max'] is not None:
    maxi = args['max']
else:
    maxi = maxv

imaxden = np.argmax(pdata)
inearest = find_nearest_index(alts[iminalt:],270)
maxden = pdata[inearest]
if plotmaxden:
    pp.ax([-999,1e30],[alts[imaxden],alts[imaxden]],'r--')
# pp.plot([maxden,maxden],[0,300],'r--',alpha=.7)
pp.ylim([minalt,300])
pp.xlim([mini,maxi])

### Test the average 
def testave():
    sum = np.zeros(len(alldata[0][0][0,0]))
    for data in alldata:
        sum = sum + data[29][0,0]
        
    sum = sum/len(alldata)
    print('difference between averages is:')
    print(sum - meandata[29][0,0])

    vars = args["var"].split(',')

if sats:
    # satsdir = '/home/dpawlows/Docs/Research/MGITM-MAVENcomparison2023/DD2/NGIMS/'
    satsdir = '/media/dpawlows/Mars/NGIMS/2017/'
    start = alldata[0]['time'].strftime('%Y%m%d')
    end = alldata[-1]['time'].strftime('%Y%m%d')

    averageAltBin = 3.5 #km
    minalt = 100
    maxalt = 302
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
        if '+' in varcmap[vars[3]]:
            speciesColumn = 'ion_mass'
            qualityFlag = ['SCP','SC0']
            version = 'v08'
            dentype = 'ion'

        varlist = [varmap[v] for v in vars[3:]]
        files = ngims.getfiles(start,end,dentype=dentype,version=version,dir=satsdir)
        if len(files) == 0:
            print('No NGIMS files found!')

        if not args['average']:

            for fi in files:
                data = ngims.readNGIMS(fi)
                data = data[(data["alt"] < 350)]
                data = data[data["quality"].isin(qualityFlag)]

                for pvar in varlist:
                    searchVar = int(pvar) if dentype == 'ion' else pvar
                    newdf = data[(data[speciesColumn] == searchVar)]
                    if newdf.shape[0] == 0:
                        print("Error in ngims_plot_profile: Empty data frame from {}".format(fi))
                        exit(1)
                        
                    if inboundonly:
                        minalt = newdf['alt'].idxmin()
                        indices = list(newdf.index.values)
                        imin = indices.index(minalt)+1
                        newdf = newdf.loc[indices[0:imin]] #update the df with only inbound data
                    breakpoint()
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
                
                
                    data = ngims.readNGIMS(fi)
                    data = data[(data["alt"] < 350)]
                    data = data[data["quality"].isin(qualityFlag)]
                    newdf = data[(data[speciesColumn] == int(pvar))]

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
                data = rose.readRoseTab(f)
                newdf = data[(data['altitude'] >= minalt) & (data['altitude'] <= maxalt)]
                pp.plot(newdf['nelec'],newdf['altitude'],'.',markersize = 5,color='dimgrey')
        
        else:
            orbitavedensity = np.zeros((len(files),nbins-1))
            ifile = 0
            for f in files:
                data = rose.readRoseTab(f)
                newdf = data[(data['altitude'] >= minalt) & (data['altitude'] <= maxalt)]

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

if ndirs > 1:
   
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
pp.xlabel('Density')
# pp.xlabel('Production Rate (m$^{-3}s^{-1}$)')
# pp.xlabel('[e-] [m$^{-3}$]')
pp.ylabel('Altitude (km)')
msd = mt.getMarsSolarGeometry(data['time'])
LT = mt.getLTfromTime(data['time'],data[0][0,0,0]*180/np.pi)
SZA = mt.getSZAfromTime(msd,data[0][0,0,0]*180/np.pi, data[1][0,0,0]*180/np.pi)
pp.title(f"{data['time'].strftime("%Y%m%d-%H:%M UT")}\n{LT:.1f} LT, {SZA:.1f} SZA")
pp.savefig('plot.png')
# breakpoint()