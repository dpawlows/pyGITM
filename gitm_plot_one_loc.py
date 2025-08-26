#!/usr/bin/env python
#Plots a altitude and time for a single location
from glob import glob
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib import  ticker
from gitm_routines import *
import sys
from marstiming import getMTfromTime


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

    filelist = []
    IsLog = 0
    diff = '0'
    var = 15
    alt = 400.0
    lon = -100.0
    lat = -100.0
    alt = -100.0
    cut = 'loc'
    smin = -100.0
    smax = -100.0
    lt = -100
    average = -100.0
    mini = None
    maxi = None

    help = 0


    for arg in argv:

        IsFound = 0

        if (not IsFound):

            m = re.match(r'-var=(.*)',arg)
            if m:
                var = m.group(1)
                IsFound = 1

            m = re.match(r'-diff=(.*)',arg)
            if m:
                diff = m.group(1)
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

            m = re.match(r'-lt=(.*)',arg)
            if m:
                lt = int(m.group(1))
                IsFound = 1

            m = re.match(r'-average=(.*)',arg)
            if m:
                average = int(m.group(1))
                IsFound = 1

            m = re.match(r'-smin=(.*)',arg)
            if m:
                smin = int(m.group(1))
                IsFound = 1   

            m = re.match(r'-smax=(.*)',arg)
            if m:
                smax = int(m.group(1))
                IsFound = 1

            m = re.match(r'-mini=(.*)',arg)
            if m:
                mini = float(m.group(1))
                IsFound = 1   

            m = re.match(r'-maxi=(.*)',arg)
            if m:
                maxi = float(m.group(1))
                IsFound = 1

            m = re.match(r'-alog',arg)
            if m:
                IsLog = 1
                IsFound = 1

            m = re.match(r'-h',arg)
            if m:
                help = 1
                IsFound = 1


            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)

    args = {'filelist':filelist,
            'var':var,
            'help':help,
            'lat':lat,
            'lon':lon,
            'alt':alt,
            'IsLog':IsLog,
            'cut':cut,
            'smin':smin,
            'smax':smax,
            'diff':diff,
            'lt':lt,
            'average':average,
            'mini':mini,
            'maxi':maxi}

    return args

args = get_args(sys.argv)
header = read_gitm_header(args["filelist"])
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
    print('   -diff=backgroundFiles: plot the difference between 2 sets of files')
    print('   Non-KW args: files.')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()

filelist = args["filelist"]
nFiles = len(filelist)
calcsza = False
plotsza = False
if nFiles < 2:
    print('Please enter multiple files')
    exit(1)
try:
    iSZA = header["vars"].index('SolarZenithAngle')
    vars = [0,1,2,iSZA]

except: 
    calcsza = True
    header["vars"].append('SolarZenithAngle')
    iSZA = header["vars"].index('SolarZenithAngle')
    vars = [0,1,2]


# Sort filenames based on year
# Regular expression pattern to match 3D???_t', and a date and time stamp
pattern = r'(.*)?3D..._t(\d{6})_(\d{6})'

# sorted_filenames = sorted(filelist, \
#     key=lambda x: extract_year(x,pattern) if extract_year(x,pattern) is not None else float('inf'))

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


vars.extend([int(v) for v in args["var"].split(',')])
Var = [header['vars'][int(i)] for i in args['var'].split(',')]
nvars = len(args['var'].split(','))

#We want to store data for multiple variables, so we use a dict where var indices are the keys
AllData = {a:[] for a in args['var'].split(',')}
sum = []
AllData2D = []
AllAlts = []
AllTimes = []
AllSZA = []
j = 0
indexDayStart = []
newday = True 

for file in filelist:

    data = read_gitm_one_file(file, vars)
    AllTimes.append(data["time"])
    
    if (j == 0):    
        [nLons, nLats, nAlts] = data[0].shape
        Alts = data[2][0][0]/1000.0
        Lons = data[0][:,0,0]*rtod
        Lats = data[1][0,:,0]*rtod

        ialt1 = find_nearest_index(Alts,minalt)
        ialt2 = find_nearest_index(Alts,maxalt)

    if calcsza:
        data[iSZA] = np.zeros((nLons,nLats,nAlts))
        ilon = 0
        for lon in Lons:
            ilat = 0
            for lat in Lats:
                thissza = calculate_sza(lat,lon,AllTimes[-1])
                data[iSZA][ilat,ilon,:] = thissza

                ilat += 1
            ilon += 1
        if plotsza:
            Lo,La = np.meshgrid(Lons,Lats)
            pp.figure()
            cont = pp.contourf(Lo[1:-1,1:-1],La[1:-1,1:-1],data[iSZA][1:-1,1:-1,0],levels=30)
            pp.colorbar(cont)
            file = 'sza_{}.png'.format(AllTimes[-1].strftime("%Y%m%d:%H%M%S"))
            print('Writing file {}...'.format(file))
            pp.savefig(file)
    if diff:
        stime = str(AllTimes[-1].year)[2:]+str(AllTimes[-1].month).rjust(2,'0')+str(AllTimes[-1].day).rjust(2,'0')+\
            '_'+str(AllTimes[-1].hour).rjust(2,'0')+str(AllTimes[-1].minute).rjust(2,'0')
        bFile = [i for i in backgroundFilelist if stime in i][0]
        if bFile == '':
            #It is possible that we don't have an output file at the same time.
            print('Missing background file corresponding to: {}'.format(file))
            exit(1)
        background = read_gitm_one_file(bFile,vars)
        
    if args['cut'] == 'loc':
        ilon = find_nearest_index(Lons,plon)
        ilat = find_nearest_index(Lats,plat)
        for ivar in args['var'].split(','):
            if diff:
                temp = (data[int(ivar)][ilon,ilat,ialt1:ialt2+1]-background[int(ivar)][ilon,ilat,ialt1:ialt2+1])/ \
                background[int(ivar)][ilon,ilat,ialt1:ialt2+1]*100.0
            else:
                temp = data[int(ivar)][ilon,ilat,ialt1:ialt2+1]          

            AllData[ivar].append(temp)


    if args['cut'] == 'sza':        
        AllSZA.append(data[iSZA][:,:,0])
        mask = (AllSZA[-1] >= smin) & (AllSZA[-1] <= smax ) 
        for ivar in args['var'].split(','):
            if diff:

                #Calculate the mean of both sets of data and then calculate the percent difference.
                if ivar == '2':
                    mean1 = (data[6][:,:,ialt1:ialt2+1][mask].mean(axis=0)/\
                             data[4][:,:,ialt1:ialt2+1][mask].mean(axis=0))
                    mean2 = (background[6][:,:,ialt1:ialt2+1][mask].mean(axis=0)/\
                             background[4][:,:,ialt1:ialt2+1][mask].mean(axis=0))
                else:
                    mean1 = data[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)
                    mean2 = background[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)

                temp = (mean1-mean2)/mean2*100.

            else:
                temp = data[int(ivar)][:,:,ialt1:ialt2+1][mask].mean(axis=0)

            AllData[ivar].append(temp)
            # AllData[ivar].append(temp[mask].mean(axis=0))

    if args['cut'] == 'lt':
        marstime = getMTfromTime([data['time'].year,data['time'].month,data['time'].day,\
            data['time'].hour,data['time'].minute,data['time'].second])

        #subsolarlon is in degrees west so convert to east first
        subsolarlon = 360 - marstime.subSolarLon
        ltdiff = args['lt'] - 12  #subsolar is at 12:00 LT
        plon = (subsolarlon + ltdiff*360/24) % 360
        if lineplot:
            ialt = find_nearest_index(Alts,palt)

        ilon = find_nearest_index(Lons,plon)
        ilat = find_nearest_index(Lats,plat)
        tempData =  {a:[] for a in args['var'].split(',')}

        for ivar in args['var'].split(','):
            if lineplot:

                if diff:
                    AllData[ivar].append((data[int(ivar)][ilon,ilat,ialt] -
                                         background[int(ivar)][ilon,ilat,ialt])/\
                                         background[int(ivar)][ilon,ilat,ialt]*100.0)
                else:
                    AllData[ivar].append(data[int(ivar)][ilon,ilat,ialt])                    
            else:

                if diff:
                    temp = (data[int(ivar)][ilon,ilat,ialt1:ialt2+1]-background[int(ivar)][ilon,ilat,ialt1:ialt2+1])/ \
                        background[int(ivar)][ilon,ilat,ialt1:ialt2+1]*100.0
                else:
                    temp = data[int(ivar)][ilon,ilat,ialt1:ialt2+1]          

                if averaging:
                    if newday:
                        sum = temp
                        newday = False
                        indexDayStart.append(j)
                        aveTStart = AllTimes[j]
                    else:
                        sum = sum + temp 
                    if (data['time'] - aveTStart).total_seconds() > marsDay:
                        newday = True
                        AllData[ivar].append(sum/(j-indexDayStart[-1]+1))
                        sum = []
                        
                else:
                    AllData[ivar].append(temp)


    j+=1


for ivar in args['var'].split(','):
    AllData[ivar] = np.array(AllData[ivar])

if args['cut']  == 'sza':
    AllSZA = np.array(AllSZA)

Alts = Alts[ialt1:ialt2+1]

cmap = 'plasma'
i=0

if averaging:
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

    mini = hcmini[i]
    maxi = hcmaxi[i]
    if lineplot:
         thisax.plot(Times,AllData2D)
         thisax.set_ylim([mini,maxi])
         thisax.set_ylabel(Var[i])
         pp.tight_layout()
    else:
        levels = np.linspace(mini,maxi,30)
        #cont = thisax.contourf(Times,Alts,np.transpose(AllData2D),levels=levels,cmap=cmap)
        cont = thisax.contourf(Times,Alts,np.transpose(AllData2D),levels=levels,cmap=cmap,\
                locator=ticker.LogLocator())    
 
        if int(ivar) == 3:

            i250 = np.argmin(np.abs(Alts-250))
            imax = np.argmax(AllData2D[:,i250])
            print('Max at time {}'.format(Times[imax].strftime('%Y %m %d:%H %M %S')))

            
        pp.colorbar(cont,ax=thisax,label=Var[i])


    i += 1
    
    
    if args['cut'] != 'lt':
        pp.ylabel('Alt (km)')

    # if i < len(Var)-1:
    #     ax.get_xaxis().set_ticklabels([])
    #     breakpoint()



pp.xlabel('Time (UT)')
myFmt = mdates.DateFormatter("%H:%M:%S")
thisax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()

pp.savefig('plot.png')
