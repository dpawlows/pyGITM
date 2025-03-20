#!/usr/bin/env python

### Plot a GITM satellite file

import sys 
import numpy as np 
import re 
from matplotlib import pyplot as pp
from gitm_routines import *
import pandas as pd 


minalt = 100 

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
            
            m = re.match(r'-stddev',arg)
            if m:
                stddev = True
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

    }

    return args

plotmaxden = False
args = get_args(sys.argv)

if args['stddev'] and not args['average']:
    print('You must specify -average if you are using -stddev')
    args['help'] = True

header = read_gitm_header(args["filelist"])

if args['var'] == -1:
    args['help'] = True

if (args["help"]):

    print('Plot many 1D GITM files (1D??? or a satellite file)')
    print('Usage : ')
    print('gitm_plot_many_lines.py -var=N -alog ')
    print('       -help [*.bin or a file]')
    print('   -help : print this message')
    print('   -var=var : variable to plot')
    print('   -alog : plot the log of the variable')
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -average: plot orbit averages')
    print('   -stddev: if using average, also plot stddev')
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
pvar = int(args['var'])

varmap = {29:44,28:32,27:16,4:'CO2',6:'O',
7:'N2',9:'Ar',5:'CO',
}

varcmap = {27:'O$^+$',28:'O$_2^+$',29:'CO$_2^+$',
4:'CO$_2$',6:'O',7:'N$_2$',9:'Ar',5:'CO'}

fig = pp.figure() 

minv = 9.e20
maxv = -9.e20

alldata = [] 
directories = [] 
i = 0
linestyle='-'
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

colors = ['blue','green','orange','red','yellow']
colors = iter(pp.cm.rainbow(np.linspace(0, 1, len(alldata))))
if not args['average']:
    for ifile in range(len(alldata)):
        ivar = 0

        pdata = alldata[ifile][int(pvar)][0,0,iminalt:]
        if args['alog']: 
            pdata = np.log10(pdata)
        if min(pdata) < minv:
            minv = min(pdata)
        if max(pdata) > maxv:
            maxv = max(pdata)
            

        line, = pp.plot(pdata,alts[iminalt:],color=next(colors),ls=linestyle)
 
else: 
    meandata = df.mean()

    pdata = meandata[int(pvar)][0,0]
    if args['alog']: 
        pdata = np.log10(pdata)
    if min(pdata) < minv:
        minv = min(pdata)
    if max(pdata) > maxv:
        maxv = max(pdata)

    pp.plot(pdata,alts[iminalt:],'k',linewidth=2,label='MGITM') 

    if args['stddev']:
        tempdata = df[int(pvar)].to_numpy()
        newdata = np.zeros((len(df),np.shape(tempdata[0])[2]))
        for i in range(len(newdata)):
            newdata[i,:] = tempdata[i][0,0]
        stddata = np.std(newdata,0)
        if args['alog']: 

            stddata = np.log10(stddata)
        pp.fill_betweenx(alts[iminalt:],pdata-stddata,pdata+stddata)
if args['min']:
    mini = args['min']
else:
    mini = minv

if args['max']:
    maxi = args['max']
else:
    maxi = maxv

pp.ylim([minalt,250])
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


pp.xlabel(header['vars'][vars[3]]+' Density [m$^{-3}$]')
# pp.xlabel('[e-] [m$^{-3}$]')
pp.ylabel('Altitude (km)')
pp.savefig('plot.png')