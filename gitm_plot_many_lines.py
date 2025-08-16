#!/usr/bin/env python

### Plot a GITM satellite file

import sys
import numpy as np
import re
import os
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

var_indices = [int(v) for v in args["var"].split(',')]
vars = [0, 1, 2]
vars.extend(var_indices)
pvar = var_indices[0]

varmap = {29:44,28:32,27:16,4:'CO2',6:'O',
7:'N2',9:'Ar',5:'CO',
}

varcmap = {27:'O$^+$',28:'O$_2^+$',29:'CO$_2^+$',
4:'CO$_2$',6:'O',7:'N$_2$',9:'Ar',5:'CO'}

alldata = []
linestyle = '-'
for i, file in enumerate(filelist):
    data = read_gitm_one_file(file, vars)
    if i == 0:
        alts = data[2][0, 0] / 1000.0
        iminalt = find_nearest_index(alts, minalt)
    alldata.append(data)

df = pd.DataFrame(alldata)

def plot_single_figure(minv, maxv, xlabel):
    mini = args['min'] if args['min'] else minv
    maxi = args['max'] if args['max'] else maxv
    pp.ylim([minalt, 250])
    pp.xlim([mini, maxi])
    pp.ylabel('Altitude (km)')
    pp.xlabel(xlabel)
    pp.legend()

if not args['average']:
    if len(filelist) > 1 and len(var_indices) > 1:
        for fname, data in zip(filelist, alldata):
            pp.figure()
            colors = iter(pp.cm.rainbow(np.linspace(0, 1, len(var_indices))))
            minv = 9e20
            maxv = -9e20
            for v in var_indices:
                pdata = data[v][0, 0, iminalt:]
                if args['alog']:
                    pdata = np.log10(pdata)
                minv = min(minv, np.min(pdata))
                maxv = max(maxv, np.max(pdata))
                pp.plot(pdata, alts[iminalt:], color=next(colors), ls=linestyle, label=header['vars'][v])
            plot_single_figure(minv, maxv, 'Value')
            outname = os.path.splitext(os.path.basename(fname))[0] + '.png'
            pp.savefig(outname)
    else:
        pp.figure()
        if len(filelist) > 1:
            colors = iter(pp.cm.rainbow(np.linspace(0, 1, len(alldata))))
            minv = 9e20
            maxv = -9e20
            v = var_indices[0]
            for fname, data in zip(filelist, alldata):
                pdata = data[v][0, 0, iminalt:]
                if args['alog']:
                    pdata = np.log10(pdata)
                minv = min(minv, np.min(pdata))
                maxv = max(maxv, np.max(pdata))
                pp.plot(pdata, alts[iminalt:], color=next(colors), ls=linestyle, label=fname)
            xlabel = header['vars'][v] + ' Density [m$^{-3}$]'
            plot_single_figure(minv, maxv, xlabel)
        else:
            colors = iter(pp.cm.rainbow(np.linspace(0, 1, len(var_indices))))
            minv = 9e20
            maxv = -9e20
            data = alldata[0]
            for v in var_indices:
                pdata = data[v][0, 0, iminalt:]
                if args['alog']:
                    pdata = np.log10(pdata)
                minv = min(minv, np.min(pdata))
                maxv = max(maxv, np.max(pdata))
                pp.plot(pdata, alts[iminalt:], color=next(colors), ls=linestyle, label=header['vars'][v])
            xlabel = header['vars'][var_indices[0]] + ' Density [m$^{-3}$]' if len(var_indices) == 1 else 'Value'
            plot_single_figure(minv, maxv, xlabel)
        pp.savefig('plot.png')
else:
    pp.figure()
    meandata = df.mean()
    pdata = meandata[pvar][0, 0]
    if args['alog']:
        pdata = np.log10(pdata)
    minv = np.min(pdata)
    maxv = np.max(pdata)
    pp.plot(pdata, alts[iminalt:], 'k', linewidth=2, label='MGITM')
    if args['stddev']:
        tempdata = df[pvar].to_numpy()
        newdata = np.zeros((len(df), np.shape(tempdata[0])[2]))
        for i in range(len(newdata)):
            newdata[i, :] = tempdata[i][0, 0]
        stddata = np.std(newdata, 0)
        if args['alog']:
            stddata = np.log10(stddata)
        pp.fill_betweenx(alts[iminalt:], pdata - stddata, pdata + stddata)
    xlabel = header['vars'][pvar] + ' Density [m$^{-3}$]'
    plot_single_figure(minv, maxv, xlabel)
    pp.savefig('plot.png')
