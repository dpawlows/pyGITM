#!/usr/bin/env python

from glob import glob
from datetime import datetime
from datetime import timedelta
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from pylab import cm
from gitm_routines import *
import argparse
import sys

rtod = 180.0/3.141592

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def get_args(argv):

    parser = argparse.ArgumentParser(
        prog='gitm_comp_3d.py',
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Compare two GITM 3D files on a 2D cut. The 1st file is '
                    'the perturbation and the\n2nd file is the baseline. By '
                    'default the percent difference,\n'
                    '100*(perturbation - baseline)/baseline, is plotted.',
        epilog='Options that take a value accept either -opt=value or '
               '-opt value.\nIf a file is given along with -h, the variables '
               'in that file are listed.')
    parser.add_argument('filelist', nargs='*', metavar='file',
                        help='files to compare: perturbation file, then '
                             'baseline file. Must be two.')
    parser.add_argument('-h', '-help', action='store_true', dest='help',
                        help='print this message')
    parser.add_argument('-var', type=int, default=15, metavar='number',
                        help='number of the variable to plot (default: 15)')
    parser.add_argument('-ratio', action='store_true',
                        help='plot the ratio perturbation/baseline instead '
                             'of the percent difference')
    parser.add_argument('-tec', action='store_true',
                        help='compare TEC (overrides -var)')
    parser.add_argument('-cut', choices=['alt', 'lat', 'lon'], default='alt',
                        help='which cut you would like (default: alt)')
    parser.add_argument('-alt', type=float, default=400.0, metavar='altitude',
                        help='can be either alt in km or grid number '
                             '(closest) (default: 400)')
    parser.add_argument('-lat', type=float, default=-100.0, metavar='latitude',
                        help='latitude in degrees (closest)')
    parser.add_argument('-lon', type=float, default=-100.0,
                        metavar='longitude',
                        help='longitude in degrees (closest)')
    parser.add_argument('-alog', action='store_true',
                        help='plot the log of the plotted quantity')
    parser.add_argument('-winds', '-wind', action='store_true', dest='winds',
                        help='overplot wind differences')
    parser.add_argument('-min', type=float, default=None, metavar='minimum',
                        help='minimum value for the plot scale')
    parser.add_argument('-max', type=float, default=None, metavar='maximum',
                        help='maximum value for the plot scale')
    parser.add_argument('-cmap', default='plasma', metavar='name',
                        help='matplotlib colormap to use (default: plasma)')

    args = parser.parse_args(argv[1:])

    if (args.help):
        parser.print_help()
        if (len(args.filelist) > 0):
            header = read_gitm_header(args.filelist)
            print('')
            print('variables:')
            iVar = 0
            for var in header["vars"]:
                print(iVar,var)
                iVar=iVar+1
        exit()

    if (len(args.filelist) != 2):
        parser.error('Can only compare 2 files.')

    if (args.tec):
        args.var = 34

    return args

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Main Code!
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

args = get_args(sys.argv)

header = read_gitm_header(args.filelist)

filelist = args.filelist
cut = args.cut
vars = [0,1,2]
vars.append(args.var)

if (args.winds):
    if (cut=='alt'):
        iUx_ = 16
        iUy_ = 17
    if (cut=='lat'):
        iUx_ = 16
        iUy_ = 18
    if (cut=='lon'):
        iUx_ = 17
        iUy_ = 18
    vars.append(iUx_)
    vars.append(iUy_)
    AllWindsX = []
    AllWindsY = []

Var = header["vars"][args.var]
AllData2D = []
AllAlts = []
AllTimes = []
j = 0

dataperturb =  read_gitm_one_file(glob(filelist[0])[0], vars)
data = read_gitm_one_file(glob(filelist[1])[0], vars)
[nLons, nLats, nAlts] = data[0].shape

Alts = data[2][0][0]/1000.0;
Lons = data[0][:,0,0]*rtod;
Lats = data[1][0,:,0]*rtod;
if (cut == 'alt'):
    xPos = Lons
    yPos = Lats
    if (len(Alts) > 1):
        if (args.alt < 50):
            iAlt = int(args.alt)
        else:
            if (args.alt > Alts[nAlts-3]):
                iAlt = nAlts-3
            else:
                iAlt = 2
                while (Alts[iAlt] < args.alt):
                    iAlt=iAlt+1
    else:
        iAlt = 0
    Alt = Alts[iAlt]

if (cut == 'lat'):
    xPos = Lons
    yPos = Alts
    if (args.lat < Lats[1]):
        iLat = int(nLats/2)
    else:
        if (args.lat > Lats[nLats-2]):
            iLat = int(nLats/2)
        else:
            iLat = 2
            while (Lats[iLat] < args.lat):
                iLat=iLat+1
    Lat = Lats[iLat]

if (cut == 'lon'):
    xPos = Lats
    yPos = Alts
    if (args.lon < Lons[1]):
        iLon = int(nLons/2)
    else:
        if (args.lon > Lons[nLons-2]):
            iLon = int(nLons/2)
        else:
            iLon = 2
            while (Lons[iLon] < args.lon):
                iLon=iLon+1
    Lon = Lons[iLon]

AllTimes.append(data["time"])

if (args.tec):
    iAlt = 2
    tec = np.zeros((nLons, nLats))
    tecperturb = np.zeros((nLons, nLats))
    for Alt in Alts:
        if (iAlt > 0 and iAlt < nAlts-3):
            tec = tec + data[args.var][:,:,iAlt] * (Alts[iAlt+1]-Alts[iAlt-1])/2 * 1000.0
            tecperturb = tecperturb + dataperturb[args.var][:,:,iAlt] * (Alts[iAlt+1]-Alts[iAlt-1])/2 * 1000.0
        iAlt=iAlt+1
    base2D = tec/1e16
    perturb2D = tecperturb/1e16
else:
    if (cut == 'alt'):
        base2D = data[args.var][:,:,iAlt]
        perturb2D = dataperturb[args.var][:,:,iAlt]
    if (cut == 'lat'):
        base2D = data[args.var][:,iLat,:]
        perturb2D = dataperturb[args.var][:,iLat,:]
    if (cut == 'lon'):
        base2D = data[args.var][iLon,:,:]
        perturb2D = dataperturb[args.var][iLon,:,:]
    if (args.winds):
        if (cut == 'alt'):
            AllWindsX.append(dataperturb[iUx_][:,:,iAlt]-data[iUx_][:,:,iAlt])
            AllWindsY.append(dataperturb[iUy_][:,:,iAlt]-data[iUy_][:,:,iAlt])
        if (cut == 'lat'):
            AllWindsX.append(dataperturb[iUx_][:,iLat,:]-data[iUx_][:,iLat,:])
            AllWindsY.append(dataperturb[iUy_][:,iLat,:]-data[iUy_][:,iLat,:])
        if (cut == 'lon'):
            AllWindsX.append(dataperturb[iUx_][iLon,:,:]-data[iUx_][iLon,:,:])
            AllWindsY.append(dataperturb[iUy_][iLon,:,:]-data[iUy_][iLon,:,:])


if (args.ratio):
    AllData2D.append(perturb2D/base2D)
else:
    AllData2D.append((perturb2D-base2D)/base2D*100.0)

AllData2D = np.array(AllData2D)
if (args.winds):
    AllWindsX = np.array(AllWindsX)
    AllWindsY = np.array(AllWindsY)

Negative = 0

AllData2D = np.log10(AllData2D) if (args.alog) else AllData2D

maxi  = np.max(AllData2D[0,2:-2,2:-2])*1.05
mini  = np.min(AllData2D[0,2:-2,2:-2])*0.95
if args.min is not None:
    mini = args.min
if args.max is not None:
    maxi = args.max
#mini = 0
#maxi=4
if not args.ratio and maxi == 0 and mini == 0:
    print("Error: There doesn't seem to be a difference between the data sets")
    print("are you sure the files are actually different?")
    exit()

# if (mini < 0):
#     Negative = 1

if (Negative):
    maxi = np.max(abs(AllData2D))*1.05
    mini = -maxi

print("Data shape: {}".format(AllData2D.shape))
if (cut == 'alt'):
    maskNorth = ((yPos>45) & (yPos<90.0))
    maskSouth = ((yPos<-45) & (yPos>-90.0))
    DoPlotNorth = np.max(maskNorth)
    DoPlotSouth = np.max(maskSouth)
    DoPlotNorth = False
    DoPlotSouth = False

    if (DoPlotNorth):
        maxiN = np.max(abs(AllData2D[:,2:-2,maskNorth]))*1.05
        if (Negative):
            miniN = -maxiN
        else:
            miniN = np.min(AllData2D[:,2:-2,maskNorth])*0.95
    if (DoPlotSouth):
        maxiS = np.max(abs(AllData2D[:,2:-2,maskSouth]))*1.05
        if (Negative):
            miniS = -maxiS
        else:
            miniS = np.min(AllData2D[:,2:-2,maskSouth])*0.95
dr = (maxi-mini)/31
levels = np.arange(mini, maxi, dr)

i = 0

# Define plot range:
minX = (xPos[ 1] + xPos[ 2])/2
maxX = (xPos[-2] + xPos[-3])/2
minY = (yPos[ 1] + yPos[ 2])/2
maxY = (yPos[-2] + yPos[-3])/2

file = "diff%2.2d_" % args.var
file = file+cut
if (args.ratio):
    file = file+'_ratio'

for time in AllTimes:

    ut = time.hour + time.minute/60.0 + time.second/3600.0
    shift = ut * 15.0
    print(ut)

#    fig = plt.figure(constrained_layout=False,
#                     tight_layout=True, figsize=(10, 5.5))
    fig = plt.figure(tight_layout=True, figsize=(10, 5.5))

    # gs1 = GridSpec(nrows=1, ncols=1, wspace=0.0, hspace=0)
    # gs = GridSpec(nrows=1, ncols=1, wspace=0.0, left=0.0, right=0.9)

    norm = cm.colors.Normalize(vmax=mini, vmin=maxi)
    print(mini)
    # if (mini >= 0):
    try:
        cmap = cm.get_cmap(args.cmap)
    except ValueError:
        print("Error: invalid colormap '{}'. Use a valid matplotlib colormap name.".format(args.cmap))
        exit()
    # else:
    #     cmap = cm.bwr

    d2d = np.transpose(AllData2D[i])
    if (args.winds):
        Ux2d = np.transpose(AllWindsX[i])
        Uy2d = np.transpose(AllWindsY[i])

    sTime = time.strftime('%y%m%d_%H%M%S')
    outfile = file+'_'+sTime+'.png'

    ax = fig.add_subplot()
    cax = ax.pcolor(xPos, yPos, d2d, vmin=mini, vmax=maxi, shading='auto', cmap=cmap)


    if (args.winds):
        ax.quiver(xPos,yPos,Ux2d,Uy2d)
    ax.set_ylim([minY,maxY])
    ax.set_xlim([minX,maxX])

    if (cut == 'alt'):
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlabel('Longitude (deg)')
        title = time.strftime('%b %d, %Y %H:%M:%S')+'; Alt : '+"%.2f" % Alt + ' km'
        ax.set_aspect(1.0)

    if (cut == 'lat'):
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Altitude (km)')
        title = time.strftime('%b %d, %Y %H:%M:%S')+'; Lat : '+"%.2f" % Lat + ' km'

    if (cut == 'lon'):
        ax.set_xlabel('Latitude (deg)')
        ax.set_ylabel('Altitude (km)')
        title = time.strftime('%b %d, %Y %H:%M:%S')+'; Lon : '+"%.2f" % Lon + ' km'

    ax.set_title(title)
    cbar = fig.colorbar(cax, ax=ax, shrink = 0.75, pad=0.02)
    if (args.ratio):
        cbar.set_label(Var+' Ratio',rotation=90)
    else:
        cbar.set_label(Var+' % Difference',rotation=90)

    if (cut == 'alt'):

        if (DoPlotNorth):
            # Top Left Graph Northern Hemisphere
            ax2 = fig.add_subplot(gs[0, 0],projection='polar')
            r, theta = np.meshgrid(90.0-yPos[maskNorth], (xPos+shift-90.0)*3.14159/180.0)
            cax2 = ax2.pcolor(theta, r, AllData2D[i][:,maskNorth], vmin=miniN, vmax=maxiN, shading='auto', cmap=cmap)
            xlabels = ['', '12', '18', '00']
            ylabels = ['80', '70', '60', '50']

            ax2.set_xticklabels(xlabels)
            ax2.set_yticklabels(ylabels)
            cbar2 = fig.colorbar(cax2, ax=ax2, shrink = 0.5, pad=0.01)
            ax2.grid(linestyle=':', color='black')
            pi = 3.14159
            ax2.set_xticks(np.arange(0,2*pi,pi/2))
            ax2.set_yticks(np.arange(10,50,10))

        if (DoPlotSouth):
            # Top Right Graph Southern Hemisphere
            r, theta = np.meshgrid(90.0+yPos[maskSouth], (xPos+shift-90.0)*3.14159/180.0)
            ax3 = fig.add_subplot(gs[0, 1],projection='polar')
            cax3 = ax3.pcolor(theta, r, AllData2D[i][:,maskSouth], vmin=miniS, vmax=maxiS, shading='auto', cmap=cmap)
            xlabels = ['', '12', '18', '00']
            ylabels = ['80', '70', '60', '50']
            ax3.set_xticklabels(xlabels)
            ax3.set_yticklabels(ylabels)
            cbar3 = fig.colorbar(cax3, ax=ax3, shrink = 0.5, pad=0.01)
            ax3.grid(linestyle=':', color='black')
            pi = 3.14159
            ax3.set_xticks(np.arange(0,2*pi,pi/2))
            ax3.set_yticks(np.arange(10,50,10))


    print("Writing file : "+outfile)
    fig.savefig(outfile)
    plt.close()

    i=i+1
