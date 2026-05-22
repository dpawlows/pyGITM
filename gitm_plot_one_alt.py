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
import re
import sys

rtod = 180.0/3.141592
boltzmann = 1.380649e-23

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def get_args(argv):

    filelist = []
    IsLog = 0
    IsContour = 0
    var = 15
    alt = 400.0
    lon = -100.0
    lat = -100.0
    tec = 0
    cut = 'alt'
    help = 0
    winds = 0
    norm = 0
    diff = 0
    minv = None
    maxv = None
    oco2 = 0
    pressure = 0
    hmf2 = 0
    cmap = 'plasma'

    for arg in argv:

        IsFound = 0

        if (not IsFound):

            m = re.match(r'-var=(.*)',arg)
            if m:
                var = int(m.group(1))
                IsFound = 1

            m = re.match(r'-diff',arg)
            if m:
                diff = 1
                IsFound = 1

            m = re.match(r'-tec',arg)
            if m:
                var = 34
                tec = 1
                IsFound = 1

            m = re.match(r'-alt=(.*)',arg)
            if m:
                alt = int(m.group(1))
                IsFound = 1

            m = re.match(r'-lat=(.*)',arg)
            if m:
                lat = int(m.group(1))
                IsFound = 1

            m = re.match(r'-lon=(.*)',arg)
            if m:
                lon = int(m.group(1))
                IsFound = 1

            m = re.match(r'-cut=(.*)',arg)
            if m:
                cut = m.group(1)
                IsFound = 1

            m = re.match(r'-alog',arg)
            if m:
                IsLog = 1
                IsFound = 1

            m = re.match(r'-contour',arg)
            if m:
                IsContour = 1
                IsFound = 1

            m = re.match(r'-h$',arg)
            if m:
                help = 1
                IsFound = 1

            m = re.match(r'-min=(.*)',arg)
            if m:
                minv = float(m.group(1))
                IsFound = 1   

            m = re.match(r'-max=(.*)',arg)
            if m:
                maxv = float(m.group(1))
                IsFound = 1  

            m = re.match(r'-wind',arg)
            if m:
                winds = 1
                IsFound = 1

            m = re.match(r'-norm',arg)
            if m:
                norm = 1
                IsFound = 1

            m = re.match(r'-O/CO2',arg)
            if m:
                oco2 = 1
                IsFound = 1

            m = re.match(r'-pressure',arg)
            if m:
                pressure = 1
                IsFound = 1

            m = re.match(r'-hmf2',arg)
            if m:
                hmf2 = 1
                IsFound = 1

            m = re.match(r'-cmap=(.*)',arg)
            if m:
                cmap = m.group(1)
                IsFound = 1

            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)

    args = {'filelist':filelist,
            'var':var,
            'cut':cut,
            'diff':diff,
            'tec':tec,
            'help':help,
            'winds':winds,
            'norm':norm,
            'alt':alt,
            'lat':lat,
            'lon':lon,
            'IsLog':IsLog,
            'IsContour':IsContour,
            'minv':minv,
            'maxv':maxv,
            'oco2':oco2,
            'pressure':pressure,
            'hmf2':hmf2,
            'cmap':cmap}

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

header = read_gitm_header(args["filelist"])

if (args["help"]):

    print('Usage : ')
    print('gitm_plot_one_alt.py -var=N -tec -winds -cut=alt,lat,lon')
    print('                     -alt=alt -lat=lat -lon=lon -alog ')
    print('                     -O/CO2 -norm -help [*.bin or a file]')
    print('   -help : print this message')
    print('   -var=number : number is variable to plot')
    print('   -cut=alt,lat,lon : which cut you would like')
    print('   -alt=altitude : can be either alt in km or grid number (closest)')
    print('   -lat=latitude : latitude in degrees (closest)')
    print('   -lon=longitude: longitude in degrees (closest)')
    print('   -alog : plot the log of the variable')
    print('   -contour: plot a contour instead of a pseudocolor')
    print('   -winds: overplot winds')
    print('   -norm : normalize wind vectors to show direction only (use with -winds)')
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -cmap=name: colormap to use (default: plasma)')
    print('   -O/CO2 : plot O/CO2 number density ratio (requires 3DALL files)')
    print('   -pressure: calculate and plot pressure from neutral densities and temperature')
    print('   -hmf2 : plot altitude of peak electron density [e-] as a function of lat/lon')
    print('   At end, list the files you want to plot')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


filelist = args["filelist"]
nFiles = len(filelist)

cut = args["cut"]

if args["hmf2"]:
    cut = 'alt'
    iE_ = None
    for iV, vname in enumerate(header["vars"]):
        if vname.strip() == '[e-]':
            iE_ = iV
            break
    if iE_ is None:
        print("Error: could not find [e-] in file variables.")
        print("Available variables:")
        for iV, v in enumerate(header["vars"]):
            print("  {:3d}  {}".format(iV, v))
        exit()
    print("Found [e-] at var index {}".format(iE_))

pressure_density_indices = []
pressure_temp_index = None
if args["pressure"]:
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

if args["oco2"]:
    for f in filelist:
        if not re.search(r'3DALL', f):
            print("Error: -O/CO2 requires 3DALL files, but '{}' does not appear to be a 3DALL file.".format(f))
            exit()
    # Find variable indices for [O] and [CO2] from the header
    iO_ = None
    iCO2_ = None
    for iV, vname in enumerate(header["vars"]):
        vstrip = vname.strip()
        if vstrip == '[O]':
            iO_ = iV
        if vstrip == '[CO!D2!N]':
            iCO2_ = iV
    if iO_ is None or iCO2_ is None:
        print("Error: could not find [O] (found={}) or [CO2] (found={}) in file variables.".format(iO_, iCO2_))
        print("Available variables:")
        for iV, v in enumerate(header["vars"]):
            print("  {:3d}  {}".format(iV, v))
        exit()
    print("Found [O] at var index {} and [CO2] at var index {}".format(iO_, iCO2_))
    vars = [0, 1, 2, iO_, iCO2_]
elif args["pressure"]:
    vars = [0, 1, 2] + pressure_density_indices + [pressure_temp_index]
elif args["hmf2"]:
    vars = [0, 1, 2, iE_]
else:
    vars = [0,1,2]
    vars.append(args["var"])

if (args["winds"]):
    iEast_ = None
    iNorth_ = None
    iUp_ = None
    for iV, vname in enumerate(header["vars"]):
        vstrip = vname.strip()
        if 'V!Dn!N(east)' in vstrip:
            iEast_ = iV
        if 'V!Dn!N(north)' in vstrip:
            iNorth_ = iV
        if 'V!Dn!N(up)' in vstrip:
            iUp_ = iV
    if iEast_ is None or iNorth_ is None or iUp_ is None:
        print("Error: could not find wind variables (east={}, north={}, up={}) in header.".format(iEast_, iNorth_, iUp_))
        exit()
    if (cut=='alt'):
        iUx_ = iEast_
        iUy_ = iNorth_
    if (cut=='lat'):
        iUx_ = iEast_
        iUy_ = iUp_
    if (cut=='lon'):
        iUx_ = iNorth_
        iUy_ = iUp_
    vars.append(iUx_)
    vars.append(iUy_)
    AllWindsX = []
    AllWindsY = []

if args["oco2"]:
    Var = r'$[O]/[CO_2]$'
elif args["pressure"]:
    Var = 'Pressure (Pa)'
elif args["hmf2"]:
    Var = 'HmF2 (km)'
else:
    Var = header["vars"][args["var"]]
    Var = Var.replace('!U','^')
    Var = Var.replace('!D','_')
    Var = Var.replace('!N','')
    Var = '$'+Var+'$'
AllData2D = []
AllAlts = []
AllTimes = []
j = 0
for file in filelist:
    data = read_gitm_one_file(file, vars)
    if args["oco2"]:
        _dbg = data[iO_]
        print("Data shape:", _dbg.shape)
        print("Min/Max [O]:", np.nanmin(_dbg), np.nanmax(_dbg))
        print("NaN count:", np.sum(np.isnan(_dbg)))
        print("Inf count:", np.sum(np.isinf(_dbg)))
    elif args["pressure"]:
        _dbg = data[pressure_temp_index]
        print("Data shape:", _dbg.shape)
        print("Min/Max Temperature:", np.nanmin(_dbg), np.nanmax(_dbg))
        print("NaN count:", np.sum(np.isnan(_dbg)))
        print("Inf count:", np.sum(np.isinf(_dbg)))
    elif args["hmf2"]:
        _dbg = data[iE_]
        print("Data shape:", _dbg.shape)
        print("Min/Max [e-]:", np.nanmin(_dbg), np.nanmax(_dbg))
        print("NaN count:", np.sum(np.isnan(_dbg)))
        print("Inf count:", np.sum(np.isinf(_dbg)))
    else:
        print("Data shape:", data[args["var"]].shape)
        print("Min/Max:", np.nanmin(data[args["var"]]), np.nanmax(data[args["var"]]))
        print("NaN count:", np.sum(np.isnan(data[args["var"]])))
        print("Inf count:", np.sum(np.isinf(data[args["var"]])))
    if (j == 0):
        [nLons, nLats, nAlts] = data[0].shape
        Alts = data[2][0][0]/1000.0;
        Lons = data[0][:,0,0]*rtod;
        Lats = data[1][0,:,0]*rtod;
        if (cut == 'alt'):
            xPos = Lons
            yPos = Lats
            if (len(Alts) > 1):
                if (args["alt"] > Alts[nAlts-3]):
                    iAlt = nAlts-3
                else:
                    iAlt = 2
                    while (Alts[iAlt] < args["alt"]):
                        iAlt=iAlt+1
            else:
                iAlt = 0
            Alt = Alts[iAlt]

        if (cut == 'lat'):
            xPos = Lons
            yPos = Alts
            if (args["lat"] < Lats[1]):
                iLat = int(nLats/2)
            else:
                if (args["lat"] > Lats[nLats-2]):
                    iLat = int(nLats/2)
                else:
                    iLat = 2
                    while (Lats[iLat] < args["lat"]):
                        iLat=iLat+1
            Lat = Lats[iLat]

        if (cut == 'lon'):
            xPos = Lats
            yPos = Alts
            if (args["lon"] < Lons[1]):
                iLon = int(nLons/2)
            else:
                if (args["lon"] > Lons[nLons-2]):
                    iLon = int(nLons/2)
                else:
                    iLon = 2
                    while (Lons[iLon] < args["lon"]):
                        iLon=iLon+1
            Lon = Lons[iLon]

    AllTimes.append(data["time"])

    if args["oco2"]:
        ratio = data[iO_] / np.where(data[iCO2_] == 0, np.nan, data[iCO2_])
        if (cut == 'alt'):
            AllData2D.append(ratio[:,:,iAlt])
        if (cut == 'lat'):
            AllData2D.append(ratio[:,iLat,:])
        if (cut == 'lon'):
            AllData2D.append(ratio[iLon,:,:])
    elif args["hmf2"]:
        iAlt_peak = np.argmax(data[iE_], axis=2)
        AllData2D.append(Alts[iAlt_peak])
    elif (args["tec"]):
        iAlt = 2
        tec = np.zeros((nLons, nLats))
        for Alt in Alts:
            if (iAlt > 0 and iAlt < nAlts-3):
                tec = tec + data[args["var"]][:,:,iAlt] * (Alts[iAlt+1]-Alts[iAlt-1])/2 * 1000.0
            iAlt=iAlt+1
        AllData2D.append(tec/1e16)
    elif (args["pressure"]):
        if (cut == 'alt'):
            number_density = np.zeros_like(data[pressure_density_indices[0]][:,:,iAlt])
            for idens in pressure_density_indices:
                number_density += data[idens][:,:,iAlt]
            AllData2D.append(number_density * boltzmann * data[pressure_temp_index][:,:,iAlt])
        if (cut == 'lat'):
            number_density = np.zeros_like(data[pressure_density_indices[0]][:,iLat,:])
            for idens in pressure_density_indices:
                number_density += data[idens][:,iLat,:]
            AllData2D.append(number_density * boltzmann * data[pressure_temp_index][:,iLat,:])
        if (cut == 'lon'):
            number_density = np.zeros_like(data[pressure_density_indices[0]][iLon,:,:])
            for idens in pressure_density_indices:
                number_density += data[idens][iLon,:,:]
            AllData2D.append(number_density * boltzmann * data[pressure_temp_index][iLon,:,:])
        if (args["winds"]):
            if (cut == 'alt'):
                AllWindsX.append(data[iUx_][:,:,iAlt])
                AllWindsY.append(data[iUy_][:,:,iAlt])
            if (cut == 'lat'):
                AllWindsX.append(data[iUx_][:,iLat,:])
                AllWindsY.append(data[iUy_][:,iLat,:])
            if (cut == 'lon'):
                AllWindsX.append(data[iUx_][iLon,:,:])
                AllWindsY.append(data[iUy_][iLon,:,:])
    else:
        if (cut == 'alt'):
            AllData2D.append(data[args["var"]][:,:,iAlt])
        if (cut == 'lat'):
            AllData2D.append(data[args["var"]][:,iLat,:])
        if (cut == 'lon'):
            AllData2D.append(data[args["var"]][iLon,:,:])
        if (args["winds"]):
            if (cut == 'alt'):
                AllWindsX.append(data[iUx_][:,:,iAlt])
                AllWindsY.append(data[iUy_][:,:,iAlt])
            if (cut == 'lat'):
                AllWindsX.append(data[iUx_][:,iLat,:])
                AllWindsY.append(data[iUy_][:,iLat,:])
            if (cut == 'lon'):
                AllWindsX.append(data[iUx_][iLon,:,:])
                AllWindsY.append(data[iUy_][iLon,:,:])
    j=j+1


AllData2D = np.array(AllData2D)
if (args["winds"]):
    AllWindsX = np.array(AllWindsX)
    AllWindsY = np.array(AllWindsY)

Negative = 0

AllData2D = np.log10(AllData2D) if (args['IsLog']) else AllData2D
logvar = "Log " if args['IsLog'] else ""

if args['minv'] == None:
    mini  = np.min(AllData2D[0,2:-2,2:-2])*0.95
else:
    mini = args['minv']

if args['maxv'] == None:
    maxi  = np.max(AllData2D[0,2:-2,2:-2])*1.05
else:
    maxi = args['maxv']

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

if args["oco2"]:
    file = "O_CO2_ratio_" + cut
elif args["pressure"]:
    file = "pressure_" + cut
elif args["hmf2"]:
    file = "hmf2_alt"
else:
    file = "var%2.2d_" % args["var"]
    file = file+cut

for time in AllTimes:

    ut = time.hour + time.minute/60.0 + time.second/3600.0
    shift = ut * 15.0
    print(ut)

    fig = plt.figure(constrained_layout=False,
                     tight_layout=True, figsize=(10, 8.5))

    gs1 = GridSpec(nrows=2, ncols=2, wspace=0.0, hspace=0)
    gs = GridSpec(nrows=2, ncols=2, wspace=0.0, left=0.0, right=0.9)

    norm = cm.colors.Normalize(vmax=mini, vmin=maxi)
    print(mini)
    cmap = args['cmap']

    d2d = np.transpose(AllData2D[i])
    if (args["winds"]):
        Ux2d = np.transpose(AllWindsX[i])
        Uy2d = np.transpose(AllWindsY[i])
        if (args["norm"]):
            mag = np.sqrt(Ux2d**2 + Uy2d**2)
            mag = np.where(mag == 0, np.nan, mag)
            Ux2d = Ux2d / mag
            Uy2d = Uy2d / mag

    sTime = time.strftime('%y%m%d_%H%M%S')
    outfile = file+'_'+sTime+'.png'

    ax = fig.add_subplot(gs1[1, :2])
    if args['IsContour']:
        cax = ax.contourf(xPos, yPos, d2d, vmin=mini, vmax=maxi, levels=30, cmap=cmap)
    else:
        cax = ax.pcolor(xPos, yPos, d2d, vmin=mini, vmax=maxi, shading='auto', cmap=cmap)

    if (args["winds"]):
        ax.quiver(xPos,yPos,Ux2d,Uy2d)
    ax.set_ylim([minY,maxY])
    ax.set_xlim([minX,maxX])

    if (cut == 'alt'):
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlabel('Longitude (deg)')
        if args["hmf2"]:
            title = time.strftime('%b %d, %Y %H:%M:%S') + '; HmF2'
        else:
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
    cbar.set_label(logvar + Var,rotation=90)

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


