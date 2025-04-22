#!/usr/bin/env python

import sys 
import numpy as np 
import re 
from matplotlib import pyplot as pp
from gitm_routines import *
import netCDF4 as nc

nSecondsInDay = 88775.244

def get_args(argv):

    help = False
    filelist = [] 
    var = -1

    for arg in argv:

        IsFound = 0

        m = re.match(r'-var=(.*)',arg)
        if m:
            var = m.group(1)
            IsFound = 1     

        m = re.match(r'-h',arg)
        if m:
            help = True
            IsFound = 1
        
        if (not IsFound):
            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)

        args = {'filelist':filelist,
        'help':help,
        'var':var,
        }

    return args

args = get_args(sys.argv)

try:
    header = read_gitm_header(args["filelist"])
except:
    args['help'] = True   
    header = {"vars":[]}

if (args["help"]):

    print('Calculate zonal averages from a 3D GITM file (3D*bin)')
    print('Usage : ')
    print('gitm_calc_averages.py filelist -var=var1 [-h]')
    print('   Required: filelist- a list of gitm .bin files')
    print('   -var=var1 variable to plot')
    print('   -h: print this message')


    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()

filelist = args['filelist']

vars = [0,1,2]
vars.extend([int(v) for v in args["var"].split(',')])
Var = [header['vars'][int(i)] for i in args['var'].split(',')]

stime = [] #keep track of the start of each day
nfiles = len(filelist)
newDay = True
timediffold = 0
iday = 0

zonaldailyaverage = {}

for file in filelist:
    data = read_gitm_one_file(file,vars)

    if newDay:
        stime.append(data['time'])
        alldata = np.zeros((data['nLons']-4,data['nLats']-4,data['nAlts']-4))
        ntimes = 0
        newDay = False

    timediff = (data['time'] - stime[0]).total_seconds()

    if timediff % nSecondsInDay < timediffold % nSecondsInDay:
        dailyaverage = alldata/ntimes # calculate the "daily" average
        zonaldailyaverage[stime[-1]] = np.mean(dailyaverage,axis=0)

        newDay = True
        iday += 1

    else:
        alldata += data[vars[-1]][2:-2,2:-2,2:-2]
        ntimes += 1
    

    timediffold = timediff


# Create a NetCDF file
# Set Unix epoch
outputfile = 'output.nc'
reference_time = "seconds since 1970-01-01 00:00:00 UTC"
time_keys = sorted(zonaldailyaverage.keys())
time_values = np.array([nc.date2num(t, units=reference_time, calendar="standard") for t in time_keys])


with nc.Dataset("output.nc", "w", format="NETCDF4") as dataset:
    # Define dimensions
    dataset.createDimension("time", len(zonaldailyaverage.keys()))
    dataset.createDimension("alt", len(data[2][0,0,2:-2]))
    dataset.createDimension("lat", len(data[1][0,2:-2,0]))

    # Create time variable
    time_var = dataset.createVariable("time", "f8", ("time",))
    time_var.units = reference_time
    time_var.calendar = "standard"
    time_var[:] = time_values  # Store time data
    
    # Create the altitude variable
    alt_var = dataset.createVariable("alt", "f4", ("alt",))
    alt_var.units = "km"
    alt_var.long_name = "Altitude"
    alt_var[:] = data[2][0,0,2:-2]/1000.  # Store altitudes

    # Create the 1D latitude variable
    lat_var = dataset.createVariable("lat", "f4", ("lat",))
    lat_var.units = "degrees_north"
    lat_var.long_name = "Latitude"
    lat_var[:] = data[1][0,2:-2,0]*180/np.pi  # Store latitudes

    # Create a variable for the 2D array data
    data_var = dataset.createVariable("density", "f8", ("time", "lat", "alt"))
    data_var.units = "1/m^3"  # Example: Watts per square meter (change as needed)
    data_var.long_name = "Number Density"  # Optional descriptive name

    # Store the 2D data for each time step
    for i, t in enumerate(time_keys):
        data_var[i, :, :] = zonaldailyaverage[t]

print("{} file has been created".format(outputfile))
