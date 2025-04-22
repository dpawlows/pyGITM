#!/usr/bin/env python

import numpy as np 
from matplotlib import pyplot as pp 
import sys 
import re 
from gitm_routines import * 
import pandas as pd 

def get_args(argv):

    filelist = []
    var = -1
    help = False
    alog = False
    min = None 
    max = None 
    alt = 400.0
    lon = -100.0
    lat = -100.0
    cut = 'loc'
    smin = -100.0
    smax = -100.0

    help = 0

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

            m = re.match(r'-smin=(.*)',arg)
            if m:
                smin = int(m.group(1))
                IsFound = 1   

            m = re.match(r'-smax=(.*)',arg)
            if m:
                smax = int(m.group(1))
                IsFound = 1

            if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)
            


    args = {'filelist':filelist,
        'var':var,
        'help':help,
        'alog':alog,
        'min':min,
        'max':max,
        'cut':cut,
        'lat':lat,
        'lon':lon,
        'alt':lat,
        'smin':smin,
        'smax':smax,
    }

    return args

def readheader(filename):
    f = open(filename,'r')
    nlats = nlons = nalts = None
    started = False
    while not started:
        line = f.readline()
        lat_match = re.search(r'Number of Latitude points:\s+(\d+)', line)
        lon_match = re.search(r'Number of Longitude points:\s+(\d+)', line)
        alt_match = re.search(r'Number of Altitude points:\s+(\d+)', line)
        if re.search(r'\bLongitude\b', line) and re.search(r'\bLatitude\b', line) and \
            re.search(r'\bAltitude\b', line):
            vars = re.split(r'\s{2,}', line.strip())

        start_match = re.search(r'#START', line)

        if lat_match:
            nlats = int(lat_match.group(1))
        if lon_match:
            nlons = int(lon_match.group(1))
        if alt_match:
            nalts = int(alt_match.group(1))

        if start_match:
            started = True 
            if not (nlats and nlons and nalts):
                print("Error finding grid size in {}".format(filename))
                exit(1)
            else:
                return {'nlons':nlons,
                'nlats':nlats,
                'nalts':nalts,
                'vars':vars
                } 


args = get_args(sys.argv)
header = readheader(args["filelist"][0])

if args['cut'] == 'loc' and args['lon'] > -50:
    plon = args['lon']
    plat = args['lat']
elif args['cut'] == 'sza' and args['smin'] > -50:
    smin = args['smin']
    smax = args['smax']
else:
    args["help"] = '-h'    

if (args["help"]):

    print('Usage : ')
    print('plotMarsGRAM.py -var=N1[,N2,N3,...] -lat=lat -lon=lon -alog')
    print('                     -help [file]')
    print('   -help : print this message')
    print('   -var=number[,num2,num3,...] : number is variable to plot')
    print('   -cut=loc,sza: Plot type ')
    print('   -lat=latitude : latitude in degrees (closest) (cut=loc) ')
    print('   -lon=longitude: longitude in degrees (closest) (cut=loc)')
    print('   -smin=minsza: minimum solar zenith angle (cut=sza)')
    print('   -smax=maxsza: maximum solar zenigh angle (cut=sza)')
    print('   -min=min: minimum value to plot')
    print('   -max=max: maximum value to plot')
    print('   -alog: plot the log of the variable')
    print('   Non-KW arg: files.')

    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


filelist = args["filelist"]




