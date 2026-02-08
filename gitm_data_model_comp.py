#!/usr/bin/env python

### Plot a GITM satellite file and the corresponding data

import sys
import numpy as np
import re
import os
import datetime as dt
from matplotlib import pyplot as pp
from gitm_routines import *
import ngims
import rose
import marstiming as mt
import argparse
import gitmconcurrent as gc


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GITM sat file with corresponding data. By default\
        files are processed in parallel using ProcessPoolExecutor. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--directory',default='.',help='Directory that contains data files')
    parser.add_argument('-i','--instrument',default='ngims',help='Instrument name')
    parser.add_argument('files',nargs='+', help='Input data files (can be globbed by the shell)')
    parser.add_argument('--version',default=None,help='Data version if applicable')
    parser.add_argument('-v','--vars',nargs='+',type=int,required=True,help='Variable(s) to plot')
    parser.add_argument('-l','--list_vars',action='store_true',help='List available variables')
    parser.add_argument('-s','--serial',action='store_true',help='Process files serially')
    parser.add_argument('--verbose',action='store_true',help='Verbose output')

    return parser.parse_args()

def main():
    # Get the header before completely parseing the args.
    args = parse_args()
    header = read_gitm_header(args.files)
    cleaned_vars = [clean_varname(v) for v in header['vars']]
    
    if args.list_vars:
        print(f"GITM variables for {args.files[0]}:")
        for i, var in enumerate(cleaned_vars):
            print(f'{i} {var}')
        exit()
    datadir = args.directory

    #################################
    # Start by getting the file lists and reading the data
    
    #GITM first
    filelist_g = args.files

    vars = [0,1,2] #Lon lat alt
    vars.extend(args.vars) # add user vars

    # get GITM results. Returns list of dictionaries. 1 element per time.    
    data_g = gc.process_batch(filelist_g,vars,serial=args.serial,verbose=args.verbose)

    file_times = [file_time(f) for f in filelist_g]
    file_times.sort()
    start = file_times[0]
    end = file_times[-1]

    if args.instrument == 'ngims':
        version = args.version
        inboundonly = True 

        # NGIMS separates neutrals and ions into different files        
        # Assume neutral
        speciesColumn = 'species'
        qualityFlag = ['IV','IU']
        dentype = 'csn'

        if '+' in cleaned_vars[args.vars[0]]:
            # If there is a +, it's an ion!
            speciesColumn = 'ion_mass'
            qualityFlag = ['SCP','SC0']
            dentype = 'ion'

        filelist_d = ngims.getfiles(start,end,dentype=dentype,version=version,dir=datadir)
        
        # Get the data
        data_d = ngims.readNGIMS(filelist_d)


    ################################
    # Next, loop through data files, get data and data location. Then pull gitm results at those
    # locations and times

    # Clean and filter the data
    for rawdata in data_d:
        data = rawdata[rawdata["alt"] < 350]
        data = data[data["quality"].isin(qualityFlag)]

        for var in args.vars:
            datavar = ngims.varmap[cleaned_vars[var]]
            data_df = data[data[speciesColumn] == datavar]

            if data_df.empty:
                print(f"Skipping {datavar} in orbit {data['t_utc'].iloc[0]}: no data")
                continue
            
            # breakpoint()


if __name__ == '__main__':
    main()