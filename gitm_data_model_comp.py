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
    parser.add_argument("-o", "--orbit",type=int, help="Orbit number to highlight in plots")

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
    # Setup the data and the plot
    matched = {} #holds the data and model results at each point in a GITM sat file
        
    fig = pp.figure()

    # Start by getting the file lists and reading the data
    
    #GITM first
    filelist_g = args.files

    vars = [0,1,2] #Lon lat alt
    vars.extend(args.vars) # add user vars

    # get GITM results. Returns list of dictionaries. 1 element per time.    
    data_g = gc.process_batch(filelist_g,vars,serial=args.serial,verbose=args.verbose)

    file_times = [file_time(f) for f in filelist_g]
    file_times.sort()
    gtimes = np.array(file_times)
    start = gtimes[0]
    end = gtimes[-1]

    if args.instrument.lower() == 'ngims':
        version = args.version
        inboundonly = True 

        # NGIMS separates neutrals and ions into different files        
        # Assume neutral
        speciesColumn = 'species'
        qualityFlag = ['IV','IU']
        dentype = 'csn'
        varColumn = 'abundance'
        datamultiplier = 1e6 #cm-3 to m-3

        if '+' in cleaned_vars[args.vars[0]]:
            # If there is a +, it's an ion!
            speciesColumn = 'ion_mass'
            qualityFlag = ['SCP','SC0']
            dentype = 'ion'
            varColumn = 'density'
        filelist_d = ngims.getfiles(start,end,dentype=dentype,version=version,dir=datadir)
        
        # Get the data
        data_d = ngims.readNGIMS(filelist_d)


    ################################
    # Next, loop through data files, get data and data location. Then pull gitm results at those
    # locations and times

    for var in args.vars:
        datavar = ngims.varmap[cleaned_vars[var]]

        matched[datavar] = {
            "time": [],
            "alt_data": [],
            "alt_model": [],
            "lat_model": [],
            "lat_data": [],
            "lon_model": [],
            "lon_data": [],
            "data": [],
            "model": [],
            "sza": [],
            "orbit": [],
        }

    # Clean and filter the data
    for rawdata in data_d:
        data = rawdata[rawdata["alt"] < 300]
        data = data[data["quality"].isin(qualityFlag)]

        for var in args.vars:
            datavar = ngims.varmap[cleaned_vars[var]]
            m = matched[datavar]
       
            data_df = data[data[speciesColumn] == datavar]

            if data_df.empty:
                print(f"Skipping {datavar} in orbit {data['t_utc'].iloc[0]}: no data")

                continue

            dtimes = np.asarray([dt.datetime.fromisoformat(t) for t in data_df['t_utc']])
            d_start = dtimes[0]
            d_end = dtimes[-1]
 
            indices = np.where((gtimes >= d_start) & (gtimes <= d_end))[0]
            g_subset = [data_g[i] for i in indices]
            # There are probably fewer GITM files than data points, so loop over those
            # This is because we may not have good data for all vars.
            # At this point, we know that we have matching GITM results and data
            for i,gdata in enumerate(g_subset):
                target = gdata['time']

                # working with numpy time is better when subtracting a bunch of times
                dtimes64 = dtimes.astype("datetime64[ns]")
                target64 = np.datetime64(target)
                deltas = np.abs(dtimes64 - target64)
                idx_d = deltas.argmin() # Row with the time corresponding to our GITM file

                # Get the data
                alt_data = data_df['alt'].iloc[idx_d]
                var_data = data_df[varColumn].iloc[idx_d]*datamultiplier
                lon_data = data_df['long'].iloc[idx_d]
                lon_data = lon_data + 360 if lon_data < 0 else lon_data

                # Get GITM result at corresponding altitude
                deltas = np.abs(gdata['alt'] - alt_data)
                idx = deltas.argmin()
                alt_gitm = gdata['alt'][idx]
                var_gitm = gdata[var][0,0,idx]

                m["time"].append(target)
                m["alt_data"].append(alt_data)
                m["alt_model"].append(alt_gitm)
                m["lat_data"].append(data_df['lat'].iloc[idx_d])
                m["lat_model"].append(gdata['lat'][0])
                m["lon_data"].append(lon_data)
                m["lon_model"].append(gdata['lon'][0])
                m["data"].append(var_data)
                m["model"].append(var_gitm)
                m["sza"].append(data_df['sza'].iloc[idx_d])
                m["orbit"].append(data_df["orbit"].iloc[idx_d])

    for var in args.vars:
        datavar = ngims.varmap[cleaned_vars[var]]
        m = matched.get(datavar)

        if m is None or len(m["data"]) == 0:
            continue

        # convert to arrays (do this once per variable)
        for k in m:
            m[k] = np.asarray(m[k])

        label = cleaned_vars[var]

        highlight_orbit = args.orbit
        orbits = np.unique(m["orbit"])

        # loop over orbits
        for orbit in orbits:
            #get data for this orbit
            mask = m["orbit"] == orbit

            alt_d = m["alt_data"][mask]
            alt_m = m["alt_model"][mask]
            data = m["data"][mask]
            model = m["model"][mask]

            #highlight an orbit if specified
            is_highlight = (highlight_orbit is not None and orbit == highlight_orbit)
            if is_highlight:
                lw = 2.5
                alpha = 1.0
                zorder = 10
                label_data = f"NGIMS orbit {orbit}"
                label_model = f"M-GITM orbit {orbit}"
            else:
                lw = 1.5
                alpha = 0.7
                zorder = 1
                label_data = None
                label_model = None

            # data: solid line
            pp.plot(data, alt_d, lw=lw, alpha=alpha, label=label_data,
                color='cornflowerblue',zorder=zorder
                )

            # model: dashed line, same color
            pp.plot(model,alt_m,lw=lw, ls="--", alpha=alpha,color='orange',
                label=label_model, zorder=zorder,
            )

        pp.xlabel(datavar)
        pp.ylabel("Altitude (km)")

        # legend control (important!)
        if len(np.unique(m["orbit"])) <= 10:
            pp.legend(fontsize=10,frameon=False)
        else:
            pp.legend([], [], frameon=False)

        pp.savefig(f"profile_{datavar}_by_orbit.png", dpi=150, bbox_inches="tight")
        pp.close()

    breakpoint()

if __name__ == '__main__':
    main()