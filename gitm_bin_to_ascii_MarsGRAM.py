#!/usr/bin/env python

import argparse
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.constants import k as boltzmann
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from gitm_routines import *
import gitm_coordinates as gc

rtod = 180.0 / 3.141592
coordoptions = ['geographic', 'geodetic']
altitudeGrid = np.arange(60, 302, 2.5)  # Output vertical grid
vars = [0, 1, 2, 15, 4, 5, 6, 7, 9, 14, 16, 17, 18]
logarithmic = [4, 5, 6, 7, 9, 14]
rhovars = [4, 5, 6, 7, 9, 14]
header = None

def get_args():
    parser = argparse.ArgumentParser(
        description='Convert GITM binary output to MarsGRAM ASCII format.'
    )
    parser.add_argument(
        'filelist',
        nargs='*',
        help='Input GITM .bin files to process',
    )
    parser.add_argument(
        '-coordinates', '--coordinates',
        default='geographic',
        choices=coordoptions,
        help='Coordinate system to use for interpolation (default: geographic)',
    )
    parser.add_argument(
        '-minalt', '--minalt',
        type=float,
        default=40.0,
        help='Minimum altitude (km) used to begin interpolation (default: 40)',
    )
    parser.add_argument(
        '-serial', '--serial',
        action='store_true',
        help='Run file conversions serially (default: parallel)',
    )
    return parser.parse_args()


def process_one_file(file, minalt, coordinates,header):
    data = read_gitm_one_file(file, vars)
    pos = file.rfind('.bin')
    tstamp = file[pos-13:pos]
    output = 'GITM_'+tstamp+'.dat'
    year = tstamp[0:2]
    month = tstamp[2:4]
    day = tstamp[4:6]
    hour = tstamp[7:9]
    minute = tstamp[9:11]
    second = tstamp[11:13]

    year = '20'+year if int(year) < 40 else '19'+year
    date = year+'-'+month+'-'+day
    time = hour+':'+minute+":"+second


    nLons = data['nLons'] 
    nLats = data['nLats'] 
    nAlts = data['nAlts'] 

    Alts = data[2][0][0]/1000.
    ialtstart = np.where(Alts > minalt)[0][0]
    tempalts = Alts[ialtstart:-2]/1000.0
    Lons = data[0][:,0,0]
    Lats = data[1][0,:,0]
    
    #This is the grid that we want to output on
    newX = Lats[2:-2] 
    totalLons = len(Lons[2:-2])
    totalLats = len(newX)
    totalAlts = len(altitudeGrid)

    f = open(output,'w')
    f.write("#MGITM Results on "+date+" at "+time+" UT."+"\n")
    f.write("#Each column contains the following variable at the given longitude, latitude, and altitude"+"\n")
    f.write("#Number of Longitude points: "+str(totalLons)+"\n")
    f.write("#Number of Latitude points: "+str(totalLats)+"\n")
    f.write("#Number of Altitude points: "+str(totalAlts)+"\n")
    f.write("#Units   Densities: #/m3, temperatures: K, wind speeds: m/s., rho: kg/m3,   pressure: Pa"+"\n")
    myvars = ["".join(data['vars'][i].decode().split()) for i in vars]
    myvars2 = "   ".join(name_dict[i] for i in myvars)
    f.write(myvars2+'   rho   pressure'+"\n")
    f.write("#START\n")

    n_lons_core = nLons - 4
    n_lats_core = nLats - 4
    n_alts_core = nAlts - ialtstart - 2
    lons_core = Lons[2:-2]

    gdLats = np.broadcast_to(Lats[2:-2, None], (n_lats_core, n_alts_core)).copy()
    gdAlts = np.broadcast_to(tempalts[None, :], (n_lats_core, n_alts_core)).copy()
    
 
    if coordinates == 'geodetic':
        #Create the grid
        #Our geodetic grid will default to the same as the geocentric grid
        #Longitudes are not affected
    
        ilon = 50
        for ialt in range(ialtstart,nAlts-2):
            for ilat in range(0,n_lats_core):
                gcoordinates = np.array([[Lats[ilat+2],Lons[ilon],Alts[ialt]]]).transpose()
                gd = gc.geo2geodetic(gcoordinates,planet='mars')
                gdLats[ilat,ialt-ialtstart] = gd[0,0]
                gdAlts[ilat,ialt-ialtstart] = gd[2,0]

    # First, we will interpolate along the horizontal direction
    # for our interpolation
    # X = original Aerodetic Latitudes (assumed constant with altitude)
    # Y = our chosen variable (e.g. Tn or Ti, etc.)
    # newX = our desired Aerodetic grid

    interp_vars = vars[3:]
    data_slices = {var: data[var][2:-2, 2:-2, ialtstart:nAlts-2] for var in interp_vars}
    number_density = np.zeros((n_lons_core, n_lats_core, n_alts_core))
    rho_raw = np.zeros((n_lons_core, n_lats_core, n_alts_core))
    for i in rhovars:
        ni = data[i][2:-2, 2:-2, ialtstart:nAlts-2]
        number_density += ni
        rho_raw += ni * masses[header['vars'][i]] * AMU

    pressure_raw = boltzmann * number_density * data[15][2:-2, 2:-2, ialtstart:nAlts-2]

    newData = {k:np.zeros((n_lons_core, n_lats_core, n_alts_core)) for k in interp_vars}
    newData['rho'] = np.zeros((n_lons_core, n_lats_core, n_alts_core))
    newData['pressure'] = np.zeros((n_lons_core, n_lats_core, n_alts_core))

    lat_spline_cache = {}
    
    for ilon in range(0, n_lons_core):
        for ialt in range(ialtstart,nAlts-2):
            ialt_idx = ialt - ialtstart

            X = gdLats[:, ialt_idx] #We have data here
            #Do the interpolation- cubic spline in the horizontal direction for everything
            Y = gdLats[:,ialt-ialtstart]
            cs = CubicSpline(X,Y) 
            newLats = cs(newX)  #Latitude is weird. This is just a validation- newLats should equal newX. We want the grid to be regular.

            spline_key = tuple(np.round(X, 10))
            if spline_key not in lat_spline_cache:
                lat_spline_cache[spline_key] = (X.copy(), np.argsort(X))
            x_sorted, x_order = lat_spline_cache[spline_key]

            def spline_to_newx(y_values):
                cs_local = CubicSpline(x_sorted, y_values[x_order], extrapolate=True)
                return cs_local(newX)

            for var in interp_vars:
                Y = data_slices[var][ilon, :, ialt_idx]
                newData[var][ilon,:,ialt_idx] = spline_to_newx(Y)

                
                if np.min(newData[var][ilon,:,ialt_idx]) < 0 and var in rhovars:
                    #while rare, it is possible for the spline to give a negative number
                    imin = np.argmin(newData[var][ilon,:,ialt_idx])
                    if imin == 0:
                        newData[var][ilon,imin,ialt_idx] = Y[0]
                    else:
                        #so use linear instead
                        
                        lenvar = len(x_sorted)
                        valuesAfterMin = (lenvar-1) - imin # subtract 1 because we are comparing indices

                        if valuesAfterMin == 0:
                            #extrapoloate
                            iend = lenvar
                            extrapolate = True
                        else:
                            iend = imin + min(valuesAfterMin,5)

                        #Y = data[var][ilon,imin:imin+5,ialt] #this caused a problem if the negative number was near the edge
                        # of the grid
                        y_patch = Y[imin:iend+1] #get the original data surrounding the negative
                        x_patch = x_sorted[max(imin-2,0):max(imin-2,0)+len(y_patch)]
                        od = interp1d(x_patch, y_patch, fill_value='extrapolate')
                        newData[var][ilon,imin,ialt_idx] = od(newX[imin])
                        
                        if newData[var][ilon,imin,ialt_idx] < 0:
                            # Still have an issue? Exponential interpolation
                            y_patch = np.log10(np.clip(Y[imin:iend+1], 1e-300, None)) #get the original data surrounding the negative
                            x_patch = x_sorted[max(imin-2,0):max(imin-2,0)+len(y_patch)]
                            od = interp1d(x_patch, y_patch, fill_value='extrapolate')
                            newData[var][ilon,imin,ialt_idx] = 10**od(newX[imin])

                        # if (ilon == 10 and ialt >= 117):
                        #     breakpoint(newData[var][ilon,imin,ialt-ialtstart])

            newData['rho'][ilon,:,ialt_idx] = spline_to_newx(rho_raw[ilon, :, ialt_idx])
            newData['pressure'][ilon,:,ialt_idx] = spline_to_newx(pressure_raw[ilon, :, ialt_idx])

    # Next, interpolate in the vertical
    # We have already done the latitudes

    nAltsNew = len(altitudeGrid)
    gdData = {k:np.zeros((n_lons_core,n_lats_core,nAltsNew)) for k in interp_vars}
    gdData['rho'] = np.zeros((n_lons_core,n_lats_core,nAltsNew))
    gdData['pressure'] = np.zeros((n_lons_core,n_lats_core,nAltsNew))

    for ilon in range(0, n_lons_core):
        for ilat in range(0, n_lats_core):

            X = gdAlts[ilat,:]
            # the "new x" will be simply altitudeGrid, specified above

            for var in interp_vars:
                if var not in logarithmic:
                    Y = newData[var][ilon,ilat,:]
                    cs = CubicSpline(X,Y)
                    gdData[var][ilon,ilat,:] = cs(altitudeGrid)
                else:                        
                    Y = np.log(newData[var][ilon,ilat,:])
                    cs = CubicSpline(X,Y)
                    gdData[var][ilon,ilat,:] = np.exp(cs(altitudeGrid))
                 
            Y = np.log(np.clip(newData['rho'][ilon,ilat,:], 1e-300, None))
            cs = CubicSpline(X,Y)
            gdData['rho'][ilon,ilat,:] = np.exp(cs(altitudeGrid))

            Y = np.log(np.clip(newData['pressure'][ilon,ilat,:], 1e-300, None))
            cs = CubicSpline(X,Y)
            gdData['pressure'][ilon,ilat,:] = np.exp(cs(altitudeGrid))

    lon_grid, lat_grid, alt_grid = np.meshgrid(
        lons_core * rtod,
        newX * rtod,
        altitudeGrid,
        indexing='ij',
    )

    output_columns = [lon_grid.ravel(), lat_grid.ravel(), alt_grid.ravel()]
    for var in interp_vars:
        output_columns.append(gdData[var].ravel())
    output_columns.append(gdData['rho'].ravel())
    output_columns.append(gdData['pressure'].ravel())
    out_matrix = np.column_stack(output_columns)
    np.savetxt(f, out_matrix, fmt='%g', delimiter='    ')

    f.close()



def main():

    args = get_args()
    filelist = args.filelist
    coordinates = args.coordinates
    minalt = args.minalt
    run_serial = args.serial

    if len(filelist) < 1:
        print('No input files provided.')
        print('Use -h/--help for usage information.')
        return

    header = read_gitm_header(filelist)
    worker = partial(process_one_file, minalt=minalt, coordinates=coordinates,header=header)

    if run_serial:
        for file in tqdm(filelist, desc="Serial processing"):
            worker(file)
    else:
        with ProcessPoolExecutor(max_workers=16) as executor:
                    list(tqdm(executor.map(worker, filelist, chunksize=1),
                            total=len(filelist), desc="Parallel processing"))

if __name__ == "__main__":
    main()
