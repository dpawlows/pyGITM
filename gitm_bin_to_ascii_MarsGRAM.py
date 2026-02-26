#!/usr/bin/env python

# Switched from CubicSpine to interp1d(cubic) as we weren't using the full spline anyway so it is faster.

from glob import glob
import sys
import numpy as np
import re
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

def get_args(argv):
    filelist = []
    coordinates = 'geographic'
    help = False
    minalt = 40
    serial = False 

    for arg in argv:
        IsFound = 0

        if not IsFound:
            m = re.match(r'-h', arg)
            if m:
                help = 1
                IsFound = 1

            m = re.match(r'-serial', arg)
            if m:
                serial = True
                IsFound = 1

            m = re.match(r'-coordinates=(.*)', arg)
            if m:
                coordinates = m.group(1)
                IsFound = 1

            m = re.match(r'-minalt=(.*)', arg)
            if m:
                minalt = float(m.group(1))
                IsFound = 1

        if IsFound == 0 and not (arg == argv[0]):
            filelist.append(arg)

    return {
        'filelist': filelist,
        'coordinates': coordinates.lower(),
        'help': help,
        'minalt': minalt,
        'serial':serial,
    }


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

    gdLats = np.zeros((nLats-4, nAlts-ialtstart-2))
    gdAlts = np.zeros((nLats-4, nAlts-ialtstart-2))
 
    if coordinates == 'geodetic':
        #Create the grid
        #Our geodetic grid will default to the same as the geocentric grid
        #Longitudes are not affected
    
        ilon = 50
        for ialt in range(ialtstart,nAlts-2):
            for ilat in range(0,nLats-4):
                gcoordinates = np.array([[Lats[ilat+2],Lons[ilon],Alts[ialt]]]).transpose()
                gd = gc.geo2geodetic(gcoordinates,planet='mars')
                gdLats[ilat,ialt-ialtstart] = gd[0,0]
                gdAlts[ilat,ialt-ialtstart] = gd[2,0]

    # First, we will interpolate along the horizontal direction
    # for our interpolation
    # X = original Aerodetic Latitudes (assumed constant with altitude)
    # Y = our chosen variable (e.g. Tn or Ti, etc.)
    # newX = our desired Aerodetic grid

    newData = {k:np.zeros((nLons-4,nLats-4,nAlts-ialtstart-2)) for k in vars[3:]}
    newData['rho'] = np.zeros((nLons-4,nLats-4,nAlts-ialtstart-2))
    newData['pressure'] = np.zeros((nLons-4,nLats-4,nAlts-ialtstart-2))
    mass_array = np.array([masses[header['vars'][i]] * AMU for i in rhovars])


    for ilon in range(0,nLons - 4):
        for ialt in range(ialtstart,nAlts-2):

            X = gdLats[:,ialt-ialtstart] #We have data here
            #Do the interpolation- cubic spline in the horizontal direction for everything
            Y = gdLats[:,ialt-ialtstart]

            cs = CubicSpline(X,Y) 
            newLats = cs(newX)  #Latitude is weird. This is just a validation- newLats should equal newX. We want the grid to be regular.

            for var in vars[3:]:
                Y = data[var][ilon+2,2:-2,ialt]
                # cs = CubicSpline(X,Y)
                cs = interp1d(X, Y, kind='cubic', bounds_error=False, fill_value="extrapolate")
                newData[var][ilon,:,ialt-ialtstart] = cs(newX)

                
                if np.min(newData[var][ilon,:,ialt-ialtstart]) < 0 and var in rhovars:
                    #while rare, it is possible for the spline to give a negative number
                    imin = np.argmin(newData[var][ilon,:,ialt-ialtstart])
                    if imin == 0:
                        newData[var][ilon,imin,ialt-ialtstart] = Y[0]
                    else:
                        #so use linear instead
                        
                        lenvar = len(X)
                        valuesAfterMin = (lenvar-1) - imin # subtract 1 because we are comparing indices

                        if valuesAfterMin == 0:
                            #extrapoloate
                            iend = lenvar
                            extrapolate = True
                        else:
                            iend = imin + min(valuesAfterMin,5)

                        #Y = data[var][ilon,imin:imin+5,ialt] #this caused a problem if the negative number was near the edge
                        # of the grid
                        Y = data[var][ilon,imin:iend+1,ialt] #get the original data surrounding the negative
                        od = interp1d(X[max(imin-2,0):max(imin-2,0)+len(Y)],Y,fill_value='extrapolate')
                        newData[var][ilon,imin,ialt-ialtstart] = newX[imin-1:imin+2][1]
                        
                        if newData[var][ilon,imin,ialt-ialtstart] < 0:
                            # Still have an issue? Exponential interpolation
                            Y = np.log10(data[var][ilon,imin:iend+1,ialt]) #get the original data surrounding the negative
                            od = interp1d(X[imin-2:imin-2+len(Y)],Y,fill_value='extrapolate')
                            newData[var][ilon,imin,ialt-ialtstart] = 10**newX[imin-1:imin+2][1]

            densities = np.array([data[i][ilon+2, 2:-2, ialt] for i in rhovars])

            numberDensity = densities.sum(axis=0)
            rho = (densities * mass_array[:, None]).sum(axis=0)

            # cs = CubicSpline(X,rho)
            cs = interp1d(X, rho, kind='cubic', bounds_error=False, fill_value="extrapolate")
            newData['rho'][ilon,:,ialt-ialtstart] = cs(newX)

            #calculate pressure; p = nkT
            pressure = boltzmann*numberDensity*data[15][ilon+2,2:-2:,ialt]

            # cs = CubicSpline(X,pressure)
            cs = interp1d(X, pressure, kind='cubic', bounds_error=False, fill_value="extrapolate")
            newData['pressure'][ilon,:,ialt-ialtstart] = cs(newX)

    # Next, interpolate in the vertical
    # We have already done the latitudes

    nAltsNew = len(altitudeGrid)
    gdData = {k:np.zeros((nLons-4,nLats-4,nAltsNew)) for k in vars[3:]}
    gdData['rho'] = np.zeros((nLons-4,nLats-4,nAltsNew))
    gdData['pressure'] = np.zeros((nLons-4,nLats-4,nAltsNew))

    for ilon in range(0,nLons-4):
        for ilat in range(0,nLats-4):

            X = gdAlts[ilat,:]
            # the "new x" will be simply altitudeGrid, specified above

            for var in vars[3:]:
                if var not in logarithmic:
                    Y = newData[var][ilon,ilat,:]
                    # cs = CubicSpline(X,Y)
                    cs = interp1d(X, Y, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    gdData[var][ilon,ilat,:] = cs(altitudeGrid)
                else:                        
                    Y = np.log(newData[var][ilon,ilat,:])
                    # cs = CubicSpline(X,Y)
                    cs = interp1d(X, Y, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    gdData[var][ilon,ilat,:] = np.exp(cs(altitudeGrid))
                 
            Y = np.log(newData['rho'][ilon,ilat,:])
            # cs = CubicSpline(X,Y)
            cs = interp1d(X, Y, kind='cubic', bounds_error=False, fill_value="extrapolate")
            gdData['rho'][ilon,ilat,:] = np.exp(cs(altitudeGrid))
            
            Y = np.log(newData['pressure'][ilon,ilat,:])
            # cs = CubicSpline(X,Y)
            cs = interp1d(X, Y, kind='cubic', bounds_error=False, fill_value="extrapolate")
            gdData['pressure'][ilon,ilat,:] = np.exp(cs(altitudeGrid))


    for ilon in range(totalLons):
        for ilat in range(totalLats):
            for ialt in range(totalAlts):
                thisdata = [Lons[ilon+2]*rtod,newX[ilat]*rtod,altitudeGrid[ialt]]
                for var in vars[3:]:
                    thisdata.append(gdData[var][ilon,ilat,ialt])
                
                thisdata.append(gdData['rho'][ilon,ilat,ialt])
                #calculate and output pressure
                thisdata.append(gdData['pressure'][ilon,ilat,ialt])
                f.write("    ".join('{:g}'.format(ele) for ele in thisdata)+"\n")

    f.close()



def main():
    
    args = get_args(sys.argv)
    filelist = args['filelist']
    coordinates = args['coordinates']
    minalt = args['minalt']
    run_serial = args['serial']
 

    if coordinates not in coordoptions:
        print(f'{coordinates} is not a coordinate option')
        args['help'] = True

    if args['help'] or len(filelist) < 1:
        header = read_gitm_header(filelist)
        print('Usage:')
        print('gitm_bin_to_ascii_MarsGRAM.py -coordinates=geographic|geodetic -minalt=N [*.bin files]')
        print('Available variables:')
        for i, var in enumerate(header["vars"]):
            print(i, var)
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