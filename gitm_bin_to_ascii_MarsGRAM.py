#!/usr/bin/env python

#Convert GITM output to ascii for use with MarsGram

# 1.Longitude(degree) 2.Latitude(degree) 3.SZA(degree) 4.Altitude(km) 5.Tn(K) 6.Ti(K) 7.Te(K)
# 8.nCO2(#/cc) 9.nO(#/cc) 10.nN2(#/cc) 11.nCO(#/cc) 12.nO2P(#/cc) 13.ne(#/cc) 14.UN(m/s)
# 15.VN(m/s), 16.WN(m/s)


from glob import glob
import sys
from gitm_routines import *
import gitm_coordinates as gc 
import re 
from scipy.interpolate import CubicSpline,interp1d
from scipy.constants import k as boltzmann

rtod = 180.0/3.141592
def get_args(argv):

    filelist = []
    coordinates = 'geographic'
    help = False 
    minalt = 40

    for arg in argv:
        IsFound = 0

        if (not IsFound):
            m = re.match(r'-h',arg)
            if m:
                help = 1
                IsFound = 1

            m = re.match(r'-coordinates=(.*)',arg)
            if m:
                coordinates = m.group(1)
                IsFound = 1

            m = re.match(r'-minalt=(.*)',arg)
            if m:
                minalt = float(m.group(1))
                IsFound = 1

        if IsFound==0 and not(arg==argv[0]):
                filelist.append(arg)

    args = {'filelist':filelist,
            'coordinates':coordinates.lower(),
            'help':help,
            'minalt':minalt,
            }

    return args


args = get_args(sys.argv)
coordoptions = ['geographic','geodetic']
coordinates = args['coordinates'].lower()
if coordinates not in coordoptions:
    print('{} is not a coordinate option'.format(coordinates))
    args['help'] = True 

filelist = args['filelist']
vars = [0,1,2,15,4,5,6,7,9,14,16,17,18]
logarithmic = [4,5,6,7,9,14]
minalt = args['minalt']
nFiles = len(filelist)
header = read_gitm_header(args["filelist"])

if args['help'] or len(filelist) < 1:
    print('Usage : ')
    print('gitm_bin_to_ascii_MarsGRAM.py -coordinates=coordinates -help')
    print('                   -minalt=minalt  [*.bin or a file]')
    print('   -help : print this message')
    print('   -coordinates=geographic or geodetic (default = geographic)')
    print('   At end, list the files you want to plot')
    
    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


i = 0
rhovars = [4,5,6,7,9,14] 
altitudeGrid = np.arange(60,302,2.5) #default grid is 2.5 km spacing between 60 and 300 km.
for file in filelist:

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

    if i == 0:
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
    
    for ilon in range(0,nLons - 4):
        for ialt in range(ialtstart,nAlts-2):

            X = gdLats[:,ialt-ialtstart] #We have data here
            #Do the interpolation- cubic spline in the horizontal direction for everything
            Y = gdLats[:,ialt-ialtstart]
            cs = CubicSpline(X,Y) 
            newLats = cs(newX)  #Latitude is weird. This is just a validation- newLats should equal newX. We want the grid to be regular.

            for var in vars[3:]:
                Y = data[var][ilon+2,2:-2,ialt]

                cs = CubicSpline(X,Y)
                newData[var][ilon,:,ialt-ialtstart] = cs(newX)
                if np.min(newData[var][ilon,:,ialt-ialtstart]) < 0 and var in rhovars:
                    #while rare, it is possible for the spline to give a negative number
                    #so use linear instead
                    imin = np.argmin(newData[var][ilon,:,ialt-ialtstart])
                    Y = data[var][ilon,imin:imin+5,ialt] #get the original data surrounding the negative
                    od = interp1d(X[imin-2:imin+3],Y)
                    newData[var][ilon,imin,ialt-ialtstart] = od(newX[imin-1:imin+2])[1]

            rho = 0
            numberDensity = 0
            for i in rhovars:
                n = data[i][ilon+2,2:-2,ialt]*masses[header['vars'][i]]
                thisDensity = n * AMU
                numberDensity = numberDensity + n 
                rho = rho + thisDensity
            cs = CubicSpline(X,rho)
            newData['rho'][ilon,:,ialt-ialtstart] = cs(newX)

            #calculate pressure; p = nkT
            pressure = boltzmann*numberDensity*data[15][ilon+2,2:-2:,ialt]
            cs = CubicSpline(X,pressure)
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
                    cs = CubicSpline(X,Y)
                    gdData[var][ilon,ilat,:] = cs(altitudeGrid)
                else:                        
                    Y = np.log(newData[var][ilon,ilat,:])
                    cs = CubicSpline(X,Y)
                    gdData[var][ilon,ilat,:] = np.exp(cs(altitudeGrid))
                 
                Y = np.log(newData['rho'][ilon,ilat,:])
                cs = CubicSpline(X,Y)
                gdData['rho'][ilon,ilat,:] = np.exp(cs(altitudeGrid))
                
                Y = np.log(newData['pressure'][ilon,ilat,:])
                cs = CubicSpline(X,Y)
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
