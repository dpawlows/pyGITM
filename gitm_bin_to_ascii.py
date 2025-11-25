#!/usr/bin/env python

#Convert GITM output to ascii for use with M-AMPS
#Example output file is at https://app.box.com/s/hr6w0i2zsm6az9gzb2s2qysxtjaf5jse/file/840383303910
# 1.Longitude(degree) 2.Latitude(degree) 3.SZA(degree) 4.Altitude(km) 5.Tn(K) 6.Ti(K) 7.Te(K)
# 8.nCO2(#/m3) 9.nO(#/m3) 10.nN2(#/m3) 11.nCO(#/m3) 12.nAr($/m3) 13.nO2P(#/m3) 14.ne(#/m3) 15.UN(m/s)
# 16.VN(m/s), 17.WN(m/s)

#Usage: gitm_bin_to_ascii.py filename(s)

from glob import glob
import sys
from gitm_routines import *
import re 
import marstiming
import datetime 

rtod = 180.0/3.141592

def get_args(argv):

    filelist = []
    IsFound = 0
    var = -1
    help = 0
    ghost = -1

    for arg in argv:
        m = re.match(r'-var=(.*)',arg)
        if m:
            var = m.group(1)
            IsFound = 1
        m = re.match(r'-h',arg)
        if m:
            help = 1
            IsFound = 1

        m = re.match(r'-g',arg)
        if m:
            ghost = 1
            IsFound = 1

        if IsFound==0 and not(arg==argv[0]):
            filelist.append(arg)

        args = {'filelist':filelist,
                 'var':var,
                 'help':help,
                 'ghost':ghost,
                }
    return args


args = get_args(sys.argv)
filelist = args['filelist']
header = read_gitm_header(filelist)
if args['var'] == -1:
    args['help'] = True

if (args["help"]):

    print('Usage : ')
    print('gitm_bin_to_ascii.py -var=n1[,n2,n3]')
    print('                     -help [*.bin or a file]')
    print('   -help : print this message')
    print('   -var=num1,num2,... : number(s) is variable to plot')
    print('   -g: include ghost cells')
    iVar = 0
    for var in header["vars"]:
        print(iVar,var)
        iVar=iVar+1

    exit()


vars = [0,1,2]
myvars = [int(v) for v in args["var"].split(',')]
vars.extend(myvars)
nFiles = len(filelist)
i = 0

ghost = False 
if args['ghost'] > 0:
    ghost = True

for file in filelist:
    print(f"Processing {file}...")
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
        Alts = data[2][0][0]/1000.0;
        Lons = data[0][:,0,0]*rtod;
        Lats = data[1][0,:,0]*rtod;
        nLons = len(Lons)
        nLats = len(Lats)
        nAlts = len(Alts)

        ilonstart = 0
        ilonend = len(Lons)
        ilatstart = 0
        ilatend = len(Lats)
        ialtstart = 0
        ialtend = len(Alts)
    
        ialtstart = 0
        
        if not ghost:
            ialtstart = np.where(Alts > 98)[0][0] 
            ialtend -= 2 
            ilonstart += 2
            ilonend -= 2
            ilatstart += 2
            ilatend -= 2    
     
    f = open(output,'w')
    MSG = marstiming.getMarsSolarGeometry(datetime.datetime(int(year),int(month),int(day),int(hour),
    int(minute),int(second)))

    f.write("#MGITM Results on "+date+" at "+time+" UT."+ "\n")
    f.write("# LS:"+str(int(MSG.ls)))
    f.write("#Each column contains the following variable at the given longitude, latitude, and altitude"+"\n")
    f.write("#Number of Longitude points: "+str(ilonend-ilonstart)+"\n")
    f.write("#Number of Latitude points: "+str(ilatend-ilatstart)+"\n")
    f.write("#Number of Altitude points: "+str(ialtend-ialtstart)+"\n")
    f.write("#Units-Densities: #/m3, temperatures: K, wind velocitiy: m/s."+"\n")
    f.write("#")
    for i,v in enumerate(vars):
        f.write(f"{i+1}.{clean_varname(name_dict[header['vars'][v]])} ")
    f.write("\n")
    f.write("#START\n")


    #Begin 3D loop over data cube
    for ialt in range(ialtstart,ialtend):
        for ilat in range(ilatstart,ilatend):
            for ilon in range(ilonstart,ilonend):
                thisdata = [Lons[ilon],Lats[ilat],Alts[ialt]]
                for var in vars[3:]:
                    thisdata.append(data[var][ilon,ilat,ialt])

                f.write("    ".join('{:g}'.format(ele) for ele in thisdata)+"\n")

    i += 1
    f.close()
