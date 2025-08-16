#!/usr/bin/env python

#Convert GITM output to ascii for use with M-AMPS
#Example output file is at https://app.box.com/s/hr6w0i2zsm6az9gzb2s2qysxtjaf5jse/file/840383303910
# 1.Longitude(degree) 2.Latitude(degree) 3.SZA(degree) 4.Altitude(km) 5.Tn(K) 6.Ti(K) 7.Te(K)
# 8.nCO2(#/m3) 9.nO(#/m3) 10.nN2(#/m3) 11.nCO(#/m3) 12.nAr($/m3) 13.nO2P(#/m3) 14.ne(#/m3) 15.UN(m/s)
# 16.VN(m/s), 17.WN(m/s)

"""Convert GITM output to ASCII for use with M-AMPS.

This version can process multiple files in parallel using multiple CPU
cores.  Files are converted independently, so the work is farmed out to a
``ProcessPoolExecutor`` and executed concurrently.
"""

import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from gitm_routines import *

rtod = 180.0/3.141592


def get_args(argv):

    filelist = []
    IsFound = 0

    for arg in argv:
        if IsFound==0 and not(arg==argv[0]):
            filelist.append(arg)

        args = {'filelist':filelist,
                }
    return args


def process_file(file, vars):
    """Convert a single GITM binary file to ASCII."""

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

    Alts = data[2][0][0]/1000.0
    Lons = data[0][:,0,0]*rtod
    Lats = data[1][0,:,0]*rtod
    with open(output,'w') as f:
        f.write("#MGITM Results on "+date+" at "+time+" UT."+"\n")
        f.write("#Each column contains the following variable at the given longitude, latitude, and altitude"+"\n")
        f.write("#Number of Longitude points: "+str(nLons)+"\n")
        f.write("#Number of Latitude points: "+str(nLats)+"\n")
        f.write("#Number of Altitude points: "+str(nAlts)+"\n")
        f.write("#Units-Densities: #/m3, temperatures: K, wind velocitiy: m/s."+"\n")
        f.write("#1.Longitude(degree) 2.Latitude(degree) 3.Altitude(km) 4.Tn(K) 5.Ti(K) 6.Te(K) 7.nCO2(#/m3)"
                "    8.nO(#/m3) 9.nN2(#/m3) 10.nCO(#/m3) 11.nO2P(#/m3) 12.nAr(#/m3) 12.ne(#/m3) 13.UN(m/s) 14.VN(m/s), 15.WN(m/s)\n")

        f.write("#START\n")

        ialtstart = np.where(Alts > 98)[0][0]

        #Begin 3D loop over data cube
        for ialt in range(ialtstart,nAlts-2):
            for ilat in range(2,nLats-2):
                for ilon in range(2,nLons-2):
                    thisdata = [Lons[ilon],Lats[ilat],Alts[ialt]]
                    for var in vars[3:]:
                        thisdata.append(data[var][ilon,ilat,ialt])

                    f.write("    ".join('{:g}'.format(ele) for ele in thisdata)+"\n")


def main():
    newargs = get_args(sys.argv)
    filelist = newargs['filelist']
    vars = [0,1,2,15,34,33,4,6,7,5,9, 28, 32, 16, 17, 18]

    if len(filelist) == 0:
        print('Usage: gitm_bin_to_ascii.py filename(s)')
        return

    max_workers = min(multiprocessing.cpu_count(), len(filelist))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file, vars): file for file in filelist}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting", unit="file"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")


if __name__ == '__main__':
    main()
