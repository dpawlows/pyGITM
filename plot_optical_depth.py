#!/usr/bin/env python
# Plots optical depth vs altitude for every wavelength bin in a 3DUSR file
# at the grid location whose solar zenith angle is closest to -sza.

import argparse
import re
import numpy as np
import matplotlib.pyplot as pp
from matplotlib import cm, colors
import gitm_routines as gr
import marstiming as mt

rtod = 180.0/np.pi
nGhost = 2


def get_args():
    parser = argparse.ArgumentParser(
        description='Plot optical depth vs altitude for all wavelength bins')
    parser.add_argument('filename', nargs='?', default=None,
                        help='3DUSR file (default: first 3DUSR*.bin found)')
    parser.add_argument('-sza', type=float, default=0.0,
                        help='solar zenith angle in degrees (default: 0)')
    parser.add_argument('-minalt', type=float, default=None,
                        help='minimum altitude to plot (km)')
    parser.add_argument('-maxalt', type=float, default=None,
                        help='maximum altitude to plot (km)')
    parser.add_argument('-mintau', type=float, default=1e-4,
                        help='minimum optical depth to plot (default: 1e-4)')
    parser.add_argument('-maxtau', type=float, default=None,
                        help='maximum optical depth to plot (default: data max)')
    parser.add_argument('-o', dest='outfile', default=None,
                        help='output png filename')
    return parser.parse_args()


args = get_args()

if args.filename is None:
    from glob import glob
    filelist = sorted(glob('3DUSR*.bin'))
    if len(filelist) == 0:
        raise SystemExit('No 3DUSR files found.')
    args.filename = filelist[0]

header = gr.read_gitm_header([args.filename])
iTau = [i for i, v in enumerate(header['vars']) if v.startswith('Tau')]
wavelengths = np.array([float(header['vars'][i][3:]) for i in iTau])
print(f"Found {len(iTau)} wavelength bins in {args.filename}")

data = gr.read_gitm_one_file(args.filename, [0, 1, 2] + iTau)

# Strip ghost cells
s = slice(nGhost, -nGhost)
lons = data[0][s, s, 0]*rtod
lats = data[1][s, s, 0]*rtod
alts = data[2][0, 0, s]/1000.0

# Find the grid point closest to the requested SZA
timedata = mt.getMarsSolarGeometry(data['time'])
sza = mt.getSZAfromTime(timedata, lons, lats)
iLon, iLat = np.unravel_index(np.argmin(np.abs(sza - args.sza)), sza.shape)
szaLoc = sza[iLon, iLat]
print(f"Time: {data['time']}  Ls: {timedata.ls:.1f}")
print(f"Requested SZA: {args.sza:.1f}  Closest SZA: {szaLoc:.2f} at "
      f"lon {lons[iLon, iLat]:.1f}, lat {lats[iLon, iLat]:.1f}")

# Build an altitude x wavelength array, with bins sorted by wavelength.
# Columns are evenly spaced by bin (labeled with wavelength) since the bins
# are irregular and some are only a fraction of an angstrom apart.
order = np.argsort(wavelengths)
nBins = len(order)
tau = np.array([data[iTau[j]][s, s, s][iLon, iLat, :] for j in order]).T
tau = np.ma.masked_less_equal(tau, 0)

maxtau = args.maxtau if args.maxtau is not None else tau.max()
norm = colors.LogNorm(vmin=args.mintau, vmax=maxtau)

# Cell edges
xEdges = np.arange(nBins + 1) - 0.5
altMid = 0.5*(alts[1:] + alts[:-1])
yEdges = np.concatenate(([alts[0] - (altMid[0] - alts[0])], altMid,
                         [alts[-1] + (alts[-1] - altMid[-1])]))

fig, ax = pp.subplots(figsize=(12, 7))
mesh = ax.pcolormesh(xEdges, yEdges, tau, norm=norm, cmap=cm.turbo)
ax.contour(np.arange(nBins), alts, tau, levels=[1.0],
           colors='w', linewidths=1.5)

ax.set_xticks(np.arange(nBins))
ax.set_xticklabels([f"{wavelengths[j]:g}" for j in order],
                   rotation=90, fontsize=7)
ax.set_ylim(args.minalt if args.minalt is not None else yEdges[0],
            args.maxalt if args.maxalt is not None else yEdges[-1])
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Altitude (km)')
ax.set_title(f"{data['time']:%Y-%m-%d %H:%M:%S} UT   SZA = {szaLoc:.1f}°   "
             f"Lon {lons[iLon, iLat]:.1f}°E, Lat {lats[iLon, iLat]:.1f}°N\n"
             f"(white line: τ = 1)")

cbar = fig.colorbar(mesh, ax=ax, extend='both')
cbar.set_label('Optical depth')

outfile = args.outfile
if outfile is None:
    m = re.search(r'_t(\d{6}_\d{6})', args.filename)
    stamp = m.group(1) if m else 'out'
    outfile = f"tau_alt_sza{args.sza:03.0f}_{stamp}.png"
pp.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"Saved {outfile}")
