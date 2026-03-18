#!/usr/bin/env python
# Plots photochemical and transport timescales from a single M-GITM binary file.
#
# Default mode:  requires -lat and -lon; produces an altitude profile of
#                tau_chem and tau_transport at that location.
#
# -plottransition: no -lat/-lon needed; produces a lat-lon colormesh of the
#                  altitude at which tau_chem = tau_transport.

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from gitm_routines import read_gitm_header, read_gitm_one_file, find_nearest_index
import marstiming as mt

rtod      = 180.0 / np.pi
minalt    = 100    # km
maxalt    = 250    # km

# Physical constants (SI)
kB        = 1.38065e-23          # J/K
mi        = 32.0 * 1.6605e-27   # kg, O2+ mass
g0        = 3.72                 # m/s^2, Mars surface gravity
R_Mars    = 3.3895e6             # m
K_O2p_CO2 = 2.85e-9 * 1e-6     # m^3/s

font    = {"label": 12, "tick": 11, "legend": 11, "text": 11}

def find_var_idx(hvars, *names):
    """Return index of first header var whose name contains any of *names (case-insensitive)."""
    for name in names:
        for i, v in enumerate(hvars):
            if name.lower() in v.lower():
                return i
    return None


def compute_timescales(ne, Te, Ti, n_CO2, alts_m, g_alt):
    """Return (tau_chem, tau_trans) arrays [s] on the altitude grid.

    ne, Te, Ti, n_CO2 : 1-D arrays over altitude
    alts_m            : altitude in metres (same length)
    g_alt             : gravitational acceleration in m/s^2 (same length)
    """
    # Recombination rate coefficient alpha [cm^3/s], branched on Ti
    alpha    = np.where(Ti <= 1200.0,
                        1.95e-13 * (Te / 300.0 ) ** (-0.7 ),
                        7.39e-14 * (Te / 1200.0) ** (-0.56))
    tau_chem  = 1.0 / (alpha * ne)                        # s

    H         = kB * Ti / (mi * g_alt)                   # plasma scale height [m]
    nu_in     = n_CO2 * K_O2p_CO2                        # ion-neutral collision freq [s^-1]
    Da        = 2.0 * kB * Ti / (mi * nu_in)             # ambipolar diffusion [m^2/s]
    tau_trans = H ** 2 / Da                              # s

    return tau_chem, tau_trans


def find_transition_alt(tau_chem, tau_trans, Alts_prof):
    """Find the altitude [km] where tau_chem crosses tau_trans (from below as alt increases).

    Returns np.nan if no crossing exists within the profile.
    """
    ratio = np.log(tau_chem / tau_trans)          # negative below transition, positive above
    for k in range(len(ratio) - 1):
        if ratio[k] <= 0.0 and ratio[k + 1] > 0.0:
            frac = -ratio[k] / (ratio[k + 1] - ratio[k])
            return Alts_prof[k] + frac * (Alts_prof[k + 1] - Alts_prof[k])
    return np.nan


def get_args(argv):
    parser = argparse.ArgumentParser(
        description='Plot photochemical and transport timescales from M-GITM binary output.')
    parser.add_argument('file', help='Single 3DALL binary file')
    parser.add_argument('-lat', type=float, default=None,
                        help='Latitude in degrees (required without -plottransition)')
    parser.add_argument('-lon', type=float, default=None,
                        help='Longitude in degrees (required without -plottransition)')
    parser.add_argument('-plottransition', action='store_true',
                        help='Plot transition altitude as a lat-lon colormesh')
    parser.add_argument('-vmin', type=float, default=None,
                        help='Minimum value for colorbar')
    parser.add_argument('-vmax', type=float, default=None,
                        help='Maximum value for colorbar')
    return parser.parse_args(argv[1:])


args = get_args(sys.argv)

if args.plottransition:
    if args.lat is not None or args.lon is not None:
        print('Warning: -lat and -lon are ignored with -plottransition')
else:
    if args.lat is None or args.lon is None:
        print('Error: -lat and -lon are required unless -plottransition is specified')
        sys.exit(1)

# --- Read header and identify required variable indices ---
header = read_gitm_header([args.file])

i_ne  = find_var_idx(header['vars'], 'e-')
i_Te  = find_var_idx(header['vars'], 'eTemperature')
i_Ti  = find_var_idx(header['vars'], 'iTemperature')
i_CO2 = find_var_idx(header['vars'], 'CO!D2!N')

missing = [n for n, i in [('e-', i_ne), ('eTemperature', i_Te),
                           ('iTemperature', i_Ti), ('[CO2]', i_CO2)] if i is None]
if missing:
    print('Error: could not find required variables: {}'.format(missing))
    sys.exit(1)

ts_vars = sorted(set([0, 1, 2, i_ne, i_Te, i_Ti, i_CO2]))

# --- Load file and set up grids ---
data     = read_gitm_one_file(args.file, ts_vars)
Alts_all = data[2][0][0] / 1000.0          # km
Lons_all = data[0][:, 0, 0] * rtod         # degrees E
Lats_all = data[1][0, :, 0] * rtod         # degrees N

ialt1 = find_nearest_index(Alts_all, minalt)
ialt2 = find_nearest_index(Alts_all, maxalt)

Alts_prof = Alts_all[ialt1:ialt2 + 1]
alts_m    = Alts_prof * 1000.0
g_alt     = g0 * (R_Mars / (R_Mars + alts_m)) ** 2

marsgeom = mt.getMarsSolarGeometry(data['time'])
ls = marsgeom.ls

# -----------------------------------------------------------------------
# Altitude-profile mode
# -----------------------------------------------------------------------
if not args.plottransition:
    plon = args.lon
    plat = args.lat
    ilon = find_nearest_index(Lons_all, plon)
    ilat = find_nearest_index(Lats_all, plat)

    ne    = data[i_ne ][ilon, ilat, ialt1:ialt2 + 1]
    Te    = data[i_Te ][ilon, ilat, ialt1:ialt2 + 1]
    Ti    = data[i_Ti ][ilon, ilat, ialt1:ialt2 + 1]
    n_CO2 = data[i_CO2][ilon, ilat, ialt1:ialt2 + 1]

    tau_chem, tau_trans = compute_timescales(ne, Te, Ti, n_CO2, alts_m, g_alt)

    sza = mt.getSZAfromTime(marsgeom, plon, plat)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.semilogx(tau_chem,  Alts_prof, label=r'$\tau_\mathrm{chem}$',      lw=2)
    ax.semilogx(tau_trans, Alts_prof, label=r'$\tau_\mathrm{transport}$', lw=2, ls='--')
    ax.set_xlabel('Time constant (s)', fontsize=font["label"])
    ax.set_ylabel('Altitude (km)', fontsize=font["label"])
    ax.set_title('Photochemical & diffusion timescales\n'
                 'Ls={:.0f}\u00b0, SZA={:.0f}\u00b0'.format(ls, sza),
                 fontsize=font["label"])
    ax.tick_params(labelsize=font["tick"])
    ax.legend(fontsize=font["legend"])
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    outfile = 'timescales_lat{}_lon{}.png'.format(int(plat), int(plon))
    plt.savefig(outfile)
    print('Wrote {}'.format(outfile))

# -----------------------------------------------------------------------
# Transition-altitude colormesh mode
# -----------------------------------------------------------------------
else:
    # Skip ghost cells at the boundaries of the GITM grid
    Lons_plot = Lons_all[1:-1]
    Lats_plot = Lats_all[1:-1]
    nLons = len(Lons_plot)
    nLats = len(Lats_plot)

    trans_alt = np.full((nLons, nLats), np.nan)

    # Compute SZA for the non-ghost grid and mask nightside cells
    Lon2d, Lat2d = np.meshgrid(Lons_plot, Lats_plot, indexing='ij')
    sza_grid = mt.getSZAfromTime(marsgeom, Lon2d, Lat2d)
    night_mask = sza_grid > 100.0

    for ii, ilon in enumerate(range(1, len(Lons_all) - 1)):
        for jj, ilat in enumerate(range(1, len(Lats_all) - 1)):
            if night_mask[ii, jj]:
                continue
            ne    = data[i_ne ][ilon, ilat, ialt1:ialt2 + 1]
            Te    = data[i_Te ][ilon, ilat, ialt1:ialt2 + 1]
            Ti    = data[i_Ti ][ilon, ilat, ialt1:ialt2 + 1]
            n_CO2 = data[i_CO2][ilon, ilat, ialt1:ialt2 + 1]

            tau_chem, tau_trans = compute_timescales(ne, Te, Ti, n_CO2, alts_m, g_alt)
            trans_alt[ii, jj]   = find_transition_alt(tau_chem, tau_trans, Alts_prof)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='lightgrey')
    trans_masked = np.ma.masked_where(night_mask, trans_alt)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    mesh = ax.pcolormesh(Lons_plot, Lats_plot, trans_masked.T,
                         cmap=cmap, shading='auto',
                         vmin=args.vmin, vmax=args.vmax)
    plt.colorbar(mesh, ax=ax, label='Transition altitude (km)',pad=0.01)
    ax.set_xlabel('Longitude (\u00b0E)',fontsize=font["label"])
    ax.set_ylabel('Latitude (\u00b0N)',fontsize=font["label"])
    # ax.set_title(r'O$_2^+$ P-T Transition Altitude (Ls={:.1f}$^\circ$)'.format(ls))
    plt.tight_layout()
    outfile = 'timescales_transition.png'
    plt.savefig(outfile)
    print('Wrote {}'.format(outfile))
