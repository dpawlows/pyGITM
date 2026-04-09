#!/usr/bin/env python
# Plots photochemical and transport timescales from M-GITM binary files.
#
# Default mode:  requires -lat and -lon; produces an altitude profile of
#                tau_chem and tau_transport at that location.
#
# -plottransition: no -lat/-lon needed; produces a lat-lon colormesh of the
#                  altitude at which tau_chem = tau_transport.
#
# -solar (with -plottransition): sun-synchronous polar view of transition
#                  altitude as a function of SZA and solar azimuth.
#
# Multiple files: subplots in 2 columns with a shared colorbar.

import argparse
import math
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

font    = {"label": 20, "tick": 18, "legend": 21, "text": 18, "title": 28}
font2d  = {"label": 22, "tick": 20, "text": 20, "title": 28}


def compute_solar_azimuth(marsgeom, lons, lats):
    """Azimuth (degrees, 0-360) of each point measured clockwise from North
    at the sub-solar point — i.e. the angular position as seen from the sun.

    lons, lats : arrays of East longitudes / latitudes in degrees
    """
    lons  = np.asarray(lons)
    lats  = np.asarray(lats)
    dlon  = np.radians(lons - marsgeom.subSolarLon)
    lat   = np.radians(lats)
    lat_s = np.radians(marsgeom.solarDec)
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat_s) * np.sin(lat) - np.sin(lat_s) * np.cos(lat) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360.0


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
    alpha    = np.where(Ti <= 1200.0,
                        1.95e-13 * (Te / 300.0 ) ** (-0.7 ),
                        7.39e-14 * (Te / 1200.0) ** (-0.56))
    tau_chem  = 1.0 / (alpha * ne)

    H         = kB * Ti / (mi * g_alt)
    nu_in     = n_CO2 * K_O2p_CO2
    Da        = 2.0 * kB * Ti / (mi * nu_in)
    tau_trans = H ** 2 / Da

    return tau_chem, tau_trans


def find_transition_alt(tau_chem, tau_trans, Alts_prof):
    """Find the altitude [km] where tau_chem crosses tau_trans (from below as alt increases).

    Returns np.nan if no crossing exists within the profile.
    """
    ratio = np.log(tau_chem / tau_trans)
    for k in range(len(ratio) - 1):
        if ratio[k] <= 0.0 and ratio[k + 1] > 0.0:
            frac = -ratio[k] / (ratio[k + 1] - ratio[k])
            return Alts_prof[k] + frac * (Alts_prof[k + 1] - Alts_prof[k])
    return np.nan


def process_file(filepath, ts_vars, i_ne, i_Te, i_Ti, i_CO2, args):
    """Load one file and return a dict of plot-ready data."""
    data     = read_gitm_one_file(filepath, ts_vars)
    Alts_all = data[2][0][0] / 1000.0
    Lons_all = data[0][:, 0, 0] * rtod
    Lats_all = data[1][0, :, 0] * rtod

    ialt1 = find_nearest_index(Alts_all, minalt)
    ialt2 = find_nearest_index(Alts_all, maxalt)
    Alts_prof = Alts_all[ialt1:ialt2 + 1]
    alts_m    = Alts_prof * 1000.0
    g_alt     = g0 * (R_Mars / (R_Mars + alts_m)) ** 2

    marsgeom = mt.getMarsSolarGeometry(data['time'])

    if not args.plottransition:
        # --- Profile mode ---
        ilon = find_nearest_index(Lons_all, args.lon)
        ilat = find_nearest_index(Lats_all, args.lat)
        ne    = data[i_ne ][ilon, ilat, ialt1:ialt2 + 1]
        Te    = data[i_Te ][ilon, ilat, ialt1:ialt2 + 1]
        Ti    = data[i_Ti ][ilon, ilat, ialt1:ialt2 + 1]
        n_CO2 = data[i_CO2][ilon, ilat, ialt1:ialt2 + 1]
        tau_chem, tau_trans = compute_timescales(ne, Te, Ti, n_CO2, alts_m, g_alt)
        sza = mt.getSZAfromTime(marsgeom, args.lon, args.lat)
        return {'mode': 'profile', 'Alts_prof': Alts_prof,
                'tau_chem': tau_chem, 'tau_trans': tau_trans,
                'ls': marsgeom.ls, 'sza': sza}

    # --- Transition-altitude modes ---
    Lons_plot = Lons_all[1:-1]
    Lats_plot = Lats_all[1:-1]
    nLons = len(Lons_plot)
    nLats = len(Lats_plot)

    trans_alt = np.full((nLons, nLats), np.nan)
    Lon2d, Lat2d = np.meshgrid(Lons_plot, Lats_plot, indexing='ij')
    sza_grid   = mt.getSZAfromTime(marsgeom, Lon2d, Lat2d)
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

    if not args.solar:
        return {'mode': 'transition',
                'Lons_plot': Lons_plot, 'Lats_plot': Lats_plot,
                'trans_alt': trans_alt, 'night_mask': night_mask,
                'ls': marsgeom.ls}

    # --- Solar / sun-synchronous mode ---
    az_grid = compute_solar_azimuth(marsgeom, Lon2d, Lat2d)

    n_sza = 18
    n_az  = 36
    sza_edges = np.linspace(0.0, 90.0,  n_sza + 1)
    az_edges  = np.linspace(0.0, 360.0, n_az  + 1)

    sum_arr   = np.zeros((n_sza, n_az))
    count_arr = np.zeros((n_sza, n_az))

    for ii in range(nLons):
        for jj in range(nLats):
            if night_mask[ii, jj] or np.isnan(trans_alt[ii, jj]):
                continue
            isza = min(int(sza_grid[ii, jj] / 90.0 * n_sza), n_sza - 1)
            iaz  = min(int(az_grid[ii, jj]  / 360.0 * n_az),  n_az  - 1)
            sum_arr[isza, iaz]   += trans_alt[ii, jj]
            count_arr[isza, iaz] += 1

    with np.errstate(invalid='ignore'):
        binned = np.where(count_arr > 0, sum_arr / count_arr, np.nan)

    if args.interpolate and np.any(np.isnan(binned)):
        from scipy.interpolate import griddata
        sza_centers = 0.5 * (sza_edges[:-1] + sza_edges[1:])
        az_centers  = 0.5 * (az_edges[:-1]  + az_edges[1:])
        SZA_c, AZ_c = np.meshgrid(sza_centers, az_centers, indexing='ij')
        valid = ~np.isnan(binned)
        if valid.any():
            # Triplicate along azimuth so interpolation wraps at 0/360
            pts_sza = np.tile(SZA_c[valid], 3)
            pts_az  = np.concatenate([AZ_c[valid] - 360.0,
                                      AZ_c[valid],
                                      AZ_c[valid] + 360.0])
            pts_val = np.tile(binned[valid], 3)
            xi = np.column_stack([SZA_c.ravel(), AZ_c.ravel()])
            filled = griddata(np.column_stack([pts_sza, pts_az]),
                              pts_val, xi, method='linear')
            filled = filled.reshape(n_sza, n_az)
            nan_mask = np.isnan(binned)
            binned[nan_mask] = filled[nan_mask]

    return {'mode': 'solar', 'binned': binned,
            'sza_edges': sza_edges, 'az_edges': az_edges,
            'ls': marsgeom.ls}


def get_args(argv):
    parser = argparse.ArgumentParser(
        description='Plot photochemical and transport timescales from M-GITM binary output.')
    parser.add_argument('files', nargs='+', help='One or more 3DALL binary files')
    parser.add_argument('-lat', type=float, default=None,
                        help='Latitude in degrees (required without -plottransition)')
    parser.add_argument('-lon', type=float, default=None,
                        help='Longitude in degrees (required without -plottransition)')
    parser.add_argument('-plottransition', action='store_true',
                        help='Plot transition altitude as a lat-lon colormesh')
    parser.add_argument('-solar', action='store_true',
                        help='With -plottransition: plot transition altitude vs SZA and '
                             'solar azimuth (sun-synchronous polar view)')
    parser.add_argument('-interpolate', action='store_true',
                        help='With -solar: fill data gaps via linear interpolation')
    parser.add_argument('-vmin', type=float, default=None,
                        help='Minimum value for colorbar')
    parser.add_argument('-vmax', type=float, default=None,
                        help='Maximum value for colorbar')
    return parser.parse_args(argv[1:])


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
args = get_args(sys.argv)

if args.interpolate and not args.solar:
    print('Error: -interpolate requires -solar')
    sys.exit(1)

if args.solar and not args.plottransition:
    print('Error: -solar requires -plottransition')
    sys.exit(1)

if args.plottransition:
    if args.lat is not None or args.lon is not None:
        print('Warning: -lat and -lon are ignored with -plottransition')
else:
    if args.lat is None or args.lon is None:
        print('Error: -lat and -lon are required unless -plottransition is specified')
        sys.exit(1)

# ---------------------------------------------------------------------------
# Read header (once) and identify required variable indices
# ---------------------------------------------------------------------------
header = read_gitm_header([args.files[0]])

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

# ---------------------------------------------------------------------------
# Process all files
# ---------------------------------------------------------------------------
results = []
for f in args.files:
    print('Processing {}...'.format(f))
    results.append(process_file(f, ts_vars, i_ne, i_Te, i_Ti, i_CO2, args))

nfiles = len(results)
ncols  = min(2, nfiles)
nrows  = math.ceil(nfiles / ncols)

# ---------------------------------------------------------------------------
# Determine shared colorbar range for 2-D modes
# ---------------------------------------------------------------------------
if args.plottransition:
    if args.solar:
        all_vals = [r['binned'] for r in results]
    else:
        all_vals = [r['trans_alt'] for r in results]
    vmin = args.vmin if args.vmin is not None else min(np.nanmin(v) for v in all_vals)
    vmax = args.vmax if args.vmax is not None else max(np.nanmax(v) for v in all_vals)
else:
    vmin = vmax = None

cmap = plt.get_cmap('viridis').copy()
cmap.set_bad(color='lightgrey')

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
if args.solar:
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             subplot_kw={'projection': 'polar'},
                             figsize=(7 * ncols, 7 * nrows))
elif args.plottransition:
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(8 * ncols, 4 * nrows))
else:
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(6 * ncols, 6 * nrows))

axes_flat = axes.ravel()

# Hide any unused subplot slots
for ax in axes_flat[nfiles:]:
    ax.set_visible(False)

# ---------------------------------------------------------------------------
# Populate subplots
# ---------------------------------------------------------------------------
last_mesh = None

for idx, result in enumerate(results):
    ax = axes_flat[idx]
    if result['mode'] == 'profile':
        ax.semilogx(result['tau_chem'],  result['Alts_prof'],
                    label=r'$\tau_\mathrm{chem}$',      lw=2)
        ax.semilogx(result['tau_trans'], result['Alts_prof'],
                    label=r'$\tau_\mathrm{transport}$', lw=2, ls='--')
        ax.set_xlabel('Time constant (s)', fontsize=font["label"])
        ax.set_ylabel('Altitude (km)',      fontsize=font["label"])
        ax.set_title('L$_S$ = {:d}\u00b0'.format(round(int(result['ls']))),
                     fontsize=font["title"])
        ax.tick_params(labelsize=font["tick"])
        ax.legend(fontsize=font["legend"])
        ax.grid(True, which='both', alpha=0.3)

    elif result['mode'] == 'transition':
        trans_masked = np.ma.masked_where(result['night_mask'], result['trans_alt'])
        last_mesh = ax.pcolormesh(result['Lons_plot'], result['Lats_plot'],
                                  trans_masked.T, cmap=cmap, shading='auto',
                                  vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (\u00b0E)', fontsize=font2d["label"])
        ax.set_ylabel('Latitude (\u00b0N)',  fontsize=font2d["label"])
        ax.set_title('L$_S$ = {:d}\u00b0'.format(round(int(result['ls']))),
                     fontsize=font2d["title"])
        ax.tick_params(labelsize=font2d["tick"])

    elif result['mode'] == 'solar':
        T, R = np.meshgrid(np.radians(result['az_edges']), result['sza_edges'])
        binned_masked = np.ma.masked_invalid(result['binned'])
        last_mesh = ax.pcolormesh(T, R, binned_masked, cmap=cmap, shading='auto',
                                  vmin=vmin, vmax=vmax)
        ax.set_rmax(90)
        ax.set_rticks([30, 60, 90])
        ax.set_rlabel_position(45)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # Suppress the 45° azimuth tick label — it overlaps the 90° SZA label
        ax.set_thetagrids([0, 90, 135, 180, 225, 270, 315])
        ax.tick_params(labelsize=font2d["tick"])
        ax.set_title('L$_S$ = {:d}\u00b0'.format(round(int(result['ls']))),
                     fontsize=font2d["title"], pad=12)

# ---------------------------------------------------------------------------
# Shared colorbar for 2-D modes
# ---------------------------------------------------------------------------
if last_mesh is not None:
    visible_axes = axes_flat[:nfiles].tolist()
    # Layout first so axis positions are finalised, leaving a right margin
    # for the colorbar.
    plt.tight_layout(rect=[0, 0, 0.88, 1], pad=0.3, w_pad=0.5, h_pad=0.5)
    # Read the exact bounding boxes of all panels and span them with a
    # manually-placed colorbar axes — this makes the colorbar height match
    # the panels exactly regardless of polar-plot padding or figure shape.
    bboxes = [ax.get_position() for ax in visible_axes]
    y0     = min(bb.y0 for bb in bboxes)
    y1     = max(bb.y1 for bb in bboxes)
    x1     = max(bb.x1 for bb in bboxes)
    cb_ax  = fig.add_axes([x1 + 0.04, y0, 0.025, y1 - y0])
    cb = fig.colorbar(last_mesh, cax=cb_ax)
    cb.ax.tick_params(labelsize=font2d["tick"])
    cb.set_label('Transition altitude (km)', fontsize=font2d["label"])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
if last_mesh is None:
    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5)

if args.solar:
    outfile = 'timescales_transition_solar.png'
elif args.plottransition:
    outfile = 'timescales_transition.png'
else:
    outfile = 'timescales_lat{}_lon{}.png'.format(int(args.lat), int(args.lon))

plt.savefig(outfile)
print('Wrote {}'.format(outfile))
