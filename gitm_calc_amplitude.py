#!/usr/bin/env python3
"""
Calculate the amplitude of a signal at a specified period from a GITM NetCDF file.

The amplitude is computed at each altitude by fitting:
    x(t) = a*cos(2π*t/P) + b*sin(2π*t/P) + c

where P is the period in days, and amplitude = sqrt(a² + b²).

Only point-based modes (e.g. subsolar, antisolar) are supported.

Usage
-----
    gitm_calc_amplitude.py case1.nc --period=27 --mode=subsolar --var=Temperature

Arguments
---------
    file        Input NetCDF file (required)
    --period    Period in days (required). If --width is omitted, the amplitude
                is computed via a direct least-squares sinusoid fit at this
                period. If --width is given, a Lomb-Scargle periodogram is
                searched over [period - width/2, period + width/2] instead.
    --width     Full width of the Lomb-Scargle search window in days. Omit to
                skip the periodogram search and fit directly at --period.
    --var       Variable name, e.g. Temperature (required)
    --mode      Point mode to use (default: subsolar)
    --alt       Altitude in km; nearest level is used. Omit to compute all altitudes.
    --check     Compute a p-value via phase randomization (permutation test).
    --niter     Number of permutations for --check (default: 1000).
    --output    Output filename for results (default: amplitude_<var>_<period>d.txt)
    --list      List available point modes, variables, and altitudes, then exit
"""

import sys
import os
import argparse

import numpy as np
import netCDF4 as nc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_char_var(var):
    """Convert a netCDF4 string/char variable to a list of stripped strings."""
    data = var[:]
    if data.dtype.kind in ("S", "U"):
        return [str(s).strip() for s in data]
    if data.ndim == 2:
        return ["".join(row.astype(str)).strip() for row in data]
    return [str(s).strip() for s in data]


def get_point_vars(ds, point_modes):
    """Return sorted list of variable base names that have a _point variant."""
    coord_names = {"time", "altitude", "latitude", "mode_lat", "mode_point",
                   "Ls", "nfiles", "average"}
    base_vars = set()
    for name in ds.variables:
        if name in coord_names:
            continue
        if name.endswith("_point"):
            base_vars.add(name[:-6])
    return sorted(base_vars)


def _fit_sinusoid(t, y, period_days):
    """Least-squares sinusoidal fit at a fixed period. Returns (amplitude, phase_deg)."""
    omega = 2.0 * np.pi / period_days
    A = np.column_stack([np.cos(omega * t),
                         np.sin(omega * t),
                         np.ones_like(t)])
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeff[0], coeff[1]
    return np.sqrt(a**2 + b**2), np.degrees(np.arctan2(b, a))


def _ls_core(t, y, omegas):
    """LS periodogram on pre-cleaned arrays; returns (amplitude, phase_deg, detected_period)."""
    from scipy.signal import lombscargle
    pgram = lombscargle(t, y - np.mean(y), omegas, normalize=False)
    peak_omega = omegas[np.argmax(pgram)]
    detected_period = 2.0 * np.pi / peak_omega
    amplitude, phase_deg = _fit_sinusoid(t, y, detected_period)
    return amplitude, phase_deg, detected_period


def _make_omegas(period_days, width_days, n=1000):
    """Build the angular-frequency grid for a given period window."""
    half = width_days / 2.0
    p_min = max(period_days - half, 1e-6)
    p_max = period_days + half
    return np.linspace(2.0 * np.pi / p_max, 2.0 * np.pi / p_min, n)


def calc_amplitude_ls(times_days, values, period_days, width_days=2.0):
    """
    Compute a Lomb-Scargle periodogram over [period - width/2, period + width/2],
    find the peak, then fit a sinusoid at that peak period.

    Returns (amplitude, phase_deg, detected_period). NaNs on insufficient data.
    """
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    omegas = _make_omegas(period_days, width_days)
    return _ls_core(t[mask], y[mask], omegas)


def calc_amplitude_fit(times_days, values, period_days):
    """
    Compute the amplitude via a direct least-squares sinusoid fit at a fixed
    period (no periodogram search).

    Returns (amplitude, phase_deg). NaNs on insufficient data.
    """
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    return _fit_sinusoid(t[mask], y[mask], period_days)


def calc_pvalue_fit(times_days, values, period_days, n_iter=1000):
    """
    Estimate significance of the direct sinusoid-fit amplitude via phase
    randomization (permutation test).

    Returns (p_value, observed_amplitude, null_amplitudes).
    """
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.array([])

    t_clean = t[mask]
    y_clean = y[mask]

    obs_amp, _ = _fit_sinusoid(t_clean, y_clean, period_days)

    rng = np.random.default_rng()
    null_amps = np.empty(n_iter)
    for i in range(n_iter):
        amp, _ = _fit_sinusoid(t_clean, rng.permutation(y_clean), period_days)
        null_amps[i] = amp

    p_value = float(np.sum(null_amps >= obs_amp)) / n_iter
    return p_value, obs_amp, null_amps


def calc_pvalue(times_days, values, period_days, width_days=2.0, n_iter=1000):
    """
    Estimate significance of the LS amplitude via phase randomization.

    The data values are randomly permuted n_iter times (keeping the time
    coordinates fixed), the LS amplitude is recomputed for each surrogate,
    and the p-value is the fraction of surrogate amplitudes >= the observed
    amplitude.

    Returns (p_value, observed_amplitude, null_amplitudes).
    """
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.array([])

    t_clean = t[mask]
    y_clean = y[mask]
    omegas  = _make_omegas(period_days, width_days)

    obs_amp, _, _ = _ls_core(t_clean, y_clean, omegas)

    rng = np.random.default_rng()
    null_amps = np.empty(n_iter)
    for i in range(n_iter):
        amp, _, _ = _ls_core(t_clean, rng.permutation(y_clean), omegas)
        null_amps[i] = amp

    p_value = float(np.sum(null_amps >= obs_amp)) / n_iter
    return p_value, obs_amp, null_amps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate signal amplitude at a given period from a GITM NetCDF file."
    )
    parser.add_argument("file", help="Input NetCDF (.nc) file")
    parser.add_argument("--period", type=float, default=None,
                        help="Period in days (required unless --list)")
    parser.add_argument("--var", default=None,
                        help="Variable name (e.g. Temperature)")
    parser.add_argument("--mode", default="subsolar",
                        help="Point mode to use (default: subsolar)")
    parser.add_argument("--width", type=float, default=None,
                        help="Full width of the Lomb-Scargle search window in days. "
                             "If omitted, the amplitude is computed via a direct "
                             "least-squares sinusoid fit at --period (no search).")
    parser.add_argument("--alt", type=float, default=None,
                        help="Altitude in km; nearest available level is used. "
                             "Omit to compute all altitudes.")
    parser.add_argument("--output", default=None,
                        help="Output text file (default: amplitude_<var>_<period>d.txt)")
    parser.add_argument("--lsmin", type=float, default=None,
                        help="Minimum solar longitude (Ls, degrees) for subsetting the data")
    parser.add_argument("--lsmax", type=float, default=None,
                        help="Maximum solar longitude (Ls, degrees) for subsetting the data")
    parser.add_argument("--check", action="store_true",
                        help="Compute p-value via phase randomization (permutation test)")
    parser.add_argument("--niter", type=int, default=1000,
                        help="Number of permutations for --check (default: 1000)")
    parser.add_argument("--list", action="store_true",
                        help="List available point modes and variables, then exit")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found: {args.file}")

    ds = nc.Dataset(args.file)

    # Read coordinate metadata
    try:
        point_modes = decode_char_var(ds.variables["mode_point"])
    except KeyError:
        ds.close()
        sys.exit("Error: file has no 'mode_point' dimension — only point-mode files are supported.")

    point_vars = get_point_vars(ds, point_modes)
    alt_values = np.array(ds.variables["altitude"][:])

    if args.list:
        print("Available point modes (--mode):")
        print("  " + ", ".join(point_modes))
        print("\nAvailable variables (--var):")
        print("  " + ", ".join(point_vars))
        print(f"\nAltitudes (km): {alt_values[0]:.1f} – {alt_values[-1]:.1f} "
              f"({len(alt_values)} levels)")
        ds.close()
        return

    # Validate required arguments
    if args.period is None:
        ds.close()
        sys.exit("Error: --period is required (use --list to see available variables/modes).")
    if args.var is None:
        ds.close()
        sys.exit("Error: --var is required (use --list to see available variables/modes).")
    if args.period <= 0:
        ds.close()
        sys.exit("Error: --period must be positive.")

    # Validate mode
    if args.mode not in point_modes:
        ds.close()
        sys.exit(
            f"Error: mode '{args.mode}' not found in point modes.\n"
            f"Available: {', '.join(point_modes)}\n"
            "Only point modes are supported (use --list to see all)."
        )

    # Validate variable
    nc_varname = args.var + "_point"
    if nc_varname not in ds.variables:
        ds.close()
        sys.exit(
            f"Error: variable '{args.var}' not found as a point variable.\n"
            f"Available: {', '.join(point_vars)}\n"
            "(Use --list to see all.)"
        )

    mode_idx = point_modes.index(args.mode)

    # Read time and convert to days from the first time step
    time_var = ds.variables["time"]
    raw_times = nc.num2date(time_var[:], units=time_var.units,
                            calendar=getattr(time_var, "calendar", "standard"))
    import datetime as _dt
    t0 = _dt.datetime(raw_times[0].year, raw_times[0].month, raw_times[0].day,
                      raw_times[0].hour, raw_times[0].minute, raw_times[0].second)
    times_days = np.array([
        (_dt.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) - t0
         ).total_seconds() / 86400.0
        for t in raw_times
    ])

    # Read Ls if subsetting is requested
    ls_values = None
    if args.lsmin is not None or args.lsmax is not None:
        if "Ls" in ds.variables:
            ls_values = np.array(ds.variables["Ls"][:])
        else:
            ds.close()
            sys.exit("Error: --lsmin/--lsmax requested but 'Ls' variable not found in file.")

    # Read variable: shape (time, altitude, mode_point)
    data = np.array(ds.variables[nc_varname][:, :, mode_idx])   # (time, alt)
    ds.close()

    # Apply Ls subset
    if ls_values is not None:
        ls_mask = np.ones(len(times_days), dtype=bool)
        if args.lsmin is not None:
            ls_mask &= ls_values >= args.lsmin
        if args.lsmax is not None:
            ls_mask &= ls_values <= args.lsmax
        n_kept = ls_mask.sum()
        if n_kept < 3:
            sys.exit(f"Error: Ls subset [{args.lsmin}, {args.lsmax}] leaves only "
                     f"{n_kept} time steps — too few to fit.")
        times_days = times_days[ls_mask]
        data       = data[ls_mask, :]
        print(f"Ls subset [{args.lsmin}°, {args.lsmax}°]: {n_kept} of "
              f"{ls_mask.size} time steps retained.")

    # Altitude selection
    if args.alt is not None:
        nearest_idx = int(np.argmin(np.abs(alt_values - args.alt)))
        nearest_alt = alt_values[nearest_idx]
        if abs(nearest_alt - args.alt) > 0:
            print(f"Note: requested altitude {args.alt} km; using nearest level {nearest_alt:.3f} km")
        alt_indices = [nearest_idx]
    else:
        alt_indices = list(range(len(alt_values)))

    amplitudes       = np.full(len(alt_values), np.nan)
    phases           = np.full(len(alt_values), np.nan)
    detected_periods = np.full(len(alt_values), np.nan)
    pvalues          = np.full(len(alt_values), np.nan)

    use_ls = args.width is not None

    n_alts = len(alt_indices)
    for i, ia in enumerate(alt_indices):
        if use_ls:
            amp, ph, det_p = calc_amplitude_ls(times_days, data[:, ia],
                                               args.period, args.width)
        else:
            amp, ph = calc_amplitude_fit(times_days, data[:, ia], args.period)
            det_p = args.period
        amplitudes[ia]       = amp
        phases[ia]           = ph
        detected_periods[ia] = det_p

        if args.check:
            print(f"  Phase randomization: altitude {alt_values[ia]:.1f} km "
                  f"({i + 1}/{n_alts}) …", end="\r", flush=True)
            if use_ls:
                pv, _, _ = calc_pvalue(times_days, data[:, ia],
                                       args.period, args.width, args.niter)
            else:
                pv, _, _ = calc_pvalue_fit(times_days, data[:, ia],
                                           args.period, args.niter)
            pvalues[ia] = pv

    if args.check:
        print()  # newline after the progress line

    # Output filename
    if args.output is None:
        alt_tag = f"_{args.alt:.0f}km" if args.alt is not None else ""
        args.output = f"amplitude_{args.var}_{args.period:.0f}d{alt_tag}.txt"

    col_header = ("# Columns: altitude(km)  detected_period(days)  amplitude  phase(deg)"
                  + ("  p_value" if args.check else ""))
    ls_range_str = ""
    if args.lsmin is not None or args.lsmax is not None:
        lo = f"{args.lsmin}°" if args.lsmin is not None else "—"
        hi = f"{args.lsmax}°" if args.lsmax is not None else "—"
        ls_range_str = f"  [Ls {lo} – {hi}]"

    period_str = (f"{args.period} days  (search window: ±{args.width/2} days)" if use_ls
                  else f"{args.period} days  (fixed, direct least-squares fit)")
    header_lines = [
        f"# GITM amplitude analysis ({'Lomb-Scargle' if use_ls else 'least-squares fit'})",
        f"# File    : {args.file}",
        f"# Variable: {args.var}",
        f"# Mode    : {args.mode}",
        f"# Period  : {period_str}",
        f"# Time span: {times_days[0]:.2f} – {times_days[-1]:.2f} days "
        f"({len(times_days)} time steps){ls_range_str}",
        *([ f"# p-value : phase randomization, n_iter={args.niter}" ] if args.check else []),
        f"#",
        col_header,
    ]

    with open(args.output, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        for ia in alt_indices:
            row = (f"{alt_values[ia]:10.3f}  {detected_periods[ia]:10.4f}"
                   f"  {amplitudes[ia]:14.6e}  {phases[ia]:10.3f}")
            if args.check:
                row += f"  {pvalues[ia]:8.4f}"
            fh.write(row + "\n")

    print(f"Amplitude profile written to: {args.output}")

    # Compact stdout summary
    valid_mask = np.isfinite(amplitudes[alt_indices])
    valid_amps = amplitudes[alt_indices][valid_mask]
    if valid_amps.size:
        period_summary = (f"{args.period} days  (window ±{args.width/2} days)" if use_ls
                          else f"{args.period} days  (fixed)")
        if args.alt is not None:
            ia = alt_indices[0]
            print(f"  Period (input)   : {period_summary}")
            if use_ls:
                print(f"  Period (detected): {detected_periods[ia]:.4f} days")
            print(f"  Variable         : {args.var}, mode: {args.mode}")
            print(f"  Altitude         : {alt_values[ia]:.1f} km")
            print(f"  Amplitude        : {amplitudes[ia]:.4e}")
            print(f"  Phase            : {phases[ia]:.2f} deg")
            if args.check:
                print(f"  p-value          : {pvalues[ia]:.4f}  (n={args.niter})")
        else:
            peak_idx = np.nanargmax(amplitudes)
            print(f"  Period (input)   : {period_summary}")
            print(f"  Variable         : {args.var}, mode: {args.mode}")
            print(f"  Altitude range   : {alt_values[0]:.1f} – {alt_values[-1]:.1f} km")
            if use_ls:
                print(f"  Peak amplitude   : {amplitudes[peak_idx]:.4e}  at {alt_values[peak_idx]:.1f} km"
                      f"  (detected period {detected_periods[peak_idx]:.4f} days)")
            else:
                print(f"  Peak amplitude   : {amplitudes[peak_idx]:.4e}  at {alt_values[peak_idx]:.1f} km")
            if args.check:
                sig = np.sum(pvalues[alt_indices] < 0.05)
                print(f"  Significant (p<0.05): {sig}/{n_alts} altitude levels")


if __name__ == "__main__":
    main()
