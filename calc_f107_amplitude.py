#!/usr/bin/env python3
"""
Calculate the amplitude of a signal at a specified period in F10.7 solar flux data.

Uses the same Lomb-Scargle pipeline as gitm_calc_amplitude.py: a periodogram
is computed over [period - width/2, period + width/2], the peak frequency is
located, and a sinusoid is fitted at that frequency to extract amplitude and phase.

Usage
-----
    calc_f107_amplitude.py f107.txt --period=27
    calc_f107_amplitude.py f107.txt --period=27 --start=2000-01-01 --end=2002-12-31
    calc_f107_amplitude.py f107.txt --period=27 --my=35 --start=0 --end=90
    calc_f107_amplitude.py f107.txt --period=27 --check --niter=2000

Arguments
---------
    file      Path to F10.7 data file (whitespace-delimited, # comments)
    --period  Centre of the period search window in days (required)
    --width   Full width of the search window in days (default: 2)
    --my      Mars year. If given, --start/--end are interpreted as Ls
              (areocentric solar longitude, degrees) instead of dates, and
              converted to UTC via ~/VCProjects/marstiming/src/marstiming.py.
    --start   Start date for subsetting, e.g. 2000-01-01 (or Ls if --my given)
    --end     End date for subsetting, e.g. 2002-12-31 (or Ls if --my given)
    --check   Compute p-value via phase randomization (permutation test)
    --niter   Number of permutations for --check (default: 1000)
    --output  Output text file (default: amplitude_f107_<period>d.txt)
    --plot    Plot the Lomb-Scargle periodogram and save as PNG
              (default: <output>_periodogram.png)
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lomb-Scargle pipeline (mirrors gitm_calc_amplitude.py)
# ---------------------------------------------------------------------------

def _fit_sinusoid(t, y, period_days):
    """Least-squares sinusoidal fit at a fixed period. Returns (amplitude, phase_deg)."""
    omega = 2.0 * np.pi / period_days
    A = np.column_stack([np.cos(omega * t),
                         np.sin(omega * t),
                         np.ones_like(t)])
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeff[0], coeff[1]
    return np.sqrt(a**2 + b**2), np.degrees(np.arctan2(b, a))


def _make_omegas(period_days, width_days, n=1000):
    """Angular-frequency grid for the search window."""
    half = width_days / 2.0
    p_min = max(period_days - half, 1e-6)
    p_max = period_days + half
    return np.linspace(2.0 * np.pi / p_max, 2.0 * np.pi / p_min, n)


def _ls_core(t, y, omegas):
    """LS periodogram on pre-cleaned arrays; returns (amplitude, phase_deg, detected_period)."""
    from scipy.signal import lombscargle
    pgram = lombscargle(t, y - np.mean(y), omegas, normalize=False)
    peak_omega = omegas[np.argmax(pgram)]
    detected_period = 2.0 * np.pi / peak_omega
    amplitude, phase_deg = _fit_sinusoid(t, y, detected_period)
    return amplitude, phase_deg, detected_period


def calc_amplitude_ls(times_days, values, period_days, width_days=2.0):
    """LS amplitude analysis. Returns (amplitude, phase_deg, detected_period)."""
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    omegas = _make_omegas(period_days, width_days)
    return _ls_core(t[mask], y[mask], omegas)


def calc_periodogram(times_days, values, period_days, width_days=2.0):
    """
    Lomb-Scargle periodogram over [period - width/2, period + width/2].
    Returns (periods_days, power, peak_period_days).
    """
    from scipy.signal import lombscargle

    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    t, y = t[mask], y[mask]

    omegas = _make_omegas(period_days, width_days)
    power = lombscargle(t, y - np.mean(y), omegas, normalize=False)
    periods = 2.0 * np.pi / omegas
    peak_period = periods[np.argmax(power)]
    return periods, power, peak_period


def plot_periodogram(periods, power, input_period, detected_period, output_path):
    """Plot power vs. period, mark the peak, and save as PNG. Returns the plot path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = os.path.splitext(output_path)[0] + "_periodogram.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(periods, power, color="#2a78d6", linewidth=2)
    ax.axvline(detected_period, color="#898781", linestyle="--", linewidth=1)
    peak_power = power[np.argmin(np.abs(periods - detected_period))]
    ax.annotate(f"peak: {detected_period:.3f} d",
                xy=(detected_period, peak_power),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color="#0b0b0b", va="center")

    ax.set_xlabel("Period (days)", color="#52514e")
    ax.set_ylabel("Lomb-Scargle power", color="#52514e")
    ax.set_title(f"F10.7 periodogram around {input_period:g} d", color="#0b0b0b")

    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def calc_pvalue(times_days, values, period_days, width_days=2.0, n_iter=1000):
    """
    Phase randomization p-value: fraction of permuted amplitudes >= observed.
    Returns (p_value, obs_amplitude, null_amplitudes).
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

    return float(np.sum(null_amps >= obs_amp)) / n_iter, obs_amp, null_amps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate F10.7 signal amplitude at a given period."
    )
    parser.add_argument("file", help="F10.7 data file (whitespace-delimited, # comments)")
    parser.add_argument("--period", type=float, default=None,
                        help="Centre of the period search window in days (required)")
    parser.add_argument("--width", type=float, default=2.0,
                        help="Full width of the period search window in days (default: 2)")
    parser.add_argument("--my", type=int, default=None,
                        help="Mars year. If given, --start/--end are interpreted as Ls "
                             "(areocentric solar longitude, degrees) instead of dates.")
    parser.add_argument("--start", default=None,
                        help="Start date for subsetting, e.g. 2000-01-01 "
                             "(or Ls in degrees if --my is given)")
    parser.add_argument("--end", default=None,
                        help="End date for subsetting, e.g. 2002-12-31 "
                             "(or Ls in degrees if --my is given)")
    parser.add_argument("--check", action="store_true",
                        help="Compute p-value via phase randomization (permutation test)")
    parser.add_argument("--niter", type=int, default=1000,
                        help="Number of permutations for --check (default: 1000)")
    parser.add_argument("--output", default=None,
                        help="Output text file (default: amplitude_f107_<period>d.txt)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the Lomb-Scargle periodogram and save as PNG "
                             "(default: <output>_periodogram.png)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"Error: file not found: {args.file}")
    if args.period is None:
        sys.exit("Error: --period is required.")
    if args.period <= 0:
        sys.exit("Error: --period must be positive.")

    # ── Convert Ls-based --start/--end to dates if --my is given ─────────
    if args.my is not None:
        marstiming_dir = os.path.expanduser("~/VCProjects/marstiming/src")
        if marstiming_dir not in sys.path:
            sys.path.insert(0, marstiming_dir)
        try:
            from marstiming import getUTCfromLS
        except ImportError as e:
            sys.exit(f"Error: could not import marstiming ({e}). Check that "
                     f"{marstiming_dir} exists and its dependencies "
                     "(astropy, astroquery) are installed.")

        start_ls = float(args.start) if args.start is not None else None
        end_ls   = float(args.end)   if args.end   is not None else None

        if start_ls is not None:
            args.start = getUTCfromLS(args.my, start_ls).strftime("%Y-%m-%d %H:%M:%S")
        if end_ls is not None:
            # If the end Ls wraps past 360 relative to the start, it falls in the next Mars year.
            end_my = args.my + 1 if (start_ls is not None and end_ls < start_ls) else args.my
            args.end = getUTCfromLS(end_my, end_ls).strftime("%Y-%m-%d %H:%M:%S")

        print(f"Mars Year {args.my}: Ls "
              f"[{start_ls if start_ls is not None else '—'} – "
              f"{end_ls if end_ls is not None else '—'}]  ->  "
              f"{args.start or '—'} – {args.end or '—'} (UTC)")

    # ── Load F10.7 data ───────────────────────────────────────────────────
    df = pd.read_csv(
        args.file,
        comment="#",
        delim_whitespace=True,
        names=["date", "time", "value", "qualifier", "description"],
        parse_dates=[[0, 1]],
    )
    df.rename(columns={"date_time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["value"]    = pd.to_numeric(df["value"], errors="coerce")

    # ── Date subset ───────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)
    if args.start:
        mask &= df["datetime"] >= pd.to_datetime(args.start)
    if args.end:
        mask &= df["datetime"] <= pd.to_datetime(args.end)
    df = df[mask].reset_index(drop=True)

    if len(df) < 3:
        sys.exit(f"Error: date subset leaves only {len(df)} rows — too few to fit.")

    print(f"Loaded {len(df)} data points  "
          f"({df['datetime'].iloc[0].date()} – {df['datetime'].iloc[-1].date()})")

    # Convert datetime to elapsed days from first point
    t0 = df["datetime"].iloc[0]
    times_days = (df["datetime"] - t0).dt.total_seconds().to_numpy() / 86400.0
    values     = df["value"].to_numpy()

    # ── Amplitude ─────────────────────────────────────────────────────────
    amp, phase, det_period = calc_amplitude_ls(times_days, values, args.period, args.width)

    # ── Output filename (needed before --plot, which derives its name from it) ──
    if args.output is None:
        args.output = f"amplitude_f107_{args.period:.0f}d.txt"

    # ── Periodogram plot ──────────────────────────────────────────────────
    if args.plot:
        periods, power, _ = calc_periodogram(times_days, values, args.period, args.width)
        plot_path = plot_periodogram(periods, power, args.period, det_period, args.output)
        print(f"Periodogram plot written to: {plot_path}")

    # ── p-value ───────────────────────────────────────────────────────────
    pvalue = np.nan
    if args.check:
        print(f"Running phase randomization ({args.niter} iterations) …", flush=True)
        pvalue, _, _ = calc_pvalue(times_days, values, args.period, args.width, args.niter)

    # ── Output ────────────────────────────────────────────────────────────
    date_range_str = ""
    if args.start or args.end:
        lo = args.start or "—"
        hi = args.end   or "—"
        date_range_str = f"  [{lo} – {hi}]"
    if args.my is not None:
        date_range_str += f"  (MY{args.my})"

    header_lines = [
        f"# F10.7 amplitude analysis (Lomb-Scargle)",
        f"# File      : {args.file}",
        f"# Period    : {args.period} days  (search window: ±{args.width/2} days)",
        f"# Time span : {df['datetime'].iloc[0].date()} – {df['datetime'].iloc[-1].date()}"
        f"  ({len(df)} points){date_range_str}",
        *([ f"# p-value   : phase randomization, n_iter={args.niter}" ] if args.check else []),
        f"#",
        f"# detected_period(days)  amplitude  phase(deg)"
        + ("  p_value" if args.check else ""),
    ]

    row = f"{det_period:10.4f}  {amp:14.6e}  {phase:10.3f}"
    if args.check:
        row += f"  {pvalue:8.4f}"

    with open(args.output, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        fh.write(row + "\n")

    print(f"Results written to: {args.output}")

    # Stdout summary
    print(f"  Period (input)   : {args.period} days  (window ±{args.width/2} days)")
    print(f"  Period (detected): {det_period:.4f} days")
    print(f"  Amplitude        : {amp:.4e} sfu")
    print(f"  Phase            : {phase:.2f} deg")
    if args.check:
        print(f"  p-value          : {pvalue:.4f}  (n={args.niter})")


if __name__ == "__main__":
    main()
