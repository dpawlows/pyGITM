#!/usr/bin/env python3
"""
Plot regression slope vs altitude for multiple cases.

Usage:
    plot_regress_vs_alt.py case1_regress.txt case2_regress.txt ...

One line per case, single-column figure.
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from gitm_routines import get_units

plt.rcParams.update({'lines.linewidth': 1.0})

cm = 1 / 2.54
COL_WIDTH_CM = {"single": 8, "double": 16}
BASE_FONT    = {"label": 9, "tick": 8, "legend": 8, "text": 8}

LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

VAR_LABELS = {
    "CO2":         r"[CO$_2$]",
    "O":           r"[O]",
    "Rho":         r"$\rho$",
    "Temperature": r"T",
    "Vneast":      r"u",
    "Vnnorth":     r"v",
    "Vnup":        r"w",
    "e":           r"[e-]",
}


def read_regress(filepath):
    """
    Read a regression text file produced by plot_phase1_solarvariability.py.
    Returns (varname, mode, alts, slopes, std_errs).
    """
    varname = ""
    mode = ""
    alts = []
    slopes = []
    std_errs = []
    with open(filepath) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("# Regression results:"):
                varname = stripped.split(":")[-1].strip().split(" vs ")[0].strip()
            elif stripped.startswith("# mode="):
                mode = stripped.split("=", 1)[-1].strip()
            elif stripped.startswith("#") or not stripped:
                continue
            else:
                parts = stripped.split()
                if len(parts) >= 6:
                    alts.append(float(parts[0]))
                    slopes.append(float(parts[1]))
                    std_errs.append(float(parts[5]))
    return varname, mode, np.array(alts), np.array(slopes), np.array(std_errs)


def main():
    parser = argparse.ArgumentParser(
        description="Plot regression slope vs altitude for multiple cases"
    )
    parser.add_argument("files", nargs="+", help="Regression text file(s)")
    parser.add_argument("-show", action="store_true", help="Display interactively")
    args = parser.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            sys.exit(f"Error: file not found: {f}")

    font     = BASE_FONT
    fig, ax  = plt.subplots(figsize=(COL_WIDTH_CM["single"] * cm, 8 * cm),
                            constrained_layout=True)

    varname_out = ""
    mode_out    = ""

    for i, fpath in enumerate(args.files):
        varname, mode, alts, slopes, std_errs = read_regress(fpath)
        if i == 0:
            varname_out = varname
            mode_out    = mode

        label = os.path.basename(fpath).replace("_regress.txt", "")
        ax.plot(slopes, alts, linestyle=LINESTYLES[i % len(LINESTYLES)],
                color='k', label=label)
        ax.errorbar(slopes, alts, xerr=2 * std_errs,
                    fmt='none', color='k',
                    capsize=2, capthick=0.8, elinewidth=0.8)

    varlabel = VAR_LABELS.get(varname_out, varname_out)
    units    = get_units(varname_out)
    slope_units = f"{units}/sfu" if units else "1/sfu"
    ax.set_xlabel(rf"$\Delta${varlabel}/$\Delta$F10.7 ({slope_units})",
                  fontsize=font["label"])
    ax.set_ylabel("Altitude (km)", fontsize=font["label"])
    ax.tick_params(labelsize=font["tick"])
    ax.legend(fontsize=font["legend"], frameon=False)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, linestyle=':')

    cases   = "_".join(os.path.basename(f).replace("_regress.txt", "")
                       for f in args.files)
    outfile = f"slope_vs_alt_{cases}_{mode_out}.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
