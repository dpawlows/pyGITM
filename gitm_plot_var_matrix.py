#!/usr/bin/env python3
"""Plot normalised GITM variables in a matrix of subplots.

The script reads one or more 3D GITM binary files, averages the selected
variables over longitude and latitude, and then normalises each variable's
altitude-time profile to the ``[0, 1]`` range before plotting the results.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from gitm_routines import read_gitm_header, read_gitm_one_file


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Plot normalised GITM variables from 3D output files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="GITM 3D binary files (e.g., 3DALL*.bin) to include in the plot",
    )
    parser.add_argument(
        "-var",
        "--var",
        dest="variables",
        required=True,
        help="Comma separated list of variable indices to include in the plot",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for the generated figure.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save the figure to the specified path instead of displaying it.",
    )
    return parser.parse_args(argv)


def normalise(data: np.ndarray) -> np.ndarray:
    """Normalise ``data`` to the [0, 1] range.

    ``NaN`` values are preserved. When ``data`` is constant (i.e., ``min == max``)
    the function returns an array of zeros to avoid division by zero while still
    producing a meaningful plot.
    """

    normalised = np.array(data, copy=True, dtype=float)
    valid = np.isfinite(normalised)

    if not np.any(valid):
        # All values are NaN/inf – return zeros to keep imshow happy.
        return np.zeros_like(normalised, dtype=float)

    vmin = np.min(normalised[valid])
    vmax = np.max(normalised[valid])

    if np.isclose(vmin, vmax):
        normalised[valid] = 0.0
        return normalised

    normalised[valid] = (normalised[valid] - vmin) / (vmax - vmin)
    return normalised


def build_variable_profiles(
    files: Iterable[str],
    variable_indices: Sequence[int],
) -> tuple[List[datetime], np.ndarray, Dict[int, List[np.ndarray]]]:
    """Read ``files`` and create averaged altitude profiles per variable."""

    required_vars = {0, 1, 2}
    required_vars.update(variable_indices)
    sorted_vars = sorted(required_vars)

    times: List[datetime] = []
    altitudes_km: np.ndarray | None = None
    profiles: Dict[int, List[np.ndarray]] = {idx: [] for idx in variable_indices}

    for path in files:
        data = read_gitm_one_file(path, sorted_vars)
        times.append(data["time"])

        if altitudes_km is None:
            altitudes_km = data[2][0, 0] / 1000.0

        for var_idx in variable_indices:
            field = data[var_idx]
            # Average over longitude/latitude dimensions to obtain an altitude profile.
            profile = np.nanmean(field, axis=(0, 1))
            profiles[var_idx].append(profile)

    assert altitudes_km is not None, "No altitude grid could be derived from files"
    return times, altitudes_km, profiles


def plot_variable_matrix(
    times: Sequence[datetime],
    altitudes_km: np.ndarray,
    profiles: Dict[int, List[np.ndarray]],
    variable_indices: Sequence[int],
    variable_names: Sequence[str],
    save_path: str | None = None,
    title: str | None = None,
) -> None:
    """Create the matrix plot for the provided variables."""

    num_vars = len(variable_indices)
    if num_vars == 0:
        raise ValueError("At least one variable must be provided for plotting.")

    num_cols = math.ceil(math.sqrt(num_vars))
    num_rows = math.ceil(num_vars / num_cols)

    figure, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3.5 * num_rows))
    axes_array = np.atleast_1d(axes).ravel()

    time_numbers = mdates.date2num(times)

    for ax, (var_idx, name) in zip(axes_array, zip(variable_indices, variable_names)):
        matrix = np.vstack(profiles[var_idx])
        norm_matrix = normalise(matrix)

        im = ax.imshow(
            norm_matrix.T,
            aspect="auto",
            origin="lower",
            extent=[time_numbers[0], time_numbers[-1], altitudes_km[0], altitudes_km[-1]],
            cmap="viridis",
        )
        ax.set_title(f"{name} (normalised)")
        ax.set_ylabel("Altitude [km]")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        figure.colorbar(im, ax=ax, label="Normalised value", fraction=0.046, pad=0.04)

    # Hide any unused axes if the grid is larger than required.
    for ax in axes_array[num_vars:]:
        ax.set_visible(False)

    figure.autofmt_xdate()
    if title:
        figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96) if title else None)

    if save_path:
        figure.savefig(save_path, dpi=300)
    else:
        plt.show()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    header = read_gitm_header([args.files[0]])
    available_vars = header["vars"]
    raw_indices = [int(idx) for idx in args.variables.split(",") if idx]
    seen: set[int] = set()
    variable_indices: List[int] = []
    for idx in raw_indices:
        if idx not in seen:
            variable_indices.append(idx)
            seen.add(idx)

    # Validate indices and collect names in the same order as requested.
    variable_names: List[str] = []
    for idx in variable_indices:
        if idx < 0 or idx >= len(available_vars):
            raise IndexError(
                f"Variable index {idx} is outside the available range [0, {len(available_vars) - 1}]."
            )
        variable_names.append(available_vars[idx])

    times, altitudes_km, profiles = build_variable_profiles(args.files, variable_indices)

    plot_variable_matrix(
        times=times,
        altitudes_km=altitudes_km,
        profiles=profiles,
        variable_indices=variable_indices,
        variable_names=variable_names,
        save_path=args.save,
        title=args.title,
    )


if __name__ == "__main__":
    main()
