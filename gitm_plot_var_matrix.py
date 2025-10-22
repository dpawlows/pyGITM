#!/usr/bin/env python3
"""Plot altitude profiles for all satellite variables from index 3 onward.

This script is inspired by :mod:`gitm_plot_satellite.py`, but is simplified to
accept a single GITM satellite output file as input.  The script loads all
variables beginning with index 3 (the first two indices are longitude and
latitude, index 2 is altitude) and produces a heatmap-style plot where the
x-axis corresponds to the variable index, the y-axis corresponds to altitude,
and the color encodes the value of each variable.

The intent is to provide a quick-look comparison of how the different traced
quantities vary with altitude for a single time step. Profiles can optionally
be normalized independently so that values lie between 0 and 1, making it
easier to compare relative altitude structure across variables with different
units or magnitudes.
"""

import argparse
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from gitm_routines import read_gitm_header, read_gitm_one_file


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot a heatmap of all variables (index >=3) in a GITM satellite file "
            "as a function of altitude and variable index."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the GITM satellite binary file to plot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="gitm_var_matrix.png",
        help=(
            "Filename for the generated plot. The image will be saved in PNG "
            "format. (default: gitm_var_matrix.png)"
        ),
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Normalize each variable profile to the [0, 1] range before plotting.",
    )
    return parser.parse_args()


def ensure_file_exists(path: str) -> None:
    """Ensure the requested file exists before attempting to read it."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")


def load_variable_indices(header: dict) -> Sequence[int]:
    """Return the indices of variables to load (starting at index 3)."""

    nvars = header.get("nVars", 0)
    if nvars <= 3:
        raise ValueError(
            "The supplied file does not contain variables beyond index 2."
        )
    return list(range(3, nvars))


def extract_altitude(data: dict) -> np.ndarray:
    """Extract the altitude array in kilometers from the data dictionary."""

    altitude_key = 2
    if altitude_key not in data:
        raise KeyError(
            "Altitude (variable index 2) was not loaded. Ensure it is included "
            "in the variable list."
        )
    altitude = data[altitude_key][0, 0, :] / 1000.0
    return altitude


def assemble_variable_matrix(data: dict, variable_indices: Sequence[int]) -> np.ndarray:
    """Create a 2D array of variable values sampled along altitude."""

    profiles = []
    for index in variable_indices:
        if index not in data:
            raise KeyError(
                f"Requested variable index {index} is missing from the loaded data."
            )
        profiles.append(data[index][0, 0, :])

    matrix = np.vstack(profiles)
    return matrix


def normalize_variable_matrix(value_matrix: np.ndarray) -> np.ndarray:
    """Normalize values for each variable profile to the [0, 1] range."""

    value_matrix = value_matrix.astype(float)
    mins = value_matrix.min(axis=1, keepdims=True)
    maxs = value_matrix.max(axis=1, keepdims=True)
    ranges = maxs - mins

    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(ranges > 0, (value_matrix - mins) / ranges, 0.0)

    return normalized


def plot_variable_matrix(
    altitude_km: np.ndarray,
    variable_indices: Sequence[int],
    variable_names: Sequence[str],
    value_matrix: np.ndarray,
    output_file: str,
    normalized: bool,
) -> None:
    """Generate and save the altitude-versus-variable heatmap plot."""

    # Flip axes for plotting convenience: altitude along Y, variables along X.
    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [
        variable_indices[0] - 0.5,
        variable_indices[-1] + 0.5,
        altitude_km[0],
        altitude_km[-1],
    ]
    im = ax.imshow(
        value_matrix.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )

    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("GITM Variable Profiles")

    ax.set_xticks(variable_indices)
    tick_labels = [variable_names[i].strip() for i in variable_indices]
    ax.set_xticklabels(tick_labels, rotation=90)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized Value" if normalized else "Value")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main() -> None:
    """Entry point for the script."""

    args = parse_arguments()
    ensure_file_exists(args.input_file)

    header = read_gitm_header([args.input_file])
    variable_indices = load_variable_indices(header)

    # Load altitude (index 2) plus all variables from index 3 onward.
    indices_to_read = [2] + list(variable_indices)
    data = read_gitm_one_file(args.input_file, indices_to_read)

    altitude_km = extract_altitude(data)
    value_matrix = assemble_variable_matrix(data, variable_indices)
    if args.normalize:
        value_matrix = normalize_variable_matrix(value_matrix)

    plot_variable_matrix(
        altitude_km,
        variable_indices,
        header["vars"],
        value_matrix,
        args.output,
        args.normalize,
    )

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
