#!/usr/bin/env python3
"""Create binomial confidence-interval deviations for stacked-area boundaries."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from scripts.utility.basic_utility import get_filepath, get_non_existing_path


def wilson_interval(successes: int, trials: int, z_score: float) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("The total number of trees must be positive.")
    proportion = successes / trials
    z_squared = z_score ** 2
    denominator = 1 + z_squared / trials
    enumerator = z_score * math.sqrt(
                    proportion * (1 - proportion) / trials + z_squared / (4 * trials**2)
    )
    half_width = enumerator / denominator
    centre = (proportion + z_squared / (2 * trials)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def wald_interval(successes: int, trials: int, z_score: float) -> tuple[float, float]:
    """Return the bounded Wald interval for a binomial proportion."""
    if trials < 1:
        raise ValueError("The total number of trees must be positive.")
    proportion = successes / trials
    half_width = z_score * math.sqrt(proportion * (1 - proportion) / trials)
    return max(0.0, proportion - half_width), min(1.0, proportion + half_width)


def build_intervals(input_path: Path, output_path: Path, z_score: float, method: str) -> None:
    # Parse the input file
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or reader.fieldnames[0] != "proband_number":
            raise ValueError("The first CSV column must be named 'proband_number'.")
        metadata_columns = {"proband_number", "row_count", "clade_count"}
        bin_names = [name for name in reader.fieldnames if name not in metadata_columns]
        if not bin_names:
            raise ValueError("The CSV must contain at least one bin-count column.")
        output_columns = ["proband_number"]
        for bin_name in bin_names:
            output_columns.extend(
                (
                    f"{bin_name}_lower_deviation",
                    f"{bin_name}_upper_deviation",
                )
            )
        # Build the resulting file
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                try:
                    values = [int(row[bin_name]) for bin_name in bin_names]
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid bin count on CSV line {line_number}.") from error
                if any(value < 0 for value in values):
                    raise ValueError(f"Negative bin count on CSV line {line_number}.")
                category_total = sum(values)
                if "clade_count" in reader.fieldnames:
                    try:
                        total_clades = int(row["clade_count"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"Invalid clade_count on CSV line {line_number}."
                        ) from error
                    if category_total != total_clades:
                        raise ValueError(
                            f"Bin counts do not sum to clade_count on CSV line {line_number}."
                        )
                else:
                    total_clades = category_total
                output_row: dict[str, str | float] = {"proband_number": row["proband_number"]}
                cumulative_count = 0
                for bin_name, value in zip(bin_names, values):
                    cumulative_count += value
                    if method == "wilson":
                        lower, upper = wilson_interval(cumulative_count, total_clades, z_score)
                    else:
                        lower, upper = wald_interval(cumulative_count, total_clades, z_score)
                    output_row.update(
                        {
                            f"{bin_name}_lower_deviation": max(
                                0.0, cumulative_count - lower * total_clades
                            ),
                            f"{bin_name}_upper_deviation": max(
                                0.0, upper * total_clades - cumulative_count
                            ),
                        }
                    )
                writer.writerow(output_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create binomial confidence intervals for cumulative stacked-area boundaries."
    )
    parser.add_argument(
        "input_csv", nargs="?", type=Path, help="CSV with proband_number followed by bin counts"
    )
    parser.add_argument(
        "output_csv", nargs="?", type=Path, help="Path for the confidence-interval CSV"
    )
    parser.add_argument(
        "--z-score",
        type=float,
        default=1.959963984540054,
        help="Normal critical value (default: 1.959963984540054 for 95%% intervals)",
    )
    parser.add_argument(
        "--method",
        choices=("wilson", "wald"),
        help="Confidence-interval method; prompted when omitted",
    )
    arguments = parser.parse_args()
    if arguments.input_csv is None and arguments.output_csv is None:
        arguments.input_csv = Path(get_filepath("Input CSV filepath:"))
        current_directory = os.getcwd()
        try:
            os.chdir(arguments.input_csv.parent)
            output_path = Path(get_non_existing_path("Output CSV filename:"))
            if not output_path.suffix:
                output_path = output_path.with_suffix(".csv")
            arguments.output_csv = output_path.resolve()
        finally:
            os.chdir(current_directory)
    elif arguments.input_csv is None or arguments.output_csv is None:
        parser.error("Provide both input and output paths, or provide neither to be prompted.")
    if arguments.method is None:
        arguments.method = input(
            "Confidence-interval method [wald/wilson] (default: wilson): "
        ).strip().lower() or "wilson"
        if arguments.method not in ("wald", "wilson"):
            parser.error("Method must be 'wald' or 'wilson'.")
    if arguments.z_score <= 0:
        parser.error("--z-score must be positive")
    try:
        build_intervals(arguments.input_csv, arguments.output_csv, arguments.z_score, arguments.method)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
