#!/usr/bin/env python3
"""Create binomial confidence-interval deviations for stacked-area boundaries."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
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


def percentile(sorted_values: list[float], quantile: float) -> float:
    """Return an interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("At least one bootstrap replicate is required.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("The percentile must be between zero and one.")
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (position - lower_index) * (upper_value - lower_value)


def output_columns(bin_names: list[str]) -> list[str]:
    """Return the CSV columns used for interval deviations."""
    columns = ["proband_number"]
    for bin_name in bin_names:
        columns.extend(
            (
                f"{bin_name}_lower_deviation",
                f"{bin_name}_upper_deviation",
            )
        )
    return columns


def group_clades_by_definition(
    clades_by_proband: dict[str, list[list[int]]],
    bin_definition_path: Path,
    bin_names: list[str],
) -> dict[str, list[list[int]]]:
    """Group consecutive proband results according to aggregate bin counts."""
    with bin_definition_path.open(newline="", encoding="utf-8-sig") as definition_file:
        reader = csv.DictReader(definition_file)
        required_columns = {"proband_number", *bin_names}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "The bin-definition CSV must contain proband_number and every bin-count column."
            )
        bin_definitions: list[tuple[str, list[int], int | None]] = []
        seen_labels: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            label = row["proband_number"]
            if label in seen_labels:
                raise ValueError(f"Duplicate bin label on CSV line {line_number}.")
            try:
                counts = [int(row[bin_name]) for bin_name in bin_names]
                row_count = int(row["row_count"]) if "row_count" in reader.fieldnames else None
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid bin definition on CSV line {line_number}.") from error
            if any(count < 0 for count in counts) or row_count is not None and row_count < 1:
                raise ValueError(f"Invalid bin definition on CSV line {line_number}.")
            seen_labels.add(label)
            bin_definitions.append((label, counts, row_count))

    if not bin_definitions:
        raise ValueError("The bin-definition CSV must contain at least one data row.")
    try:
        raw_proband_groups = sorted(
            clades_by_proband.items(), key=lambda item: int(item[0])
        )
    except ValueError as error:
        raise ValueError("Proband numbers must be integers when using a bin-definition CSV.") from error

    group_index = 0
    clades_by_bin: dict[str, list[list[int]]] = {}
    for label, expected_counts, expected_row_count in bin_definitions:
        observed_counts = [0] * len(bin_names)
        clades: list[list[int]] = []
        while observed_counts != expected_counts:
            if group_index >= len(raw_proband_groups):
                raise ValueError(f"Bin '{label}' cannot be matched to the raw clade results.")
            _, proband_clades = raw_proband_groups[group_index]
            group_index += 1
            clades.extend(proband_clades)
            for index in range(len(bin_names)):
                observed_counts[index] += sum(
                    clade[index] for clade in proband_clades
                )
            if any(
                observed > expected
                for observed, expected in zip(observed_counts, expected_counts)
            ):
                raise ValueError(f"Bin '{label}' cannot be matched to the raw clade results.")
        if expected_row_count is not None and len(clades) != expected_row_count:
            raise ValueError(f"Bin '{label}' has an unexpected number of clade rows.")
        clades_by_bin[label] = clades

    if group_index != len(raw_proband_groups):
        raise ValueError("The bin-definition CSV does not cover all raw clade results.")
    return clades_by_bin


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
        fieldnames = output_columns(bin_names)
        # Build the resulting file
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
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


def build_cluster_bootstrap_intervals(
    input_path: Path,
    output_path: Path,
    bootstrap_replicates: int,
    seed: int | None,
    proband_bin_size: int | None,
    proband_bin_start: int | None,
    bin_definition_path: Path | None,
) -> None:
    """Bootstrap whole clades within each proband and write stacked-boundary intervals."""
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or reader.fieldnames[0] != "proband_number":
            raise ValueError("The first CSV column must be named 'proband_number'.")
        metadata_columns = {"proband_number", "row_count", "clade_count"}
        bin_names = [name for name in reader.fieldnames if name not in metadata_columns]
        if not bin_names:
            raise ValueError("The CSV must contain at least one bin-count column.")

        clades_by_proband: dict[str, list[list[int]]] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                values = [int(row[bin_name]) for bin_name in bin_names]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid bin count on CSV line {line_number}.") from error
            if any(value < 0 for value in values):
                raise ValueError(f"Negative bin count on CSV line {line_number}.")
            if sum(values) == 0:
                raise ValueError(f"A clade must contain at least one result on CSV line {line_number}.")
            clades_by_proband.setdefault(row["proband_number"], []).append(values)

    if proband_bin_size is not None:
        try:
            numeric_proband_numbers = {
                proband_number: int(proband_number)
                for proband_number in clades_by_proband
            }
        except ValueError as error:
            raise ValueError("Proband numbers must be integers when binning is enabled.") from error
        bin_start = (
            min(numeric_proband_numbers.values())
            if proband_bin_start is None
            else proband_bin_start
        )
        clades_by_bin: dict[str, list[list[int]]] = {}
        for proband_number, clades in clades_by_proband.items():
            numeric_proband_number = numeric_proband_numbers[proband_number]
            bin_label = bin_start + (
                (numeric_proband_number - bin_start) // proband_bin_size
            ) * proband_bin_size
            clades_by_bin.setdefault(str(bin_label), []).extend(clades)
        clades_by_proband = dict(
            sorted(clades_by_bin.items(), key=lambda item: int(item[0]))
        )
    elif bin_definition_path is not None:
        clades_by_proband = group_clades_by_definition(
            clades_by_proband, bin_definition_path, bin_names
        )

    random_generator = random.Random(seed)
    fieldnames = output_columns(bin_names)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for proband_number, clades in clades_by_proband.items():
            observed_counts = [sum(clade[index] for clade in clades) for index in range(len(bin_names))]
            observed_total = sum(observed_counts)
            observed_cumulative_counts: list[int] = []
            cumulative_count = 0
            for value in observed_counts:
                cumulative_count += value
                observed_cumulative_counts.append(cumulative_count)

            bootstrap_boundaries = [[] for _ in bin_names]
            for _ in range(bootstrap_replicates):
                sampled_clades = [random_generator.choice(clades) for _ in clades]
                sampled_counts = [
                    sum(clade[index] for clade in sampled_clades)
                    for index in range(len(bin_names))
                ]
                sampled_total = sum(sampled_counts)
                sampled_cumulative_count = 0
                for index, value in enumerate(sampled_counts):
                    sampled_cumulative_count += value
                    bootstrap_boundaries[index].append(sampled_cumulative_count / sampled_total)

            output_row: dict[str, str | float] = {"proband_number": proband_number}
            for bin_name, observed_cumulative_count, boundaries in zip(
                bin_names, observed_cumulative_counts, bootstrap_boundaries
            ):
                boundaries.sort()
                lower = percentile(boundaries, 0.025)
                upper = percentile(boundaries, 0.975)
                output_row.update(
                    {
                        f"{bin_name}_lower_deviation": max(
                            0.0, observed_cumulative_count - lower * observed_total
                        ),
                        f"{bin_name}_upper_deviation": max(
                            0.0, upper * observed_total - observed_cumulative_count
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
    parser.add_argument(
        "--cluster-bootstrap",
        action="store_true",
        help="Treat each input row as one clade and bootstrap clades within each proband",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="Number of clade-bootstrap replicates (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible bootstrap intervals",
    )
    parser.add_argument(
        "--proband-bin-size",
        type=int,
        help="Combine consecutive proband numbers into bins of this size before bootstrapping",
    )
    parser.add_argument(
        "--proband-bin-start",
        type=int,
        help="First bin label; defaults to the smallest proband number in the input",
    )
    parser.add_argument(
        "--bin-definition-csv",
        type=Path,
        help="CSV of aggregate bins used to group consecutive raw proband results",
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
    if arguments.method is None and not arguments.cluster_bootstrap:
        arguments.method = input(
            "Confidence-interval method [wald/wilson] (default: wilson): "
        ).strip().lower() or "wilson"
        if arguments.method not in ("wald", "wilson"):
            parser.error("Method must be 'wald' or 'wilson'.")
    if arguments.z_score <= 0:
        parser.error("--z-score must be positive")
    if arguments.cluster_bootstrap and arguments.bootstrap_replicates < 2:
        parser.error("--bootstrap-replicates must be at least 2")
    if arguments.proband_bin_size is not None and arguments.proband_bin_size < 1:
        parser.error("--proband-bin-size must be positive")
    if arguments.proband_bin_start is not None and arguments.proband_bin_size is None:
        parser.error("--proband-bin-start requires --proband-bin-size")
    if arguments.proband_bin_size is not None and not arguments.cluster_bootstrap:
        parser.error("--proband-bin-size requires --cluster-bootstrap")
    if arguments.bin_definition_csv is not None and not arguments.cluster_bootstrap:
        parser.error("--bin-definition-csv requires --cluster-bootstrap")
    if arguments.bin_definition_csv is not None and arguments.proband_bin_size is not None:
        parser.error("Use either --bin-definition-csv or --proband-bin-size, not both")
    try:
        if arguments.cluster_bootstrap:
            build_cluster_bootstrap_intervals(
                arguments.input_csv,
                arguments.output_csv,
                arguments.bootstrap_replicates,
                arguments.seed,
                arguments.proband_bin_size,
                arguments.proband_bin_start,
                arguments.bin_definition_csv,
            )
        else:
            build_intervals(arguments.input_csv, arguments.output_csv, arguments.z_score, arguments.method)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
