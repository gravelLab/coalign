"""Bin consecutive simulation rows by their accumulated clade count."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from scripts.utility.basic_utility import (
    get_filepath,
    get_natural_number_input,
    get_non_existing_path,
)


METADATA_COLUMNS = {"proband_number", "row_count", "clade_count"}


def get_output_path(input_path: Path) -> Path:
    """Prompt for a non-existing CSV path in the input file's directory."""
    current_directory = Path.cwd()
    try:
        os.chdir(input_path.parent)
        while True:
            output_path = Path(get_non_existing_path("Output CSV filename:"))
            if not output_path.suffix:
                output_path = output_path.with_suffix(".csv")
            output_path = output_path.resolve()
            if output_path.exists():
                print("The specified path exists, try again")
                continue
            return output_path
    finally:
        os.chdir(current_directory)


def read_input_rows(input_path: Path) -> tuple[list[str], list[dict[str, int]]]:
    """Read and validate count rows from a grouped-results CSV file."""
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("The input CSV must have a header row.")
        required_columns = {"proband_number", "clade_count"}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "The input CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        category_columns = [
            column for column in reader.fieldnames if column not in METADATA_COLUMNS
        ]
        if not category_columns:
            raise ValueError("The input CSV must contain at least one category column.")
        rows: list[dict[str, int]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed_row = {
                    column: int(value) for column, value in row.items() if value is not None
                }
            except ValueError as error:
                raise ValueError(f"Invalid integer value on CSV line {line_number}.") from error
            if any(value < 0 for value in parsed_row.values()):
                raise ValueError(f"Negative count on CSV line {line_number}.")
            if sum(parsed_row[column] for column in category_columns) != parsed_row["clade_count"]:
                raise ValueError(
                    f"Category counts do not sum to clade_count on CSV line {line_number}."
                )
            rows.append(parsed_row)
    return reader.fieldnames, rows


def bin_rows_by_clade_count(fieldnames: list[str], rows: list[dict[str, int]],
                            bin_size: int) -> list[dict[str, int]]:
    """Combine consecutive rows until each bin reaches the requested clade count."""
    category_columns = [column for column in fieldnames if column not in METADATA_COLUMNS]
    output_rows: list[dict[str, int]] = []
    pending_rows: list[dict[str, int]] = []
    pending_clade_count = 0
    def append_bin() -> None:
        if not pending_rows:
            return
        output_row = {
            "proband_number": pending_rows[0]["proband_number"],
            "clade_count": pending_clade_count,
        }
        if "row_count" in fieldnames:
            output_row["row_count"] = sum(row["row_count"] for row in pending_rows)
        for column in category_columns:
            output_row[column] = sum(row[column] for row in pending_rows)
        output_rows.append(output_row)
    for row in rows:
        pending_rows.append(row)
        pending_clade_count += row["clade_count"]
        if pending_clade_count >= bin_size:
            append_bin()
            pending_rows = []
            pending_clade_count = 0
    append_bin()
    return output_rows


def write_output_rows(output_path: Path, fieldnames: list[str],
                      rows: list[dict[str, int]]) -> None:
    """Write binned results while retaining the input column order."""
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    input_path = Path(get_filepath("Input CSV filepath:"))
    bin_size = get_natural_number_input("Clades per bin:")
    output_path = get_output_path(input_path)
    try:
        fieldnames, rows = read_input_rows(input_path)
        binned_rows = bin_rows_by_clade_count(fieldnames, rows, bin_size)
        write_output_rows(output_path, fieldnames, binned_rows)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Binned results saved to {output_path}")


if __name__ == "__main__":
    main()
