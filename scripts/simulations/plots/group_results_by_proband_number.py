import os
from pathlib import Path

import pandas as pd

from scripts.utility.basic_utility import get_filepath, get_non_existing_path


def run_interactive_session():
    filepath = Path(get_filepath("Specify the path to the data file:"))
    os.chdir(filepath.parent)

    output_filepath = get_non_existing_path(
        "Specify the output filename (without extension):"
    )
    output_filepath = f"{output_filepath}.csv"

    df = pd.read_csv(filepath)
    classification_columns = df.columns[1:]

    # Group by proband_number and accumulate values for each class
    grouped_df = (
        df.groupby("proband_number")[classification_columns]
        .sum()
        .reset_index()
    )

    # Count the number of rows accumulated for each proband
    row_count_df = (
        df.groupby("proband_number")
        .size()
        .reset_index(name="row_count")
    )

    # Merge the accumulated values with row_count
    grouped_df = pd.merge(
        grouped_df,
        row_count_df,
        on="proband_number",
        how="left"
    )

    # Calculate the total across all accumulated classification columns
    grouped_df["clade_count"] = grouped_df[classification_columns].sum(axis=1)

    # Arrange columns
    columns_order = (
        ["proband_number", "row_count", "clade_count"]
        + list(classification_columns)
    )
    grouped_df = grouped_df[columns_order]

    grouped_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    run_interactive_session()
