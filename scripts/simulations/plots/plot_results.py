import csv
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scripts.utility.basic_utility import (
    get_filepath,
    get_number_input_with_lower_bound,
    get_number_input_in_bounds,
    get_basename_without_extension,
    get_yes_or_no, get_non_existing_path,
)

axis_options = ("Types of axis:\n"
                "1) Linear axis\n"
                "2) Log-scale axis\n")
log_axis_options = 2


def main():
    input_csv_filepath = get_filepath("Specify the input csv file path:")
    input_filename_no_ext = get_basename_without_extension(input_csv_filepath)
    output_filename = f"{input_filename_no_ext}.svg"
    current_directory = os.getcwd()
    os.chdir(Path(input_csv_filepath).parent)
    output_filepath = get_non_existing_path("Specify the output file path:")
    # Prompt the user to understand how to parse the file
    x_axis_column_index = get_number_input_with_lower_bound(
        "Specify the x-axis column index in the csv file:", 0
    )
    y_axis_column_index = get_number_input_with_lower_bound(
        "Specify the y-axis column index in the csv file:", 0
    )
    size_column_index = None
    support_different_sizes = get_yes_or_no("Do you want to specify an additional column to be used as a coefficient"
                                            " to scale the dots?")
    if support_different_sizes:
        size_column_index = get_number_input_with_lower_bound(
            "Choose the column index for the scale parameter", 0)
    build_linear_regression = get_yes_or_no("Do you want to build a linear regression?")

    print(axis_options)
    x_axis_option = get_number_input_in_bounds("Choose the x-axis type:", 1, 2)
    y_axis_option = get_number_input_in_bounds("Choose the y-axis type:", 1, 2)
    # Process the file
    x_vals, y_vals = [], []
    sizes = []
    with open(input_csv_filepath, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                x_vals.append(float(row[x_axis_column_index]))
                y_vals.append(float(row[y_axis_column_index]))
                if size_column_index is not None:
                    alignment_value = float(row[size_column_index])
                    sizes.append((alignment_value ** 0.5) * 10)
            except (ValueError, IndexError):
                continue  # skip bad rows

    # Make the figure larger — increase figsize as needed
    plt.figure(figsize=(14, 10))  # larger canvas

    # Create scatter plot
    if size_column_index is not None:
        plt.scatter(x_vals, y_vals, s=sizes, alpha=0.6)
    else:
        plt.scatter(x_vals, y_vals, s=70, alpha=0.6)
    # Log scale options
    if x_axis_option == 2:
        plt.xscale("log")
    if y_axis_option == 2:
        plt.yscale("log")

    if build_linear_regression:
        result = linregress(x_vals, y_vals)
        print(f"Slope:       {result.slope:.6f}")
        print(f"Intercept:   {result.intercept:.6f}")
        print(f"Correlation: {result.rvalue:.6f}")
        print(f"P-value:     {result.pvalue:.4e}")
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        y_line = result.intercept + result.slope * x_line
        plt.plot(x_line, y_line)

    # Make tick labels larger
    plt.tick_params(axis='both', which='major', labelsize=30, pad=20)
    plt.tick_params(axis='both', which='minor', labelsize=30, pad=20)

    # Grid for better readability
    plt.grid(True, which='major', linestyle='--', linewidth=1)

    # Save with higher DPI for clarity
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=400)
    os.chdir(current_directory)
    print(f"Scatter plot saved to {output_filepath}")


if __name__ == "__main__":
    main()
