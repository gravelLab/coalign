import csv
import os
from enum import IntEnum
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scripts.utility.basic_utility import (
    get_filepath,
    get_number_input_with_lower_bound,
    get_number_input_in_bounds,
    get_yes_or_no, get_non_existing_path,
)

class AxisType(IntEnum):
    LINEAR = 1
    LOGARITHMIC = 2


class RegressionType(IntEnum):
    OLS = 1
    OLS_HC3 = 2
    FRACTIONAL_LOGIT = 3


class VariancePlotType(IntEnum):
    ROLLING_WINDOW = 1
    LOWESS = 2


axis_options = ("Types of axis:\n"
                "1) Linear axis\n"
                "2) Log-scale axis\n")

regression_options = (
    "Types of regression:\n"
    "1) Assume equal variance (OLS)\n"
    "2) Do not assume equal variance (OLS with HC3)\n"
    "3) Fractional logit regression (outcome from 0 to 1)\n"
)

variance_plotting_options = (
    "Types of variance plotting:\n"
    "1) Rolling windows\n"
    "2) LOWESS\n"
)

def draw_regression(x_vals, y_vals, regression_option: RegressionType):
    x_array = np.asarray(x_vals)
    y_array = np.asarray(y_vals)
    # Add the intercept column
    X = sm.add_constant(x_array)
    if regression_option == RegressionType.FRACTIONAL_LOGIT:
        if not np.all((0 <= y_array) & (y_array <= 1)):
            raise ValueError("Fractional logit regression requires y-axis values from 0 to 1.")
        if np.unique(y_array).size < 2:
            raise ValueError("Fractional logit regression requires at least two distinct y-axis values.")
        model = sm.GLM(y_array, X, family=sm.families.Binomial()).fit(cov_type="HC3")
        regression_name = "Fractional logit regression"
    elif regression_option == RegressionType.OLS:
        # Ordinary OLS regression assuming equal variance
        model = sm.OLS(y_array, X).fit()
        regression_name = "OLS (equal variance assumed)"
    else:
        # OLS regression with heteroscedasticity-robust inference
        model = sm.OLS(y_array, X).fit(cov_type="HC3")
        regression_name = "OLS with HC3 robust standard errors"
    if regression_option == RegressionType.FRACTIONAL_LOGIT:
        print(f"\nRegression: {regression_name}")
        print(f"Slope:       {model.params[1]:.6f}")
        print(f"Intercept:   {model.params[0]:.6f}")
        print(f"P-value:     {model.pvalues[1]:.4e}")
        print(f"Odds ratio:  {np.exp(model.params[1]):.6f}")
        print(f"Pseudo R2:   {model.pseudo_rsquared(kind='mcf'):.6f}")
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        y_line = model.predict(sm.add_constant(x_line))
        plt.plot(x_line, y_line, linewidth=3)
        plt.ylim(-0.05, 1.05)
        return x_array, y_array, model

    # Print the parameters
    print(f"\nRegression: {regression_name}")
    print(f"Slope:       {model.params[1]:.6f}")
    print(f"Intercept:   {model.params[0]:.6f}")
    print(f"P-value:     {model.pvalues[1]:.4e}")
    # Calculate R²
    print(f"R²:          {model.rsquared:.6f}")
    # Generate points for the regression line
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    X_line = sm.add_constant(x_line)
    y_line = model.predict(X_line)
    plt.plot(x_line, y_line, linewidth=3)
    return x_array, y_array, model


def draw_variance_lowess(x_array, model, frac=0.3):
    """
    Plots the estimated local residual variance against X using LOWESS.

    Parameters:
    - x_array: array-like of predictor values
    - model: fitted statsmodels OLS result object
    - ax: matplotlib axis (optional). Uses current axis if None.
    - frac: float, proportion of data used when estimating local regression (0.2 to 0.5)
    """
    x_vec = np.asarray(x_array)
    residuals = np.asarray(model.resid)
    # Squared residuals estimate point-by-point variance: E[e^2 | X]
    sq_residuals = residuals ** 2
    # Fit LOWESS curve: returns sorted columns [[x_1, y_1], [x_2, y_2], ...]
    lowess_fit = sm.nonparametric.lowess(sq_residuals, x_vec, frac=frac)
    x_smooth = lowess_fit[:, 0]
    var_smooth = lowess_fit[:, 1]
    # Set up twin axis
    ax = plt.gca()
    ax2 = ax.twinx()
    # Plot smooth variance curve
    ax2.plot(
        x_smooth,
        var_smooth,
        color="crimson",
        linewidth=2.5,
        linestyle="--",
        label="LOWESS Residual Variance"
    )
    ax2.set_ylabel("Estimated Residual Variance $\\hat{\\sigma}^2(X)$", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    return ax2

def draw_variance(x_array, model):
    # Sort observations by x
    sort_indices = np.argsort(x_array)
    x_sorted = x_array[sort_indices]
    residuals_sorted = model.resid[sort_indices]
    # Number of neighboring observations used to estimate
    # the local variance
    window_size = max(5, len(x_sorted) // 10)
    variance_x = []
    local_variances = []
    half_window = window_size // 2
    for i in range(len(x_sorted)):
        start = max(0, i - half_window)
        end = min(len(x_sorted), i + half_window + 1)
        window_residuals = residuals_sorted[start:end]
        if len(window_residuals) > 1:
            variance_x.append(x_sorted[i])
            local_variances.append(
                np.var(window_residuals, ddof=1)
            )
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(variance_x, local_variances,
            linewidth=3,
            linestyle="--",
            label="Local residual variance",
    )
    ax2.set_ylabel("Residual variance", fontsize=30)
    ax2.tick_params(axis="y", which="major", labelsize=30, pad=20)

def main():
    input_csv_filepath = get_filepath("Specify the input csv file path:")
    input_csv_path = Path(input_csv_filepath)
    if input_csv_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV data file, but received '{input_csv_path.suffix or 'a file without an extension'}'."
        )
    current_directory = os.getcwd()
    os.chdir(input_csv_path.parent)
    output_filepath = get_non_existing_path("Specify the output filename/filepath:")
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
            "Choose the column index for the scale parameter:", 0)
    build_regression = get_yes_or_no("Do you want to build a regression model?")
    regression_option = None
    plot_variance_option = None
    if build_regression:
        print(regression_options)
        regression_option = get_number_input_in_bounds(
            "Choose the regression type:", RegressionType.OLS.value, RegressionType.FRACTIONAL_LOGIT.value
        )
        regression_option = RegressionType(regression_option)
        if regression_option != RegressionType.FRACTIONAL_LOGIT:
            plot_variance = get_yes_or_no(
                "Do you want to visualize how the residual variance changes?"
            )
            if plot_variance:
                print(variance_plotting_options)
                plot_variance_option = get_number_input_in_bounds(
                    "Choose how to plot the variance:",
                    VariancePlotType.ROLLING_WINDOW.value,
                    VariancePlotType.LOWESS.value,
                )
                plot_variance_option = VariancePlotType(plot_variance_option)

    print(axis_options)
    x_axis_option = AxisType(get_number_input_in_bounds(
        "Choose the x-axis type:", AxisType.LINEAR.value, AxisType.LOGARITHMIC.value
    ))
    y_axis_option = AxisType(get_number_input_in_bounds(
        "Choose the y-axis type:", AxisType.LINEAR.value, AxisType.LOGARITHMIC.value
    ))
    if (build_regression and regression_option == RegressionType.FRACTIONAL_LOGIT
            and y_axis_option == AxisType.LOGARITHMIC):
        raise ValueError("Fractional logit regression requires a linear y-axis.")
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

    if not x_vals:
        raise ValueError(
            "No numeric data was found in the selected CSV columns. "
            "Check the input file and column indexes."
        )

    # Make the figure larger — increase figsize as needed
    plt.figure(figsize=(14, 10))  # larger canvas

    # Create scatter plot
    if size_column_index is not None:
        plt.scatter(x_vals, y_vals, s=sizes, alpha=0.6)
    else:
        plt.scatter(x_vals, y_vals, s=70, alpha=0.6)
    # Log scale options
    if x_axis_option == AxisType.LOGARITHMIC:
        plt.xscale("log")
    if y_axis_option == AxisType.LOGARITHMIC:
        plt.yscale("log")

    if build_regression:
        x_array, y_array, model = draw_regression(x_vals, y_vals, regression_option)
        if plot_variance_option == VariancePlotType.ROLLING_WINDOW:
            draw_variance(x_array, model)
        elif plot_variance_option == VariancePlotType.LOWESS:
            draw_variance_lowess(x_array, model)

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
