import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utility.basic_utility import get_filepath, get_basename_without_extension


def get_confidence_intervals(dataframe: pd.DataFrame, y_columns: list[str]) -> pd.DataFrame | None:
    confidence_interval_path = input(
        "Specify the path to the confidence-interval CSV file (leave blank to skip):"
    ).strip()
    if not confidence_interval_path:
        return None

    confidence_intervals = pd.read_csv(confidence_interval_path)
    if "proband_number" not in confidence_intervals:
        raise ValueError("The confidence-interval CSV must contain a 'proband_number' column.")

    required_columns = {
        f"{column}_{bound}_deviation"
        for column in y_columns
        for bound in ("lower", "upper")
    }
    missing_columns = required_columns.difference(confidence_intervals.columns)
    if missing_columns:
        raise ValueError(
            "The confidence-interval CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    confidence_intervals = confidence_intervals.set_index("proband_number")
    proband_numbers = dataframe["proband_number"]
    missing_proband_numbers = proband_numbers[~proband_numbers.isin(confidence_intervals.index)]
    if not missing_proband_numbers.empty:
        raise ValueError(
            "The confidence-interval CSV is missing proband numbers: "
            + ", ".join(map(str, missing_proband_numbers.tolist()))
        )
    return confidence_intervals.loc[proband_numbers].reset_index()



# Horizontal distance (in x-axis units) between neighbouring error bars when staggering is enabled
ERROR_BAR_STAGGER_STEP = 0.15
# Error bars at the same x-value closer than this (in percentage points) are treated as overlapping
ERROR_BAR_OVERLAP_MARGIN = 1.0


def ask_whether_to_stagger_error_bars() -> bool:
    answer = input("Stagger overlapping error bars horizontally? [y/N]:").strip().lower()
    return answer in ("y", "yes")


def get_overlap_stagger_offsets(x_values: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                                x_min: float, x_max: float) -> np.ndarray:
    """
    Returns horizontal offsets for the error bars, so that only the bars that overlap are moved apart.
    At every x-value, the bars whose vertical ranges overlap (or come closer than ERROR_BAR_OVERLAP_MARGIN)
    form a group. A bar that does not overlap with any other one stays at its x-value. The bars within a group
    are spread symmetrically around the x-value in the order of their lower bounds. At the ends of the x-range,
    the whole group is shifted inwards, so that no error bar is placed outside the plot area.
    """
    offsets = np.zeros(len(x_values))
    for x_value in np.unique(x_values):
        indices = np.flatnonzero(x_values == x_value)
        indices = indices[np.lexsort((upper_bounds[indices], lower_bounds[indices]))]
        groups = []
        group_upper_bound = None
        for index in indices:
            if group_upper_bound is not None and lower_bounds[index] <= group_upper_bound + ERROR_BAR_OVERLAP_MARGIN:
                groups[-1].append(index)
                group_upper_bound = max(group_upper_bound, upper_bounds[index])
            else:
                groups.append([index])
                group_upper_bound = upper_bounds[index]
        for group in groups:
            if len(group) == 1:
                continue
            half_width = (len(group) - 1) / 2 * ERROR_BAR_STAGGER_STEP
            shift = max(0.0, x_min - (x_value - half_width)) - max(0.0, x_value + half_width - x_max)
            for position, index in enumerate(group):
                offsets[index] = (position - (len(group) - 1) / 2) * ERROR_BAR_STAGGER_STEP + shift
    return offsets


def get_unique_error_bar_mask(x_values: np.ndarray, y_values: np.ndarray, lower_bounds: np.ndarray,
                              upper_bounds: np.ndarray) -> np.ndarray:
    """
    Returns a mask that keeps only the first of the identical error bars. When a category is empty at an x-value,
    its upper boundary coincides with the boundary below it, so both boundaries get the same error bar.
    """
    keys = np.round(np.column_stack((x_values, y_values, lower_bounds, upper_bounds)), 9)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    mask = np.zeros(len(x_values), dtype=bool)
    mask[first_indices] = True
    return mask


def restrict_to_x_range(x_values, y_values, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Keeps only the part of a piecewise-linear series that lies within [x_min, x_max]. The values at the
    range boundaries are linearly interpolated, so the visible shape does not change, but no geometry
    hidden by the axes clipping ends up in the saved SVG. Expects x_values sorted in ascending order.
    """
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    inside = (x_array > x_min) & (x_array < x_max)
    x_range = np.concatenate(([x_min], x_array[inside], [x_max]))
    y_range = np.concatenate((
        [np.interp(x_min, x_array, y_array)],
        y_array[inside],
        [np.interp(x_max, x_array, y_array)],
    ))
    return x_range, y_range


def main():
    filepath = get_filepath("Specify the path to the data file:")
    os.chdir(Path(filepath).parent)
    result_path = Path(filepath).parent / get_basename_without_extension(filepath)
    # title = input("Specify the title of the figure:")
    df = pd.read_csv(filepath)
    # Define x-axis and values to plot
    x = df['proband_number']
    metadata_columns = {'proband_number', 'row_count', 'clade_count'}
    y_columns = [column for column in df.columns if column not in metadata_columns]
    confidence_intervals = get_confidence_intervals(df, y_columns)
    stagger_error_bars = confidence_intervals is not None and ask_whether_to_stagger_error_bars()

    # Normalize the values to percentages. Replace 0 to avoid division by zero
    df['total'] = df[y_columns].sum(axis=1).replace(0, 1)
    for col in y_columns:
        df[col] = (df[col] / df['total']) * 100

    # Calculate cumulative sum for stacked plot
    cumulative_sum = df[y_columns].cumsum(axis=1)

    x_min, x_max = 3, max(x)

    plt.figure(figsize=(32, 16))

    colors_by_column = {
        'no_solutions': '#898989',
        'individual_and_spouses': '#00452c',
        'individual_and_non_spouse': '#00965f',
        'no_individual_spouse': '#f1c338',
        'neither_individual_nor_spouse': '#be5103',
        'neither_individual_nor_spouse_only_super_founders': '#b0aeae',
    }
    handles = []

    # bottom = None
    # for i, col in enumerate(y_columns):
    #     bar = plt.bar(x, df[col], bottom=bottom, color=colors[i], label=col, alpha=0.7, width=1.0, align='edge')
    #     handles.append(bar)
    #     bottom = df[col] if bottom is None else bottom + df[col]

    for i, col in enumerate(y_columns):
        color = colors_by_column.get(col)
        if color is None:
            raise ValueError(f"No color is configured for the '{col}' column.")
        lower_boundary = 0 if i == 0 else cumulative_sum.iloc[:, i - 1]
        upper_boundary = cumulative_sum.iloc[:, i]
        fill_x, fill_lower = restrict_to_x_range(x, np.broadcast_to(lower_boundary, len(x)), x_min, x_max)
        _, fill_upper = restrict_to_x_range(x, upper_boundary, x_min, x_max)
        fill = plt.fill_between(fill_x, fill_lower, fill_upper, color=color, label=col, alpha=0.7)
        handles.append(fill)

    if confidence_intervals is not None:
        errorbar_x, errorbar_y, lower_bounds, upper_bounds = [], [], [], []
        for i, col in enumerate(y_columns[:-1]):
            # Draw a bar wherever the interval has non-zero width. Wilson intervals are one-sided
            # at 0% and 100%, so those points must not be filtered out.
            valid_points = (confidence_intervals[f"{col}_lower_deviation"] > 0) | (
                confidence_intervals[f"{col}_upper_deviation"] > 0
            )
            # Error bars are drawn unclipped (so the one on the axis edge is fully visible),
            # so points outside the x-range must be skipped explicitly.
            valid_points &= x.between(x_min, x_max)
            if not valid_points.any():
                continue
            lower_deviation = (
                confidence_intervals.loc[valid_points, f"{col}_lower_deviation"]
                / df.loc[valid_points, 'total']
                * 100
            ).clip(lower=0)
            upper_deviation = (
                confidence_intervals.loc[valid_points, f"{col}_upper_deviation"]
                / df.loc[valid_points, 'total']
                * 100
            ).clip(lower=0)
            boundary_y = cumulative_sum.loc[valid_points].iloc[:, i].to_numpy()
            errorbar_x.append(x.loc[valid_points].to_numpy(dtype=float))
            errorbar_y.append(boundary_y)
            # Clamp the intervals to the visible [0, 100] range, so no part of an error bar
            # lies outside the axes (it would either enlarge the figure or be hidden geometry in the SVG).
            lower_bounds.append(np.clip(boundary_y - lower_deviation.to_numpy(), 0, 100))
            upper_bounds.append(np.clip(boundary_y + upper_deviation.to_numpy(), 0, 100))

        if errorbar_x:
            errorbar_x = np.concatenate(errorbar_x)
            errorbar_y = np.concatenate(errorbar_y)
            lower_bounds = np.concatenate(lower_bounds)
            upper_bounds = np.concatenate(upper_bounds)
            unique_bars = get_unique_error_bar_mask(errorbar_x, errorbar_y, lower_bounds, upper_bounds)
            errorbar_x = errorbar_x[unique_bars]
            errorbar_y = errorbar_y[unique_bars]
            lower_bounds = lower_bounds[unique_bars]
            upper_bounds = upper_bounds[unique_bars]
            if stagger_error_bars:
                errorbar_x = errorbar_x + get_overlap_stagger_offsets(
                    errorbar_x, lower_bounds, upper_bounds, x_min, x_max
                )
            plt.errorbar(
                errorbar_x,
                errorbar_y,
                # Clip at 0: floating-point cumulative sums can end up marginally above 100,
                # which would otherwise yield tiny negative error lengths.
                yerr=[np.clip(errorbar_y - lower_bounds, 0, None), np.clip(upper_bounds - errorbar_y, 0, None)],
                fmt="none",
                ecolor="black",
                capsize=5,
                elinewidth=1.5,
                zorder=10,
                clip_on=False,
            )

    # Set y-axis limits explicitly
    plt.ylim(0, 100)
    plt.xticks(ticks=x, labels=x, fontsize=40)
    plt.yticks(fontsize=40)
    plt.tick_params(axis="x", pad=40)
    plt.tick_params(axis="y", pad=40)

    plt.xlim(x_min, x_max)
    # Use tight_layout for better adjustment
    # plt.tight_layout()

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.savefig(fname=f"{result_path}.svg")
    plt.close()


if __name__ == "__main__":
    main()
