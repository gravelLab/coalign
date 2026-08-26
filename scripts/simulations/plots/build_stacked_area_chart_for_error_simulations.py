import os
from pathlib import Path

import matplotlib.pyplot as plt
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

    # Normalize the values to percentages. Replace 0 to avoid division by zero
    raw_counts = df[y_columns].copy()
    df['total'] = df[y_columns].sum(axis=1).replace(0, 1)
    for col in y_columns:
        df[col] = (df[col] / df['total']) * 100

    # Calculate cumulative sum for stacked plot
    cumulative_sum = df[y_columns].cumsum(axis=1)

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
        if i == 0:
            fill = plt.fill_between(x, 0, df[col], color=color, label=col, alpha=0.7)
        else:
            fill = plt.fill_between(x, cumulative_sum.iloc[:, i - 1], cumulative_sum.iloc[:, i], color=color,
                                     label=col, alpha=0.7)
        handles.append(fill)

    if confidence_intervals is not None:
        boundary_count = len(y_columns) - 1
        for i, col in enumerate(y_columns[:-1]):
            valid_points = (raw_counts[col] > 0) & (cumulative_sum.iloc[:, i] < 100)
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
            errorbar_x = (
                x.loc[valid_points] + (i - (boundary_count - 1) / 2) * 0.15
            ).to_numpy()
            errorbar_y = cumulative_sum.loc[valid_points].iloc[:, i].to_numpy()
            plt.errorbar(
                errorbar_x,
                errorbar_y,
                yerr=[lower_deviation.to_numpy(), upper_deviation.to_numpy()],
                fmt="none",
                ecolor="black",
                capsize=5,
                elinewidth=1.5,
                zorder=10,
            )

    #plt.xlabel("Proband Number", fontsize=40)
    #plt.ylabel("Percentage (%)", fontsize=40)

    #plt.title(title, fontsize=30, pad=30)

    # Set y-axis limits explicitly
    plt.ylim(0, 100)

    # # Define custom legend
    # legend_labels = [
    #     "No solutions",
    #     "Individual present, the rest are the spouses",
    #     "Individual present, and a non-spouse assignment present",
    #     "Individual not present, spouse present",
    #     "Neither individual nor spouse present",
    #     #"Neither individual nor spouse present, only super founders"
    # ]
    # grouped_handles = handles
    #
    # plt.legend(
    #     grouped_handles,
    #     legend_labels,
    #     title="Categories",
    #     title_fontsize=28,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.15),
    #     ncol=3,
    #     columnspacing=1.5,
    #     frameon=False,
    #     fontsize=18
    # )

    plt.xticks(ticks=x, labels=x, fontsize=40)
    plt.yticks(fontsize=40)
    plt.tick_params(axis="x", pad=40)
    plt.tick_params(axis="y", pad=40)

    plt.xlim(3, max(x))
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
