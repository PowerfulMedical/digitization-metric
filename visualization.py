import typing as t
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Names of the leads (in desired order)
LEAD_NAMES: t.Final[t.List[str]] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FIGURE_WIDTH = 29.7  # Width of A4 paper in cm
CM_TO_INCH_FACTOR = 1 / 2.54  # Inch per cm constant


def visualize(
    ecg_id: str,
    output_path: Path,
    ecg_data: np.ndarray,
    n_rows: int = 6,
    row_height: float = 2.5,
    padding_height: float = 1.5,
    major_grid_color: str = "#FF9090",
    fs: int = 500,
):
    """
    Pipeline for visualization of single ECG.

    :param ecg_id: ECG file name
    :param output_path: Output path
    :param ecg_data: Numpy array with ECG data
    :param n_rows: Number of rows in the ECG visualization
    :param row_height: Row height in the ECG
    :param padding_height: Top/bottom padding in the ECG visualization
    :param major_grid_color: Major grid color
    :param fs: Sampling frequency
    """
    # Create figure
    y_from, y_to = -(row_height * (n_rows - 1)) - padding_height, padding_height
    fig, ax = create_figure_ax(y_from, y_to)
    # Prepare grids and plot the ECG signals
    add_grid(ax, grid_c=major_grid_color)
    plot_ecg_signals(ax, ecg_data, n_rows, row_height, fs)
    # Save and close plot
    plt.savefig(output_path / f"{ecg_id}.jpg", dpi=100)
    plt.close(fig)


def add_grid(ax: Axes, grid_c: str = "#FF9090", ms_mm: float = 40, mv_mm: float = 0.1, major_grid_size: int = 5):
    """
    Add ECG grid.

    :param ax: Axes class for plotting
    :param grid_c: Major grid lines color
    :param ms_mm: How many milliseconds is one millimeter
    :param mv_mm: How many milliVolts is one millimeter
    :param major_grid_size: Number of the minor grid ticks between two major ticks
    """
    # Set grid locators
    ax.set_frame_on(False)
    ax.xaxis.set_major_locator(MultipleLocator(ms_mm * major_grid_size))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_locator(MultipleLocator(mv_mm * major_grid_size))
    # Setting major grid
    ax.grid(True, "major", axis="both", c=grid_c, linewidth=1.5, linestyle="-", drawstyle="steps-mid", alpha=0.9)
    # Prepare major ticks labels
    x_formatter = FuncFormatter(lambda x, pos: f"{x:g}ms" if int(x) % 1000 == 0 else "")
    ax.xaxis.set_major_formatter(x_formatter)
    y_formatter = FuncFormatter(lambda x, pos: f"{x}mV")
    ax.yaxis.set_major_formatter(y_formatter)
    # Setting minor grid locators
    ax.xaxis.set_minor_locator(MultipleLocator(ms_mm))
    ax.yaxis.set_minor_locator(MultipleLocator(mv_mm))
    # Setting minor grid
    ax.grid(True, "minor", axis="both", c=grid_c, linewidth=1, linestyle="-", drawstyle="steps-mid", alpha=0.7)


def create_figure_ax(y_min: float, y_max: float) -> t.Tuple[Figure, Axes]:
    """
    Create a figure with a size of A4 paper and creates an axes class with provided parameters.
    The difference between y_min and y_max is the height of the ECG grid in centimeters.

    :param y_min: y lower limit
    :param y_max: y upper limit
    :return: Figure and axes class
    """
    # Define A4 paper-like constants
    height = y_max - y_min  # Height of the image, 1mV == 1cm
    figure_height = height + 2  # Height plus padding
    horizontal_margin = 1  # Left/right margin in cm
    cm2inch = CM_TO_INCH_FACTOR * 2  # Scaling constant, so everything is correctly plotted
    # Calculate the position and size of the axes
    left = horizontal_margin / FIGURE_WIDTH  # Left margin for y labels
    width = 1 - 2 * left  # Same margin for right and left
    height /= figure_height  # Height ratio of the axes to the whole figure
    # Create figure and axis
    figure = plt.figure(figsize=(FIGURE_WIDTH * cm2inch, figure_height * cm2inch))
    ax = figure.add_axes((left, (1 - height) / 2, width, height))
    # Set range for the axes
    ax.set_ylim(y_min - 0.1, y_max + 0.1)  # Added padding so major grid line is always visible.
    # Calculate x limits and set them
    x_min = -440  # Padding for calibration sign, + padding so major grid line is fully visible
    x_max = (FIGURE_WIDTH - 2 * horizontal_margin) * 400 - 440  # Usable length times ms per 1 cm
    ax.set_xlim(x_min, x_max)
    return figure, ax


def plot_ecg_signals(ax: Axes, ecg_data: np.ndarray, n_rows: int, rows_dist: float, fs: int, signal_color: str = "k"):
    """
    Plot all ECG signals.

    :param ax: Axis class
    :param ecg_data: ECG data
    :param n_rows: Number of rows in the ECG
    :param rows_dist: Distance between the rows in ECG
    :param fs: Sampling frequency
    :param signal_color: Signal color
    """
    n_cols = 12 // n_rows
    signal_len = int(len(ecg_data) / n_cols)
    for j in range(12 // n_rows):
        for i in range(n_rows):
            # Select data and move it vertically to the corresponding row
            data = ecg_data[signal_len * j : signal_len * (j + 1), i + j * n_rows] - i * rows_dist
            # Plot ECG wave
            ms = (np.arange(0, signal_len) + signal_len * j) * 1000 / fs
            plt.plot(ms, data, c=signal_color, linewidth=1.5)
            # Plot lead name text
            lead_name = LEAD_NAMES[i + j * n_rows]
            plt.text(ms[0], -i * rows_dist + 0.5, lead_name, size=15, transform=ax.transData, weight="bold", c="k")
