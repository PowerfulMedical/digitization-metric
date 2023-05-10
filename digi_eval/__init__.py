import numpy as np
import typing as t
import cv2
from scipy.ndimage import distance_transform_edt


def bool_array_to_regions(arr: np.ndarray) -> t.List[slice]:
    """
    Convert an array of booleans into a slice of regions that contain True values.

    :param arr: Input array.
    :return: Slice with True positions.
    """

    if len(arr) == 0:
        return []

    # Find the start positions of contiguous True regions
    begs = ([0] if arr[0] else []) + ((~arr[:-1] & arr[1:]).nonzero()[0] + 1).tolist()
    # Find the end positions of contiguous True regions
    ends = ((arr[:-1] & ~arr[1:]).nonzero()[0] + 1).tolist() + ([len(arr)] if arr[-1] else [])

    # Create slices for each contiguous True region using the start and end positions
    return [slice(beg, end) for beg, end in zip(begs, ends)]


def lead_to_coords(
    lead: np.ndarray,
    lead_sampling_freq: int,
    square_size: int,
    draw_speed: int,
    draw_voltage: int,
) -> t.Tuple[int, int]:
    """
    Convert a lead array into x and y coordinates for drawing.

    :param lead: Lead array.
    :param lead_sampling_freq: Lead sampling frequency.
    :param square_size: Square size for drawing.
    :param draw_speed: Drawing speed.
    :param draw_voltage: Drawing voltage.
    :return: Tuple of x and y coordinates.
    """

    # Calculate total time in seconds
    total_s = (len(lead) - 1) / lead_sampling_freq
    # Calculate pixels per millimeter
    pixels_per_mm = square_size / 5

    # Calculate y coordinates based on lead values, draw_voltage, and pixels_per_mm
    y_coords = -lead * draw_voltage * pixels_per_mm
    # Calculate pixels per second
    pixels_per_s = draw_speed * pixels_per_mm
    # Generate x coordinates using linspace function. The linspace function creates
    # an array of evenly spaced values over the specified interval.
    x_coords = np.linspace(0, total_s * pixels_per_s, len(lead))

    return x_coords, y_coords


def draw_leads(
    leads: list,
    square_size: int = 50,
    draw_speed: int = 25,
    draw_voltage: int = 10,
    pad_left_right: int = 256,
    pad_top_bot: int = 512,
) -> list:
    """
    Draw ECG leads as images.

    :param leads: List of leads.
    :param square_size: Square size for drawing.
    :param draw_speed: Drawing speed.
    :param draw_voltage: Drawing voltage.
    :param pad_left_right: Padding left and right.
    :param pad_top_bot: Padding top and bottom.
    :return: List of images.
    """

    # Convert leads to coordinates using lead_to_coords function
    lead_coords = [
        lead_to_coords(lead["lead"], lead["lead_sampling_freq"], square_size, draw_speed, draw_voltage)
        for lead in leads
    ]

    # Find the minimum y coordinate among all leads
    y_min = min(np.nanmin(y) for _, y in lead_coords)
    # Adjust y coordinates by subtracting y_min and adding pad_top_bot
    lead_coords = [(x, y - y_min + pad_top_bot) for x, y in lead_coords]

    # Calculate the total size of the output image in y and x dimensions
    total_y_size = int(max(np.nanmax(y) for _, y in lead_coords) + pad_top_bot)
    total_x_size = int(max(np.nanmax(x) for x, _ in lead_coords) + 2 * pad_left_right)

    out = []
    for x, y in lead_coords:
        # Initialize a black image with the calculated dimensions
        image = np.zeros((total_y_size, total_x_size))
        # Find regions where y coordinates are not NaN
        nan_regions = bool_array_to_regions(~np.isnan(y).reshape((-1, 1)))
        # Create a list of lines for each region
        all_lines = [np.vstack([x[r] + pad_left_right, y[r]]).T.astype(np.int32) for r in nan_regions]

        # Draw the lines on the image using cv2.polylines
        cv2.polylines(image, all_lines, False, 1)
        out.append(image.astype(np.int64))

    return out


def compare_leads(
    ecg: np.ndarray,
    gold: np.ndarray,
    ecg_fs: int = 500,
    gold_fs: int = 500,
    max_shift_x: int = 200,
    max_shift_y: int = 300,
) -> float:
    """
    Compare two ECG leads and return a similarity score.

    :param ecg: ECG lead array.
    :param gold: Gold standard ECG lead array.
    :param ecg_fs: ECG lead sampling frequency.
    :param gold_fs: Gold standard ECG lead sampling frequency.
    :param max_shift_x: Maximum tested shift on x axis.
    :param max_shift_y: Maximum tested shift on y axis.
    :return: Similarity score.
    """
    # Check if the entire ECG lead is missing (all NaN values)
    if np.all(np.isnan(ecg)):
        # Insert one point in the middle of the ECG lead array
        ecg[len(ecg) // 2] = 0

    # Draw ECG lead images for comparison
    ecg_img, gold_img = draw_leads(
        [{"lead": ecg, "lead_sampling_freq": ecg_fs}, {"lead": gold, "lead_sampling_freq": gold_fs}]
    )
    # Calculate distance matrices for the ECG images
    ecg_dist = make_distance_matrix(ecg_img)
    gold_dist = make_distance_matrix(gold_img)

    # Find non-zero (i.e., drawn) points in the ECG images
    non_zero_ecg = np.nonzero(ecg_img)
    non_zero_gold = np.nonzero(gold_img)

    # Initialize shift matrices for comparing ECG images
    shift_mat = np.zeros((max_shift_y * 2 + 1, max_shift_x * 2 + 1), dtype=np.float64)
    shift_mat_back = np.zeros((max_shift_y * 2 + 1, max_shift_x * 2 + 1), dtype=np.float64)

    # Update shift matrices with distances between ECG image points
    for i in range(len(non_zero_ecg[0])):
        shift_mat += gold_dist[
            non_zero_ecg[0][i] - max_shift_y : non_zero_ecg[0][i] + 1 + max_shift_y,
            non_zero_ecg[1][i] - max_shift_x : non_zero_ecg[1][i] + 1 + max_shift_x,
        ]
    for i in range(len(non_zero_gold[0])):
        shift_mat_back += ecg_dist[
            non_zero_gold[0][i] - max_shift_y : non_zero_gold[0][i] + 1 + max_shift_y,
            non_zero_gold[1][i] - max_shift_x : non_zero_gold[1][i] + 1 + max_shift_x,
        ]

    # Combine the shift matrices and calculate the similarity score
    score_mat = shift_mat + shift_mat_back[::-1, ::-1]
    return np.min(score_mat)


def make_distance_matrix(img: np.ndarray) -> np.ndarray:
    """
    Create a distance matrix for the given image.

    :param img: Input image (np.ndarray)
    :return: Distance matrix (np.ndarray)
    """
    # Compute the Euclidean distance transform on the binary image where
    # the input image is equal to 0 (background)
    edt = distance_transform_edt(img == 0)

    # Subtract 1 from the distance transform and clip values below 0
    return np.maximum(edt - 1, 0)


def compare_ecgs(ecg: dict, gold: dict, ecg_fs: int = 500, gold_fs: int = 500) -> float:
    """
    Compare two sets of ECG leads and return a total error score.

    :param ecg: Dictionary of ECG leads.
    :param gold: Dictionary of gold standard ECG leads.
    :param ecg_fs: ECG lead sampling frequency.
    :param gold_fs: Gold standard ECG lead sampling frequency.
    :return: Total error score.
    """
    # Initialize the total error score
    total_err = 0

    # Iterate through each lead in the gold standard set
    for k in gold:
        # Calculate the error score between the input and gold standard leads
        lead_err = compare_leads(ecg[k], gold[k], ecg_fs, gold_fs)
        # Accumulate the lead error scores
        total_err += lead_err
    return total_err
