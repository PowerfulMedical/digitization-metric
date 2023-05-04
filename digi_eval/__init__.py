import numpy as np
import typing as t
import cv2
from scipy.ndimage import distance_transform_edt

def bool_array_to_regions(arr: np.ndarray) -> t.List[slice]:
    """
    Convert an array of booleans into a slice of regions that contain True values.
    :param arr: Input array
    :return: Slice with True positions
    """
    if len(arr) == 0:
        return []
    begs = ([0] if arr[0] else []) + ((~arr[:-1] & arr[1:]).nonzero()[0] + 1).tolist()
    ends = ((arr[:-1] & ~arr[1:]).nonzero()[0] + 1).tolist() + ([len(arr)] if arr[-1] else [])

    return [slice(beg, end) for beg, end in zip(begs, ends)]

def lead_to_coords(lead: np.ndarray, lead_sampling_freq: int, square_size: int=64, draw_speed: int=25, draw_voltage: int=10) -> tuple:
    total_s = (len(lead)-1) / lead_sampling_freq
    pixels_per_mm = square_size / 5

    y_coords = -lead * draw_voltage * pixels_per_mm
    pixels_per_s = draw_speed * pixels_per_mm
    x_coords = np.linspace(0, total_s * pixels_per_s, len(lead))

    return x_coords, y_coords


def draw_leads(leads: list, square_size: int=64, draw_speed: int=25, draw_voltage: int=10, pad_left_right: int=256, pad_top_bot:int=512) -> list:
    lead_coords = [lead_to_coords(lead["lead"], lead["lead_sampling_freq"], square_size, draw_speed, draw_voltage) for lead in leads]

    y_min = min(np.nanmin(y) for _, y in lead_coords)
    lead_coords = [(x, y - y_min + pad_top_bot) for x, y in lead_coords]

    total_y_size = int(max(np.nanmax(y) for _, y in lead_coords) + pad_top_bot)
    total_x_size = int(max(np.nanmax(x) for x, _ in lead_coords) + 2*pad_left_right)

    out = []
    for x, y in lead_coords:
        image = np.zeros((total_y_size, total_x_size))

        nan_regions = bool_array_to_regions(~np.isnan(y).reshape((-1, 1)))

        all_lines = [np.vstack([x[r] + pad_left_right, y[r]]).T.astype(np.int32) for r in nan_regions]

        color=[1]
        thickness=1

        cv2.polylines(image, all_lines, False, color, thickness=thickness)
        out.append(image.astype(np.int64))

    return out


def compare_leads(ecg: np.ndarray, gold: np.ndarray, ecg_fs=500, gold_fs=500) -> float:
    if np.all(np.isnan(ecg)):  # special case with missing whole lead
        ecg[len(ecg)//2] = 0   # insert one point in the middle       
    ecg_img, gold_img = draw_leads([{"lead": ecg, "lead_sampling_freq": ecg_fs}, {"lead": gold, "lead_sampling_freq": gold_fs}])
    ecg_dist = make_distance_matrix(ecg_img)
    gold_dist = make_distance_matrix(gold_img)
    non_zero_ecg = np.nonzero(ecg_img)
    non_zero_gold = np.nonzero(gold_img)


    max_shift_x = 200
    max_shift_y = 300
    shift_mat = np.zeros((max_shift_y*2+1, max_shift_x*2+1), dtype=np.float64)
    shift_mat_back = np.zeros((max_shift_y*2+1, max_shift_x*2+1), dtype=np.float64)

    for i in range(len(non_zero_ecg[0])):
        shift_mat += gold_dist[non_zero_ecg[0][i]-max_shift_y:non_zero_ecg[0][i]+1+max_shift_y,
                               non_zero_ecg[1][i]-max_shift_x:non_zero_ecg[1][i]+1+max_shift_x]
    for i in range(len(non_zero_gold[0])):
        shift_mat_back += ecg_dist[non_zero_gold[0][i]-max_shift_y:non_zero_gold[0][i]+1+max_shift_y,
                                   non_zero_gold[1][i]-max_shift_x:non_zero_gold[1][i]+1+max_shift_x]
    score_mat = shift_mat + shift_mat_back[::-1,::-1]
    return np.min(score_mat)

def make_distance_matrix(img: np.ndarray) -> np.ndarray:
    return np.maximum(distance_transform_edt(img == 0) - 1, 0)

def compare_ecgs(ecg: dict, gold: dict, ecg_fs=500, gold_fs=500) -> float:
    total_err = 0
    for k in gold:
        lead_err = compare_leads(ecg[k], gold[k], ecg_fs, gold_fs)
        total_err += lead_err
    return total_err
