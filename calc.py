import sys
import numpy as np
from pathlib import Path
import pandas as pd
from digi_eval import compare_ecgs
import json
from pprint import pprint
from multiprocessing import Pool
import tqdm
import argparse
import typing as t

# The lead_order list represents the standard sequence of leads in a 12-lead ECG
lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def process_ecg(row: pd.Series) -> t.Dict[str, t.Union[str, float]]:
    """
    Process an ECG and calculate error scores and normalized score.

    This function calculates the error scores and normalized score between the
    input ECG and target standard ECG. The error scores are calculated using the
    compare_ecgs function. The normalized score is calculated as max(0, 1 - digi_res / flat_res).

    :param row: One row from metadata file
    :return: Dictionary containing the path, flat_distance, ecg_distance, and normalized_score
    """
    try:
        # Extract digital ECG file path
        digi_raw_path = row["Image relative path"].split("_page")[0] + ".json"
        # Load leads and rhythms data
        leads = np.load(data_dir / "leads.npz")
        rhythms = np.load(data_dir / "rhythms.npz")

        # Extract leads for the specific ECG
        leads_part = leads[row["ECG ID"]]

        # Create target standard ECG dictionary
        target_ecg = {lead_order[i]: leads_part[:, i] for i in range(leads_part.shape[1])}

        # Add rhythm leads if present
        if row["ECG number of rhythm leads"]:
            rhythms_part = rhythms[row["ECG ID"]]
            for i in range(row["ECG number of rhythm leads"]):
                target_ecg[f"rhythm{i+1}"] = rhythms_part[:, i]

        # Create flat ECG dictionary (all leads are flat)
        flat_ecg = {k: v * 0 for k, v in target_ecg.items()}

        # Load digital ECG data
        digi_ecg_filename = digi_base_dir / digi_raw_path

        if digi_ecg_filename.exists():
            digi_ecg_raw = json.load(open(digi_ecg_filename))
            digi_ecg = {k: np.array(v["ecg"]) for k, v in digi_ecg_raw.items()}

            # Calculate error scores
            digi_res = compare_ecgs(digi_ecg, target_ecg, 320, 500)
            flat_res = compare_ecgs(flat_ecg, target_ecg, 500, 500)
        else:
            print(digi_ecg_filename, "not found", file=sys.stderr)
            # Calculate error scores when digital ECG file is not found
            flat_res = compare_ecgs(flat_ecg, target_ecg, 500, 500)
            digi_res = flat_res

    except Exception as e:
        custom_exception = RuntimeError("An error occurred while processing the ECG package.")
        raise custom_exception from e

    # Return results in a dictionary
    result = {
        "path": digi_raw_path,
        "flat_distance": flat_res,
        "ecg_distance": digi_res,
        "normalized_score": max(0, 1 - digi_res / flat_res),
    }
    print(result, file=sys.stderr)
    return result


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Calculate the normalized pixel distance score for ECGs")
    parser.add_argument("base_dir", help="Path to the base directory containing data and metadata", type=Path)
    parser.add_argument("digi_ecg_dir", help="Path to the digitized ECG directory", type=Path)
    parser.add_argument("output_csv", help="Path to the output CSV file", type=Path)
    args = parser.parse_args()

    # Set data directory and load metadata
    data_dir = args.base_dir / "data"
    meta_df = pd.read_csv(args.base_dir / "metadata.csv")

    # Filter metadata to only include the first page of ECGs
    filtered_meta = meta_df[meta_df["Image page"] == 0]
    # Set digitized ECG base directory
    digi_base_dir = args.digi_ecg_dir

    # Process ECGs using a process pool
    with Pool(12) as p:
        out = list(
            tqdm.tqdm(
                p.imap_unordered(process_ecg, (x[1] for x in filtered_meta.iterrows()), chunksize=1),
                total=len(filtered_meta),
                smoothing=0.05,
            )
        )

    # Convert results to a DataFrame and save to a CSV file
    df = pd.json_normalize(out)
    df.to_csv(args.output_csv)
