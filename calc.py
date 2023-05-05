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

lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def process_ecg(package):
    try:
        _, r = package
        digi_raw_path = r["Image relative path"].split("_page")[0] + ".json"
        leads = np.load(data_dir / "leads.npz")
        rhythms = np.load(data_dir / "rhythms.npz")

        leads_part = leads[r["ECG ID"]]

        gold_ecg = {lead_order[i]: leads_part[:,i] for i in range(leads_part.shape[1])}
        if r["ECG number of rhythm leads"]:
            rhythms_part = rhythms[r["ECG ID"]]
            for i in range(r["ECG number of rhythm leads"]):
                gold_ecg[f"rhythm{i+1}"] = rhythms_part[:,i]

        flat_ecg = {k: v*0 for k, v in gold_ecg.items()}

        digi_ecg_filename = digi_base_dir / digi_raw_path

        if digi_ecg_filename.exists():
            digi_ecg_raw = json.load(open(digi_ecg_filename))
            digi_ecg = {k: np.array(v["ecg"]) for k, v in digi_ecg_raw.items()}

            digi_res = compare_ecgs(digi_ecg, gold_ecg, 320, 500)
            flat_res = compare_ecgs(flat_ecg, gold_ecg, 500, 500)
        else:
            print(digi_ecg_filename, "not found", file=sys.stderr)
            # TODO: warning
            flat_res = compare_ecgs(flat_ecg, gold_ecg, 500, 500)
            digi_res = flat_res

    except:
        digi_res = 2
        flat_res = 1
        raise

    print(digi_raw_path, digi_res, flat_res, 1 - digi_res / flat_res, file=sys.stderr)
    return {"path": digi_raw_path, "flat_distance": flat_res, "ecg_distance": digi_res, "normalized_score": max(0, 1 - digi_res / flat_res)}

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the normalized pixel distance score for ECGs')
    parser.add_argument('base_dir', help='Path to the base directory containing data and metadata', type=Path)
    parser.add_argument('digi_ecg_dir', help='Path to the digitized ECG directory', type=Path)
    parser.add_argument('output_csv', help='Path to the output CSV file', type=Path)
    args = parser.parse_args()

    data_dir = args.base_dir / "data"
    meta_df = pd.read_csv(args.base_dir / "metadata.csv")
    filtered_meta = meta_df[meta_df["Image page"] == 0]
    digi_base_dir = args.digi_ecg_dir

    with Pool(12) as p:
        out = list(tqdm.tqdm(p.imap_unordered(process_ecg, filtered_meta.iterrows(), chunksize=1), total=len(filtered_meta), smoothing=0.05))

    df = pd.json_normalize(out)
    df.to_csv(args.output_csv)

