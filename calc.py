import sys
import numpy as np
from pathlib import Path
import pandas as pd
from digi_eval import compare_ecgs
import json
from pprint import pprint
from multiprocessing import Pool
import tqdm

lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def raw_ecg_to_formatted(raw_ecg, format_str, num_pages, page_cols):
    # TODO: verify this a lot
    out = {}
    for page_no in range(num_pages):
        leads_per_page = 12 // num_pages
        page_range_start = page_no * leads_per_page
        page_range_end = page_range_start + leads_per_page
        for col_no in range(page_cols):
            leads_per_col = leads_per_page // page_cols
            samples_per_col = raw_ecg.shape[0] // page_cols
            range_start = page_range_start + col_no * leads_per_col
            range_end = range_start + leads_per_col
            time_start = samples_per_col * col_no
            time_end = time_start + samples_per_col

            for lead_no in range(range_start, range_end):
                out[lead_order[lead_no]] = raw_ecg[time_start:time_end, lead_no]

    if "+1R" in format_str:
        out["rhythm1"] = raw_ecg[:,1]
    if "+3R" in format_str:
        out["rhythm1"] = raw_ecg[:,1]
        out["rhythm2"] = raw_ecg[:,6]
        out["rhythm3"] = raw_ecg[:,10]
    return out

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
    base_dir = Path(sys.argv[1])
    data_dir = base_dir / "data"
    meta_df = pd.read_csv(base_dir / "metadata.csv")
    filtered_meta = meta_df[meta_df["Image page"] == 0]


    digi_base_dir = Path(sys.argv[2])

    with Pool(12) as p:
        out = list(tqdm.tqdm(p.imap_unordered(process_ecg, filtered_meta.iterrows(), chunksize=1), total=len(filtered_meta), smoothing=0.05))

    df = pd.json_normalize(out)
    df.to_csv(sys.argv[3])

