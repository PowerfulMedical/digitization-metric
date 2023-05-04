import pandas as pd
import argparse
import sys

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    df["prefix"] = df["path"].str.split('/').str[0]
    
    print("By dataset")
    by_prefix = df[["prefix", "normalized_score"]].groupby("prefix").mean().sort_values("normalized_score")
    print(by_prefix.to_string())
    print("--------")
    total = df["normalized_score"].mean()
    print("Total normalized pixel distance score: %.4f" % total)

