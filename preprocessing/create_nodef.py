import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    """Parse command line arguments

    Returns:
        args: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to the cleaned AWF traces",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--n_websites",
        type=int,
        default=100,
        help="Number of websites to consider",
    )
    parser.add_argument(
        "--n_traces",
        type=int,
        default=60,
        help="Number of traces to consider per website",
    )
    return parser.parse_args()


def main(args):

    print(f"Counting traces per websites in {args.in_path}")
    # walk through all folders in the input folder
    df = []
    for _, dirs, files in os.walk(args.in_path):
        for dir in dirs:
            # walk through all files in the current folder
            for _, dirs, files in os.walk(os.path.join(args.in_path, dir)):
                for file in files:
                    fpath = os.path.join(args.in_path, dir, file)
                    df.append((dir, fpath))

    df = pd.DataFrame(df, columns=["website", "path"])
    websites = df.groupby("website").count().sort_values("path", ascending=False)

    # select the top n websites
    websites = websites.head(args.n_websites)

    if websites.iloc[-1][0] < args.n_traces:
        raise ValueError(
            f"Not enough traces per website. Counts top {args.n_websites} websites:\n{websites}"
        )

    # select the top n traces per website
    df = df[df["website"].isin(websites.index)]
    df = df.groupby("website").head(args.n_traces)

    # copy the files to the output folder
    if not os.path.exists(args.out_path):
        print(f"[*] Writing to {args.out_path}")
        os.makedirs(args.out_path)

    for idx, website in tqdm(enumerate(websites.index), total=len(websites)):
        trace_paths = df[df["website"] == website]["path"].values
        for trace_idx, trace_path in enumerate(trace_paths):
            out_path = os.path.join(args.out_path, f"{idx}-{trace_idx}")
            trace = pd.read_csv(trace_path)
            np.savetxt(out_path, trace.values, fmt="%f\t%d")


if __name__ == "__main__":
    args = parse_args()
    main(args)
