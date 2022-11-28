import argparse
import os
import tarfile

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
        help="Path to the .tar.gz file containing the AWF traces",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to the output folder",
    )
    return parser.parse_args()


def get_trace(full_info: pd.DataFrame, include_length: bool = False):
    """Extracts trace from full info dataframe

    Args:
        full_info: full info dataframe
        include_length: whether to include length in trace

    Returns:
        trace: cleaned trace
    """
    full_info = full_info.sort_values("timestamp")
    try:
        t0 = full_info.iloc[0, 1]
    except Exception:
        print(full_info)
        return pd.DataFrame()

    full_info["timestamp"] = full_info["timestamp"] - t0
    if full_info["ack"].iloc[0]:
        return pd.DataFrame()
    tor_cells = full_info[full_info["ack"] == 0]

    cols = ["timestamp", "direction"]
    if include_length:
        cols = ["timestamp", "direction", "length"]
    return tor_cells[cols]


def extract_tar(file, args):
    tar = tarfile.open(file, "r:gz")

    print(f"[*] Extracting {file} ...")
    for member in tqdm(tar.getmembers()):
        if member.isfile():
            file = tar.extractfile(member)
            if file is not None:
                full_info = pd.read_csv(file, delimiter=";")
                trace = get_trace(full_info)

                if trace.empty:
                    continue

                fname = member.name.split("/")[-1]
                folder = member.name.split("/")[1]

                if not os.path.exists(os.path.join(args.out_path, folder)):
                    os.makedirs(os.path.join(args.out_path, folder))

                trace.to_csv(os.path.join(args.out_path, folder, fname), index=False, header=False)


def main(args):
    """Reads all tar files from a folder and writes cleaned traces to output folder"""

    print(f"[*] Extracting traces from {args.in_path} to {args.out_path}.")
    for file in os.listdir(args.in_path):
        if file.endswith(".tar.gz"):
            extract_tar(os.path.join(args.in_path, file), args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
