import argparse
import os

import numpy as np
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("Prepare traces for DeepSE-WF", add_help=False)

    # Data and Setup
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to data folder which should contain traces.npy and labels.npy",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to .npz file containing the traces and labels",
    )

    parser.add_argument(
        "--n_traces", type=int, required=True, help="Number of traces to use per website"
    )

    parser.add_argument(
        "--n_websites", type=int, required=True, help="Number of websites in the dataset"
    )

    parser.add_argument(
        "--feature_length", default=5000, type=int, help="Length of each packet sequence"
    )

    return parser


# make all traces the same length
def pad_or_truncate(some_list, target_len):
    return np.concatenate((some_list[:target_len], np.array([0] * (target_len - len(some_list)))))


def main(args):
    if not args.out_path.endswith(".npz"):
        raise ValueError("Data path must end with .npz")

    # Create Directories if necessary
    folder = os.path.dirname(args.out_path)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            print(f"[*] Creation of the directory {folder} failed")

        print(f"[*] Created the directory(s): {folder}")

    X = []
    y = []

    # Go over Traces and encode in the right format
    for w in tqdm(range(args.n_websites)):
        for t in range(args.n_traces):
            trace_file = f"{w}-{t}"
            trace = np.loadtxt(os.path.join(args.in_path, trace_file))

            assert len(np.shape(trace)) > 1, "Trace should be in time 'tab' direction format."

            trace = np.sign(trace[:, 1]) * trace[:, 0]  # convert to signed time

            trace = pad_or_truncate(trace, args.feature_length)

            X.append(trace)
            y.append(w)

    X = np.array(X)
    y = np.array(y)

    print(f"[*] Data shape {X.shape}")
    np.savez_compressed(args.out_path, traces=X, labels=y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Utility to create the dataset used in DeepSE-WF", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    main(args)
