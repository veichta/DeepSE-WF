import argparse
import os

import numpy as np
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('DeepSE-WF BER and MI estiamtion', add_help=False)
    
    # Data and Setup
    parser.add_argument('--trace_path', type=str, required=True, 
        help="Path to data folder which should contain traces.npy and labels.npy")

    parser.add_argument('--save_path', type=str, required=True, 
        help="Path to data folder which should contain traces.npy and labels.npy")

    parser.add_argument('--n_traces', type=int, required=True, 
        help="Number of traces to use per website")

    parser.add_argument('--num_classes', type=int,  required=True,
        help="Number of websites in the dataset")

    parser.add_argument('--feature_length', default=5000, type=int, 
        help="Length of each packet sequence")

    return parser

# make all traces the same length
def pad_or_truncate(some_list, target_len):
    return np.concatenate((some_list[:target_len], np.array([0]*(target_len - len(some_list))) ))

def main(args):
    # Create Directories if necessary
    if not os.path.exists(args.save_path):
        try:
            os.makedirs(args.save_path)
        except OSError:
            print("[*] Creation of the directory %s failed" % args.save_path)

        print("[*] Created the directory(s): %s" % args.save_path)
    
    X = []
    y = []
   
    # Go over Traces and encode in the right format
    for w in tqdm(range(args.num_classes)):
        for t in range(args.n_traces):
            trace_file = '{}-{}'.format(w, t)
            trace = np.loadtxt(os.path.join(args.trace_path, trace_file))


            assert len(np.shape(trace)) > 1, "Trace should be in time \'tab\' direction format."

            trace = np.sign(trace[:, 1]) * trace[:, 0]

            trace = pad_or_truncate(trace, args.feature_length)

            X.append(trace)
            y.append(w)

    X = np.array(X)
    y = np.array(y)

    print('[*] Data shape {}'.format(X.shape))
    np.save(os.path.join(args.save_path, 'traces'), X)
    np.save(os.path.join(args.save_path, 'labels'), y)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Utility to create the dataset used in DeepSE-WF', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)