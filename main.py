"""DeepSE-WF BER and MI estimation."""
import argparse
import logging
import os
import sys
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold as CV
from tabulate import tabulate

from datasets.data_utils import get_split, load_data
from models.model_utils import REPRESENTATIONS, train_models
from utils.knn import compute_distance, knn_ber, knn_mi
from utils.utils import get_args_parser

LOG_LVL = logging.INFO


def estimate_security(data, embeddings):
    """Estimate BER and MI for the test sets.

    Args:
        data (dict): Dictionary containing the train and test sets.
        embeddings (dict): Dictionary containing the embeddings for the train and test sets.

    Returns:
        ber (float): Bayes Error Rate.
        mi (float): Mutual Information.
    """
    est_ber = 1
    est_mi = 0
    errors = {}
    for representation in REPRESENTATIONS:
        d = compute_distance(
            embeddings[representation]["test1"],
            embeddings[representation]["test2"],
            args.knn_measure,
        )
        logging.info(f"Estimate kNN-Error for {representation}")
        ber1, knn_error1 = knn_ber(d, data["y_test1"], data["y_test2"], args.ber_k)
        mi = knn_mi(d, data["y_test1"], data["y_test2"], args.mi_k)

        d = compute_distance(
            embeddings[representation]["test2"],
            embeddings[representation]["test1"],
            args.knn_measure,
        )
        ber2, knn_error2 = knn_ber(d, data["y_test2"], data["y_test1"], args.ber_k)
        mi += knn_mi(d, data["y_test2"], data["y_test1"], args.mi_k)

        est_ber = min(est_ber, (ber1 + ber2) / 2)
        est_mi = max(est_mi, mi / 2)
        errors[representation] = (knn_error1 + knn_error2) / 2

    return est_ber, est_mi, errors


def main(args):
    total_start = timer()
    x, y = load_data(args.data_path, args)

    cv = CV(n_splits=args.k_fold, shuffle=True, random_state=42)

    results = {f"acc_{r}": [] for r in REPRESENTATIONS}
    results["ber"] = []
    results["mi"] = []
    for cv_count, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        start_cv = timer()
        logging.info(f"------------------- CV RUN {cv_count} OF {args.k_fold} -------------------")
        data = get_split(x, y, train_idx, test_idx)

        logging.info("Train Models:")
        embeddings, history = train_models(data=data, args=args)

        for representation in REPRESENTATIONS:
            if args.model != "tf":
                results[f"acc_{representation}"].append(history[representation]["test_acc"][-1])

        table = [
            [r, history[r]["train_acc"][-1], history[r]["val_acc"][-1], history[r]["test_acc"][-1]]
            for r in REPRESENTATIONS
        ]
        table = tabulate(
            table,
            headers=["Representation", "Train Acc", "Val Acc", "Test Acc"],
            tablefmt="github",
            floatfmt=".4f",
        )
        logging.info(f"Model performance:\n\n{table}\n")

        logging.info("Estimate Security:")
        table = []
        start = timer()
        ber, mi, errors = estimate_security(data, embeddings)
        results["ber"].append(ber)
        results["mi"].append(mi)

        table.append([ber, mi])

        end = timer()
        logging.info(f"Done after {timedelta(seconds=end - start)}")

        table = tabulate(
            table,
            headers=["BER", "MI"],
            tablefmt="github",
            floatfmt=".4f",
        )
        logging.info(f"Results for CV {cv_count}:\n\n{table}\n")

        if args.model == "tf":
            for rep in REPRESENTATIONS:
                logging.info(f"kNN-Error (k={args.ber_k}) for {rep} is: {errors[rep]:.3f}")
                results[f"acc_{rep}"].append(1 - errors[rep])

        end_cv = timer()

        logging.info(f"Time CV {cv_count}: {timedelta(seconds=end_cv-start_cv)}\n")

    total_end = timer()

    logging.info(f"Total time: {timedelta(seconds=total_end-total_start)}")

    table = [[k, np.mean(v), np.std(v)] for k, v in results.items()]

    table = tabulate(
        table,
        headers=["Value", "Mean", "Std"],
        tablefmt="github",
        floatfmt=".4f",
    )
    logging.info(f"Results:\n\n{table}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeepSE-WF",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file, level=LOG_LVL, format="%(asctime)s %(levelname)s %(message)s"
        )
    else:
        logging.basicConfig(
            stream=sys.stdout, level=LOG_LVL, format="%(asctime)s %(levelname)s %(message)s"
        )

    if args.gpu_id is not None:
        gpuid = args.gpu_id
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"  # select ID of GPU that shall be used

    if args.device == "cuda":
        logging.info(f"Using GPU: {gpuid} ({torch.cuda.is_available()})")

    hyperparams = tabulate(
        [
            [
                args.model,
                args.data_path.split("/")[-1],
                args.n_traces,
                args.epochs,
                args.dropout,
                args.embedding_size,
                args.knn_measure,
                args.ber_k,
                args.mi_k,
                args.device,
            ]
        ],
        headers=[
            "model",
            "data",
            "n_traces",
            "epochs",
            "dropout",
            "embedding_size",
            "measure",
            "ber_k",
            "mi_k",
            "device",
        ],
        tablefmt="github",
        floatfmt=".4f",
    )
    logging.info(f"Hyperparameters:\n\n{hyperparams}\n")

    main(args)
