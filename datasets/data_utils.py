import logging

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def load_data(data_path, args):
    """Load the Dataset.

    Args:
        data_path: Path to the .npz file containing the traces and labels
        args: Arguments

    Returns:
        x: Matrix (n_traces*n_classes x feature_length) containing the traces
        y: Array (n_traces*n_classes) containing the labels
    """
    # Load data
    logging.info("Loading data...")
    data = np.load(data_path)
    x = data["traces"]
    y = data["labels"]

    # Convert data as float32 type
    x = x.astype("float32")
    y = y.astype("float32")

    # reduce dataset if necessary
    if args.n_traces * args.n_websites < len(y):
        x, _, y, _ = train_test_split(
            x, y, train_size=args.n_traces * args.n_websites, stratify=y, random_state=42
        )
    logging.info("\tdone.")

    return x, y


def get_split(x, y, train_idx, test_idx):
    """Get the validation splits of x and y.

    Args:
        x: traces matrix
        y: label array
        train_idx: index values for train data
        test_idx: index values for test data

    Returns:
        data: Dictionary containing train, test1 and test2
              data where a new axis is added for the traces
    """
    # get correct split of data
    x_train = np.array([v for i, v in enumerate(x) if i in train_idx]).astype("float32")
    x_test = np.array([v for i, v in enumerate(x) if i in test_idx]).astype("float32")

    y_train = np.array([v for i, v in enumerate(y) if i in train_idx]).astype("float32")
    y_test = np.array([v for i, v in enumerate(y) if i in test_idx]).astype("float32")

    # split test into tes1 and test2
    x_test1, x_test2, y_test1, y_test2 = train_test_split(
        x_test, y_test, test_size=0.5, stratify=y_test, shuffle=True, random_state=42
    )

    # we need a [Length x 1] x n shape as input to the CNN (Tensorflow)
    x_train = x_train[:, np.newaxis, :].astype("float32")
    x_test1 = x_test1[:, np.newaxis, :].astype("float32")
    x_test2 = x_test2[:, np.newaxis, :].astype("float32")

    data = {
        "x_train": x_train,
        "x_test1": x_test1,
        "x_test2": x_test2,
        "y_train": y_train,
        "y_test1": y_test1,
        "y_test2": y_test2,
    }

    logging.debug(f"Train shape: {x_train.shape}")
    logging.debug(f"Test1 shape: {x_test1.shape}")
    logging.debug(f"Test2 shape: {x_test2.shape}")

    return data


def get_dataloader(traces, labels, is_timing, is_training, args):
    """Get the dataloader for the given data.

    Args:
        traces: Traces matrix
        labels: Label array
        is_timing: True if the model is for timing
        is_training: True if the model is for training
        args: Arguments passed to the script

    Returns:
        dataloader: The dataloader
    """
    if args.model == "tf":
        from datasets.dataset import TripletDataset

        dataset = TripletDataset(data=traces, labels=labels, timing=is_timing)

    elif args.model in ["df", "awf_cnn", "var_cnn"]:
        from datasets.dataset import DefaultDataset

        dataset = DefaultDataset(data=traces, labels=labels, timing=is_timing)

    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=is_training, num_workers=args.num_workers
    )
