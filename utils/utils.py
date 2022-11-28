import argparse


def get_args_parser():
    """Get the arguments parser.

    Returns:
        parser: Arguments parser
    """
    parser = argparse.ArgumentParser("DeepSE-WF BER and MI estiamtion", add_help=False)

    # Data and Setup
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .npz file containing the traces and labels",
    )
    parser.add_argument(
        "--n_traces",
        type=int,
        required=True,
        help="Number of traces to use per website",
    )
    parser.add_argument(
        "--k_fold",
        default=5,
        type=int,
        help="Number of cross-validation runs",
    )
    parser.add_argument(
        "--n_websites",
        default=100,
        type=int,
        help="Number of websites in the dataset",
    )

    # Training
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to use for training",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs used to train the attack model",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size used to train the attack model",
    )
    parser.add_argument(
        "--feature_length",
        default=5000,
        type=int,
        help="Length of each packet sequence",
    )
    parser.add_argument(
        "--model",
        default="df",
        choices=["df", "awf_cnn", "var_cnn", "tf"],
        help="Model trained for the embeddings",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout used in awf models.",
    )
    parser.add_argument(
        "--embedding_size",
        default=512,
        type=int,
        help="Embedding size used in models.",
    )

    # Logging
    parser.add_argument(
        "--log_file",
        default="",
        type=str,
        help="File for logging",
    )

    # kNN
    parser.add_argument(
        "--knn_measure",
        default="squared_l2",
        choices=["squared_l2", "cosine"],
        type=str,
        help="Measure used for KNN distance matrix computation",
    )
    parser.add_argument(
        "--mi_k",
        default=5,
        type=int,
        help="Value for K of KNN in Mutual Information Estimation",
    )
    parser.add_argument(
        "--ber_k",
        default=1,
        type=int,
        help="Value for K of KNN in Bayes Error Estimation",
    )

    parser.add_argument(
        "--gpu_id",
        default=None,
        type=int,
        help="GPU id to use.",
    )
    return parser
