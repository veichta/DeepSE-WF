"""DeepSE-WF BER and MI estimation."""
import argparse
import logging
import math
import os
import sys
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold as CV
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from wandb.keras import WandbCallback

import wandb
from utils.tf_knn import compute_distance, knn_ber, knn_mi

REPRESENTATIONS = ["timing", "directional"]

LOG_LVL = logging.INFO


def get_args_parser():
    # fmt: off
    """Get the arguments parser.

    Returns:
        parser: Arguments parser
    """
    parser = argparse.ArgumentParser("DeepSE-WF BER and MI estiamtion", add_help=False)

    # Data and Setup
    parser.add_argument(
        "--data_path",
        type=str, required=True,
        help="Path to data folder which should contain traces.npy and labels.npy",
    )
    parser.add_argument(
        "--n_traces",
        type=int, required=True,
        help="Number of traces to use per website",
    )
    parser.add_argument(
        "--k_fold",
        default=5, type=int,
        help="Number of cross-validation runs"
    )
    parser.add_argument(
        "--num_classes",
        default=100, type=int,
        help="Number of websites in the dataset"
    )

    # Training
    parser.add_argument(
        "--epochs",
        default=50, type=int,
        help="Number of epochs used to train the attack model",
    )
    parser.add_argument(
        "--batch_size",
        default=128, type=int,
        help="Batch size used to train the attack model",
    )
    parser.add_argument(
        "--feature_length",
        default=5000, type=int,
        help="Length of each packet sequence",
    )
    parser.add_argument(
        "--early_stopping",
        default=False, type=bool,
        help="Use early stopping to avoid overfitting the model",
    )
    parser.add_argument(
        "--model",
        default="df", type=str,
        help="Model trained for the embeddings"
    )
    parser.add_argument(
        "--units",
        default=256, type=bool,
        help="Units used in awf lstm model."
    )
    parser.add_argument(
        "--dropout",
        default=0.1, type=float,
        help="Dropout used in awf models."
    )
    parser.add_argument(
        "--embedding_size",
        default=512, type=int,
        help="Embedding size used in models."
    )

    # Logging
    parser.add_argument(
        "--log_file",
        default="", type=str,
        help="File for logging"
    )
    parser.add_argument(
        "--verbose",
        default=0, type=int,
        help="Wheater to show the training progress"
    )
    parser.add_argument(
        "--defense",
        default="NoDef", type=str,
        help="Defense which is evaluated (used for logging only).",
    )

    # kNN
    parser.add_argument(
        "--knn_measure",
        default="squared_l2", choices=["squared_l2", "cosine"], type=str,
        help="Measure used for KNN distance matrix computation",
    )
    parser.add_argument(
        "--mi_k",
        default=5, type=int,
        help="Value for K of KNN in Mutual Information Estimation",
    )

    parser.add_argument(
        "--gpu_id",
        default=7, type=int,
        help="GPU id to use.",
    )
    # fmt: on
    return parser


def load_data(data_path, args):
    """Load the Dataset.

    Args:
        data_path: Data folder which should contain traces.npy and labels.npy
        args: Arguments

    Returns:
        x: Matrix (n_traces*n_classes x feature_length) containing the traces
        y: Array (n_traces*n_classes) containing the labels
    """
    trace_path = f"{data_path}/traces.npy"
    label_path = f"{data_path}/labels.npy"

    # Load data
    logging.info("Loading data...")
    x = np.load(trace_path)
    y = np.load(label_path)

    # Convert data as float32 type
    x = x.astype("float32")
    y = y.astype("float32")

    # reduce dataset if necessary
    if args.n_traces * args.num_classes < len(y):
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=args.n_traces * args.num_classes,
            stratify=y,
            random_state=42,
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
    x_train = x_train[:, :, np.newaxis].astype("float32")
    x_test1 = x_test1[:, :, np.newaxis].astype("float32")
    x_test2 = x_test2[:, :, np.newaxis].astype("float32")

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


def train_models(x, y, cv_count, args):
    """This function trains the attack for timeing as well as directional traces.

    Args:
        x: Traces
        y: Webpage labels
        cv_count: Number of cross validation folds
        args: Arguments passed to the script

    Raises:
        NotImplementedError: If the attack is not implemented.

    Returns:
        models: Dictionary containing the trained models.
    """
    models = {}

    for representation in REPRESENTATIONS:
        train_start = timer()

        x_train = x
        y_train = y
        if representation == "directional":  # create directional traces
            x_train = np.sign(x)

        wandb.init(
            project="DeepSE-WF",
            entity="bayes_error_security_wf",
            config=vars(args),
            group=WANDB_GROUPNAME,
            job_type=f"train-{representation}",
            name=f"train-{representation}-fold{cv_count}",
        )

        # build model
        if args.model == "df":
            from tf_models.tf_df import build_df_model as build_model

            model = build_model(
                input_shape=(args.feature_length, 1), classes=args.num_classes, args=args
            )
        elif args.model == "awf_cnn":
            from tf_models.tf_awf import build_awf_cnn_model as build_model

            model = build_model(
                input_shape=(args.feature_length, 1), classes=args.num_classes, args=args
            )
        elif args.model == "var_cnn":
            from tf_models.tf_varcnn import build_var_cnn_model as build_model

            time = representation != "directional"
            model = build_model(
                input_shape=(args.feature_length, 1), classes=args.num_classes, time=time, args=args
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")

        # optimizer
        optimizer = tf.keras.optimizers.Adamax(
            learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
        )

        # compile model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        # early stopping
        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=0,
        #     patience=10,
        #     verbose=0,
        #     mode="auto",
        #     baseline=None,
        #     restore_best_weights=True,
        # )
        wb_callback = WandbCallback(save_model=False, save_graph=False)
        cb = [wb_callback]
        # fit model
        model.fit(
            x_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=args.verbose,
            callbacks=cb,
            validation_split=0.1,
        )

        models[representation] = model

        train_end = timer()
        logging.info(f" {representation} model ({timedelta(seconds=train_end-train_start)})")

        wandb.join()

    return models


def eval_model(models, data, args):
    """This function evaluates the attack.

    Args:
        models: Dictionary containing trained DF and Tik-Tok attack models
        data: Dictionary containing the train and test data
        args: Arguments passed to the script
    """
    x_train, y_train = data["x_train"], data["y_train"]
    x_test1, y_test1 = data["x_test1"], data["y_test1"]
    x_test2, y_test2 = data["x_test2"], data["y_test2"]

    table = [[representation] for representation in REPRESENTATIONS]  # table for logging

    for idx, representation in enumerate(REPRESENTATIONS):
        if representation == "directional":
            x_train = np.sign(x_train)
            x_test1 = np.sign(x_test1)
            x_test2 = np.sign(x_test2)

        model = models[representation]
        score_train = model.evaluate(x_train, y_train, verbose=args.verbose)
        score_test1 = model.evaluate(x_test1, y_test1, verbose=args.verbose)
        score_test2 = model.evaluate(x_test2, y_test2, verbose=args.verbose)

        table[idx].append(score_train[1])
        table[idx].append(score_test1[1])
        table[idx].append(score_test2[1])
        wandb.log(
            {
                f"train_acc_{representation}": score_train[1],
                f"test_acc_{representation}": (score_test1[1] + score_test2[1]) / 2.0,
            }
        )

    table = tabulate(
        table, headers=["Model Features", "Train Acc", "Test1 Acc", "Test2 Acc"], tablefmt="github"
    )
    logging.info(f"Evaluation Results:\n\n{table}\n")


def extract_embeddings(models, data, args):
    """This function extracts the latent space embeddings of the traces.

    Args:
        models: Dictionary containing trained attack models
        data: Dictionary containing the train and test data
        args: Arguments

    Returns:
        embeddings: Dictionary containing the embeddings.
    """
    embeddings = {}

    for representation in REPRESENTATIONS:
        model = models[representation]

        # remove the last layers of the model and save it
        if args.model == "var_cnn":
            model.layers.pop()  # ReLu
            model.layers.pop()  # Dropout
            model.layers.pop()  # Dense
            model.layers.pop()  # Softmax
        else:
            model.pop()  # ReLU
            model.pop()  # Dropout
            model.pop()  # Dense
            model.pop()  # Softmax

        # extract embeddings
        for model_features_load in REPRESENTATIONS:
            x_test1 = data["x_test1"]
            x_test2 = data["x_test2"]

            if model_features_load == "directional":
                x_test1 = np.sign(data["x_test1"])
                x_test2 = np.sign(data["x_test2"])

            embeddings[f"test1-{representation}-{model_features_load}"] = model.predict(
                x_test1, batch_size=args.batch_size, verbose=args.verbose
            )
            embeddings[f"test2-{representation}-{model_features_load}"] = model.predict(
                x_test2, batch_size=args.batch_size, verbose=args.verbose
            )

    return embeddings


def calc_ber(data, embeddings, args):
    """This function estimates the Bayes Error Rate for all embeddings and raw representations.

    Args:
        data: Dictionary containing the train and test data
        embeddings: Dictionary containing the extraced embddings of the traces
        args: Arguments

    Returns:
        Bayes Error Rate results: Dictionary containing the estimated Bayes Error Rate.
    """
    ber_results = {}

    for representation in REPRESENTATIONS:
        for model_features_load in ["raw"] + REPRESENTATIONS:
            logging.debug(f"Combination: {representation}-{model_features_load}")

            # loading correct combination of data
            if model_features_load == "raw":
                x_test1, x_test2 = data["x_test1"], data["x_test2"]
                y_test1, y_test2 = data["y_test1"], data["y_test2"]

                if representation == "directional":
                    x_test1, x_test2 = np.sign(x_test1), np.sign(x_test2)
            else:
                x_test1, x_test2 = (
                    embeddings[f"test1-{representation}-{model_features_load}"],
                    embeddings[f"test2-{representation}-{model_features_load}"],
                )
                y_test1, y_test2 = data["y_test1"], data["y_test2"]

            x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1])
            x_test2 = x_test2.reshape(x_test2.shape[0], x_test2.shape[1])

            # x_test1 for train
            dist = compute_distance(x_test1, x_test2, args.knn_measure)
            logging.debug("Test1")
            knn_est1 = knn_ber(dist, y_test1, y_test2)
            logging.debug(f"BER: {knn_est1}")

            # x_test2 for train
            dist = compute_distance(x_test2, x_test1, args.knn_measure)
            logging.debug("Test2")
            knn_est2 = knn_ber(dist, y_test2, y_test1)
            logging.debug(f"BER: {knn_est2}\n")

            ber_results[f"{representation}-{model_features_load}"] = (knn_est1 + knn_est2) / 2

    return ber_results


def calc_mi(data, embeddings, args):
    """This function estimates the Mutual Information for all embeddings and raw representations.

    Args:
        data: Dictionary containing the train and test data
        embeddings: Dictionary containing the extraced embddings of the traces
        args: Arguments

    Returns:
        Mutual Information results: Dictionary containing the estimated Mutual Information.
    """
    mi_results = {}
    y_test1 = np.array([int(y) for y in data["y_test1"]])
    y_test2 = np.array([int(y) for y in data["y_test2"]])

    for representation in REPRESENTATIONS:
        for model_features_load in ["raw"] + REPRESENTATIONS:
            logging.debug(f"Combination: {representation}-{model_features_load}")

            # loading correct combination of data
            if model_features_load == "raw":
                x_test1, x_test2 = data["x_test1"], data["x_test2"]

                if representation == "directional":
                    x_test1, x_test2 = np.sign(x_test1), np.sign(x_test2)
            else:
                x_test1, x_test2 = (
                    embeddings[f"test1-{representation}-{model_features_load}"],
                    embeddings[f"test2-{representation}-{model_features_load}"],
                )

            x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1])
            x_test2 = x_test2.reshape(x_test2.shape[0], x_test2.shape[1])

            # x_test1 for train
            dist = compute_distance(x_test1, x_test2, args.knn_measure)
            logging.debug("Test1")
            knn_est1 = knn_mi(dist, y_test1, y_test2, args.mi_k) * np.log2(math.e)
            logging.debug(f"MI: {knn_est1}")

            # x_test2 for train
            dist = compute_distance(x_test2, x_test1, args.knn_measure)
            logging.debug("Test2")
            knn_est2 = knn_mi(dist, y_test2, y_test1, args.mi_k) * np.log2(math.e)
            logging.debug(f"MI: {knn_est2}\n")

            mi_results[f"{representation}-{model_features_load}"] = (knn_est1 + knn_est2) / 2

    return mi_results


def main(args):
    """Main function for estimating BER and MI.

    Args:
        args: Arguments
    """

    total_start = timer()
    cv = CV(n_splits=args.k_fold, shuffle=True, random_state=42)

    x, y = load_data(args.data_path, args)

    ber_estimations = []
    mi_estimations = []

    for cv_count, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        start_cv = timer()
        logging.info(f"------------------- CV RUN {cv_count} OF {args.k_fold} -------------------")
        data = get_split(x, y, train_idx, test_idx)

        # train model
        logging.info("Train Models:")
        models = train_models(data["x_train"], data["y_train"], cv_count, args)

        wandb.init(
            project="DeepSE-WF",
            entity="bayes_error_security_wf",
            config=vars(args),
            group=WANDB_GROUPNAME,
            job_type="eval",
            name=f"eval-fold{cv_count}",
        )

        eval_model(models, data, args)

        # extract all embeddings
        logging.info("Extracting Embeddings")
        embeddings = extract_embeddings(models, data, args)

        # estimate ber and mi
        logging.info("Estimating Security")
        ber_results = calc_ber(data, embeddings, args)
        ber_estimations.append(min(ber_results.values()))

        mi_results = calc_mi(data, embeddings, args)
        mi_estimations.append(max(mi_results.values()))

        end_cv = timer()
        logging.info(f"Bayes Error Rate is {min(ber_results.values()):.4f}")
        logging.info(f"Mutual Information is {max(mi_results.values()):.4f}")
        logging.info(f"Time CV {cv_count}: {timedelta(seconds=end_cv-start_cv)}\n\n")

        wandb.log({"ber": min(ber_results.values()), "mi": max(mi_results.values())})

        wandb.join()

    wandb.init(
        project="DeepSE-WF",
        entity="bayes_error_security_wf",
        config=vars(args),
        group=WANDB_GROUPNAME,
        job_type="eval",
        name="summary",
    )

    total_end = timer()
    logging.info("----------------------------- Final Results -----------------------------")

    logging.info(
        f"Bayes Error Rate is {np.mean(ber_estimations):.4f} (+-{np.std(ber_estimations):.4f})"
    )
    logging.info(
        f"Mutual Information is {np.mean(mi_estimations):.4f} (+-{np.std(mi_estimations):.4f})"
    )
    logging.info(f"Total Time: {timedelta(seconds=total_end-total_start)}\n\n")

    wandb.log({"overall_ber": np.mean(ber_estimations), "overall_mi": np.mean(mi_estimations)})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeepSE-WF",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=LOG_LVL)
    else:
        logging.basicConfig(stream=sys.stdout, level=LOG_LVL)

    WANDB_GROUPNAME = f"{args.model}-{args.defense}-{wandb.util.generate_id()}"

    gpuid = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"  # select ID of GPU that shall be used

    max_memory = 1024 * 11
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # tf.config.set_logical_device_configuration(
            #    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=max_memory)]
            # )
            # logical_gpus = tf.config.list_logical_devices("GPU")

            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logging.info(e)

    logging.info(f"Using GPU: {gpuid} ({len(gpus) > 0})")

    logging.info(f"Data Path: {args.data_path}")
    logging.info(f"Number of Traces: {args.n_traces}")
    logging.info(f"Number of Epochs: {args.epochs}")

    main(args)
