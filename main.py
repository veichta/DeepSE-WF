import argparse
import logging
import sys
import os
from tokenize import group

import numpy as np
import math

import tensorflow as tf

from tabulate import tabulate
from timeit import default_timer as timer
from datetime import timedelta

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import StratifiedKFold as CV
from sklearn.model_selection import train_test_split

from utils.knn import compute_distance, knn_ber, knn_mi


REPRESENTATIONS = ['timing', 'directional']

LOG_LVL = logging.INFO


def get_args_parser():
    parser = argparse.ArgumentParser('DeepSE-WF BER and MI estiamtion', add_help=False)
    
    # Data and Setup
    parser.add_argument('--data_path', type=str, required=True, 
        help="Path to data folder which should contain traces.npy and labels.npy")
    parser.add_argument('--n_traces', type=int, required=True, 
        help="Number of traces to use per website")
    parser.add_argument('--k_fold', default=5, type=int, 
        help="Number of cross-validation runs")
    parser.add_argument('--num_classes', default=100, type=int, 
        help="Number of websites in the dataset")

    # Training
    parser.add_argument('--epochs', default=50, type=int, 
        help="Number of epochs used to train the attack model")
    parser.add_argument('--batch_size', default=128, type=int, 
        help="Batch size used to train the attack model")
    parser.add_argument('--feature_length', default=5000, type=int, 
        help="Length of each packet sequence")
    parser.add_argument('--model', default="df", type=str, 
        help="Model trained for the embeddings")
    parser.add_argument('--early_stopping', default=False, type=bool, 
        help="Use early stopping to avoid overfitting the model")
    parser.add_argument('--defense', default="NoDef", type=str, 
        help="Defense which is evaluated (used for logging only).")
        

    # Logging
    parser.add_argument('--log_file', default='', type=str, 
        help="File for logging")
    parser.add_argument('--verbose', default=0, type=int, 
        help="Wheater to show the training progress")

    # kNN
    parser.add_argument('--knn_measure', default="squared_l2", choices=["squared_l2", "cosine"], type=str, 
        help="Measure used for KNN distance matrix computation")
    parser.add_argument('--mi_k', default=5, type=int, 
        help="Value for K of KNN in Mutual Information Estimation")
    
    return parser


def load_data(data_path, args):
    """Load the Dataset.

    Args:
        data_path: Data folder which should contain traces.npy and labels.npy
    Returns:
        X: Matrix (n_traces*n_classes x feature_length) containing the traces
        y: Array (n_traces*n_classes) containing the labels
    """
    
    trace_path = "{}/traces.npy".format(data_path)
    label_path = "{}/labels.npy".format(data_path)

    # Load data
    X = np.load(trace_path)
    y = np.load(label_path)
    
    # Convert data as float32 type
    X = X.astype('float32')
    y = y.astype('float32')

    # reduce dataset if necessary
    if args.n_traces * args.num_classes < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=args.n_traces * args.num_classes, stratify=y, random_state=42)

    return X, y


def get_split(X, y, train_idx, test_idx):
    """Get the validation splits of X and y.

    Args:
        X: traces matrix
        y: label array
        train_idx: index values for train data
        test_idx: index values for test data
    Returns:
        data: Dictionary containing train, test1 and test2 data where a new axis is added for the traces
    """
    # get correct split of data
    X_train = np.array([v for i, v in enumerate(X) if i in train_idx]).astype('float32')
    X_test = np.array([v for i, v in enumerate(X) if i in test_idx]).astype('float32')
    
    y_train = np.array([v for i, v in enumerate(y) if i in train_idx]).astype('float32')
    y_test = np.array([v for i, v in enumerate(y) if i in test_idx]).astype('float32')

    # split test into tes1 and test2
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, shuffle=True, random_state=42)

    # we need a [Length x 1] x n shape as input to the CNN (Tensorflow)
    X_train = X_train[:, :,np.newaxis].astype('float32')
    X_test1 = X_test1[:, :,np.newaxis].astype('float32')
    X_test2 = X_test2[:, :,np.newaxis].astype('float32')

    data = {
        "X_train": X_train, 
        "X_test1": X_test1,
        "X_test2": X_test2, 
        "y_train": y_train, 
        "y_test1": y_test1, 
        "y_test2": y_test2
    }

    logging.debug(f'Train shape: {X_train.shape}')
    logging.debug(f'Test1 shape: {X_test1.shape}')
    logging.debug(f'Test2 shape: {X_test2.shape}')

    return data


def build_df_model(input_shape, classes):
    """This function builds the df attack model.

    Args:
        input_shape: Shape of the input shape i.e. the trace
        classes: Number of classes in the dataset
    Returns:
        model: Tensorflow keras sequential model which implements the DF attack neural network
    """

    m = tf.keras.Sequential()

    for i, f in enumerate([32, 64, 128, 256]):
      m.add(tf.keras.layers.Conv1D(filters=f, kernel_size=8, input_shape=input_shape, strides=1, padding='same'))
      m.add(tf.keras.layers.BatchNormalization(axis=-1))
      if i == 0:
        m.add(tf.keras.layers.ELU(alpha=1.0))
      else:
        m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Conv1D(filters=f, kernel_size=8, input_shape=input_shape, strides=1, padding='same'))
      m.add(tf.keras.layers.BatchNormalization(axis=-1))
      if i == 0:
        m.add(tf.keras.layers.ELU(alpha=1.0))
      else:
        m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPool1D(8, 4, padding='same'))
      m.add(tf.keras.layers.MaxPool1D(8, 4, padding='same'))
      m.add(tf.keras.layers.Dropout(0.2))

    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    m.add(tf.keras.layers.Dropout(0.7))

    m.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    m.add(tf.keras.layers.Dropout(0.5))

    m.add(tf.keras.layers.Dense(classes, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)))
    m.add(tf.keras.layers.Activation('softmax'))

    m.build([None, input_shape])

    return m


def train_models(X, y, cv_count, args):
    """This function trains the DF and Tik-Tok attack.

    Args:
        input_shape: Shape of the input shape i.e. the trace
        classes: Number of classes in the dataset
    Returns:
        models: Dictionary containing the trained DF and Tik-Tok models as Tensorflow keras sequential.
    """
    models = {}

    for representation in REPRESENTATIONS:
        train_start = timer()

        X_train = X
        y_train = y
        if representation == "directional": # create directional traces
            X_train = np.sign(X)

        wandb.init(
            project="DeepSE-WF", 
            entity="bayes_error_security_wf",
            config=vars(args),
            group=os.environ["WANDB_GROUPNAME"],
            job_type=f"train-{representation}",
            name=f"eval-{representation}-fold{cv_count}"
        )
        
        # build model
        model = build_df_model(input_shape=(args.feature_length,1), classes=args.num_classes)
        
        # optimizer
        #optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # compile model
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        # early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        # fit model
        history = model.fit(X_train, y_train,
            batch_size=args.batch_size, 
            epochs=args.epochs,
            verbose=args.verbose,
            callbacks=[early_stopping, WandbCallback()] if args.early_stopping else [WandbCallback()],
            validation_split=0.1
        )

        models[representation] = model

        train_end = timer()
        logging.info(f" {representation} model ({timedelta(seconds=train_end-train_start)})")

        wandb.join()


    return models


def eval_model(models, data, args):
    """This function evaluates the df and tik-tok attack.

    Args:
        models: Dictionary containing trained DF and Tik-Tok attack models
        data: Dictionary containing the train and test data
    """

    X_train, y_train = data["X_train"], data["y_train"]
    X_test1, y_test1 = data["X_test1"], data["y_test1"]
    X_test2, y_test2 = data["X_test2"], data["y_test2"]

    table = [[representation] for representation in REPRESENTATIONS] # table for logging

    idx = 0
    for representation in REPRESENTATIONS:
        if representation == "directional":
            X_train = np.sign(X_train)
            X_test1 = np.sign(X_test1)
            X_test2 = np.sign(X_test2)
        
        model = models[representation]
        score_train = model.evaluate(X_train, y_train, verbose=args.verbose)
        score_test1 = model.evaluate(X_test1, y_test1, verbose=args.verbose)
        score_test2 = model.evaluate(X_test2, y_test2, verbose=args.verbose)

        table[idx].append(score_train[1])
        table[idx].append(score_test1[1])
        table[idx].append(score_test2[1])
        idx += 1

        wandb.log({
            f"train_acc_{representation}": score_train[1],
            f"test_acc_{representation}": (score_test1[1] + score_test2[1]) / 2.0,
        })

    logging.info(f"Evaluation Results:\n\n{tabulate(table, headers=['Model Features', 'Train Acc', 'Test1 Acc', 'Test2 Acc'], tablefmt='github')}\n")


def extract_embeddings(models, data, args):
    """This function extracts the df and tik-tok attack latent space embeddings of the traces.

    Args:
        models: Dictionary containing trained DF and Tik-Tok attack models
        data: Dictionary containing the train and test data
    Returns:
        embeddings: Dictionary containing the embeddings for each combination of trace representation and trained model
    """
    embeddings = {}

    for representation in REPRESENTATIONS:
        model = models[representation]

        # remove the last layers of the model and save it
        model.pop() # ReLU
        model.pop() # Dropout
        model.pop() # Dense
        model.pop() # Softmax

        # extract embeddings
        for model_features_load in REPRESENTATIONS:
            X_test1 = data["X_test1"]
            X_test2 = data["X_test2"]

            if model_features_load == "directional":
                X_test1 = np.sign(data["X_test1"])
                X_test2 = np.sign(data["X_test2"])

            embeddings[f"test1-{representation}-{model_features_load}"] = model.predict(X_test1, batch_size=args.batch_size, verbose=args.verbose)
            embeddings[f"test2-{representation}-{model_features_load}"] = model.predict(X_test2, batch_size=args.batch_size, verbose=args.verbose)
        
    return embeddings


def calc_ber(data, embeddings, args):
    """This function estimates the Bayes Error Rate for all embeddings and raw representations.

    Args:
        data: Dictionary containing the train and test data
        embeddings: Dictionary containing the extraced embddings of the traces
    Returns:
        Bayes Error Rate results: Dictionary containing the estimated Bayes Error Rate for all embeddings and raw representations
    """

    ber_results = {}

    for representation in REPRESENTATIONS:
        for model_features_load in (['raw'] + REPRESENTATIONS):
            logging.debug(f"Combination: {representation}-{model_features_load}")

            # loading correct combination of data
            if model_features_load == "raw":
                X_test1, X_test2 = data['X_test1'], data['X_test2']
                y_test1, y_test2 = data['y_test1'], data['y_test2']

                if representation == "directional":
                    X_test1, X_test2 = np.sign(X_test1), np.sign(X_test2)
            else:
                X_test1, X_test2 = embeddings[f"test1-{representation}-{model_features_load}"], embeddings[f"test2-{representation}-{model_features_load}"]
                y_test1, y_test2 = data['y_test1'], data['y_test2']

            X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1])
            X_test2 = X_test2.reshape(X_test2.shape[0], X_test2.shape[1])

            # X_test1 for train
            dist = compute_distance(X_test1, X_test2, args.knn_measure)
            logging.debug("Test1")
            knn_est1 = knn_ber(dist, y_test1, y_test2)
            logging.debug(f"BER: {knn_est1}")

            # X_test2 for train
            dist = compute_distance(X_test2, X_test1, args.knn_measure)
            logging.debug("Test2")
            knn_est2 = knn_ber(dist, y_test2, y_test1)
            logging.debug(f"BER: {knn_est2}\n")

            ber_results[f"{representation}-{model_features_load}"] = (knn_est1 + knn_est2)/2

    return ber_results


def calc_mi(data, embeddings, args):
    """This function estimates the Mutual Information for all embeddings and raw representations.

    Args:
        data: Dictionary containing the train and test data
        embeddings: Dictionary containing the extraced embddings of the traces
    Returns:
        Mutual Information results: Dictionary containing the estimated Mutual Information for all embeddings and raw representations
    """

    mi_results = {}
    y_test1 = np.array([int(y) for y in data["y_test1"]])
    y_test2 = np.array([int(y) for y in data["y_test2"]])

    for representation in REPRESENTATIONS:
        for model_features_load in (['raw'] + REPRESENTATIONS):
            logging.debug(f"Combination: {representation}-{model_features_load}")

            # loading correct combination of data
            if model_features_load == "raw":
                X_test1, X_test2 = data['X_test1'], data['X_test2']

                if representation == "directional":
                    X_test1, X_test2 = np.sign(X_test1), np.sign(X_test2)
            else:
                X_test1, X_test2 = embeddings[f"test1-{representation}-{model_features_load}"], embeddings[f"test2-{representation}-{model_features_load}"]

            X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1])
            X_test2 = X_test2.reshape(X_test2.shape[0], X_test2.shape[1])

            # X_test1 for train
            dist = compute_distance(X_test1, X_test2, args.knn_measure)
            logging.debug("Test1")
            knn_est1 = knn_mi(dist, y_test1, y_test2, args.mi_k) * np.log2(math.e)         
            logging.debug(f"MI: {knn_est1}")

            # X_test2 for train
            dist = compute_distance(X_test2, X_test1, args.knn_measure)
            logging.debug("Test2")
            knn_est2 = knn_mi(dist, y_test2, y_test1, args.mi_k) * np.log2(math.e)
            logging.debug(f"MI: {knn_est2}\n")  

            mi_results[f"{representation}-{model_features_load}"] = (knn_est1 + knn_est2)/2

    return mi_results


def main(args):
    total_start = timer()
    cv = CV(n_splits=args.k_fold, shuffle=True, random_state=42)

    X, y = load_data(args.data_path, args)
    
    cv_count = 1

    ber_estimations = []
    mi_estimations = []

    for train_idx, test_idx in cv.split(X, y):
        start_cv = timer()
        logging.info(f'------------------- CV RUN {cv_count} OF {args.k_fold} -------------------')
        data = get_split(X, y, train_idx, test_idx)

        # train model
        logging.info(f"Train Models:")
        models = train_models(data["X_train"], data["y_train"], cv_count, args)

        wandb.init(
            project="DeepSE-WF", 
            entity="bayes_error_security_wf",
            config=vars(args),
            group=os.environ["WANDB_GROUPNAME"],
            job_type="eval",
            name=f"eval-fold{cv_count}"
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


        wandb.log({
            f"ber": min(ber_results.values()),
            f"mi": max(mi_results.values())
        })


        cv_count += 1
        wandb.join()
    
    wandb.init(
        project="DeepSE-WF", 
        entity="bayes_error_security_wf",
        config=vars(args),
        group=os.environ["WANDB_GROUPNAME"],
        job_type="eval",
        name=f"summary"
    )


    total_end = timer()
    logging.info(f'----------------------------- Final Results -----------------------------')
    logging.info(f"Bayes Error Rate is {np.mean(ber_estimations):.4f} (+-{np.std(ber_estimations):.4f})")
    logging.info(f"Mutual Information is {np.mean(mi_estimations):.4f} (+-{np.std(mi_estimations):.4f})")
    logging.info(f"Total Time: {timedelta(seconds=total_end-total_start)}\n\n")

    wandb.log({
        "overall_ber": np.mean(ber_estimations),
        "overall_mi": np.mean(mi_estimations)
    })

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeepSE-WF estiamtion of Bayes Error and Mutual Information', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=LOG_LVL)
    else:
        logging.basicConfig(stream=sys.stdout, level=LOG_LVL)

    os.environ["WANDB_GROUPNAME"] = f"{args.model}-{args.defense}-{wandb.util.generate_id()}"
    

    logging.info(f"Data Path: {args.data_path}")
    logging.info(f"Number of Traces: {args.n_traces}")
    logging.info(f"Number of Epochs: {args.epochs}")
    logging.info(f"Using GPU: {len(tf.config.list_physical_devices('GPU'))>0}\n")

    main(args)


