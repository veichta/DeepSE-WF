import logging
import math

import numpy as np
import torch
from scipy.special import digamma


def _get_lowerbound(value, k, classes):
    """Compute the Bayes Error Rate based on the kNN error.

    Args:
        value: Error of kNN classifier
        k: Value of k
        classes: Number of classes in the dataset
    Returns:
        Bayes Error Rate Estimation
    """
    if classes > 2 or k == 1:
        return ((classes - 1.0) / float(classes)) * (
            1.0 - math.sqrt(max(0.0, 1 - ((float(classes) / (classes - 1.0)) * value)))
        )

    if k > 2:
        return value / float(1 + (1.0 / math.sqrt(k)))

    return value / float(1 + math.sqrt(2.0 / k))


def compute_distance(x_train, x_test, measure="squared_l2"):
    """Calculates the distance matrix between test and train.

    Args:
      x_train: Matrix (NxD) where each row represents a training sample
      x_test: Matrix (MxD) where each row represents a test sample
      measure: Distance measure (not necessarly metric) to use
    Raises:
      NotImplementedError: When the measure is not implemented
    Returns:
      Matrix (MxN) where elemnt i,j is the distance between
      x_test_i and x_train_j.
    """
    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).float().cuda()
        x_test = torch.from_numpy(x_test).float().cuda()
    else:
        if x_train.dtype != np.float32:
            x_train = np.float32(x_train)
        if x_test.dtype != np.float32:
            x_test = np.float32(x_test)

    if measure == "squared_l2":
        if torch.cuda.is_available():
            x_xt = torch.matmul(x_test, x_train.t()).cpu().numpy()

            x_train_2 = torch.sum(x_train ** 2, 1).cpu().numpy()
            x_test_2 = torch.sum(x_test ** 2, 1).cpu().numpy()
        else:
            x_xt = np.matmul(x_test, np.transpose(x_train))

            x_train_2 = np.sum(np.square(x_train), axis=1)
            x_test_2 = np.sum(np.square(x_test), axis=1)

        for i in range(np.shape(x_xt)[0]):
            x_xt[i, :] = np.multiply(x_xt[i, :], -2)
            x_xt[i, :] = np.add(x_xt[i, :], x_test_2[i])
            x_xt[i, :] = np.add(x_xt[i, :], x_train_2)

    elif measure == "cosine":
        # if len(tf.config.list_physical_devices("GPU")) > 0:
        if torch.cuda.is_available():
            x_xt = torch.matmul(x_test, x_train.t()).cpu().numpy()

            x_train_2 = torch.norm(x_train, dim=1).cpu().numpy()
            x_test_2 = torch.norm(x_test, dim=1).cpu().numpy()
        else:
            x_xt = np.matmul(x_test, np.transpose(x_train))

            x_train_2 = np.linalg.norm(x_train, axis=1)
            x_test_2 = np.linalg.norm(x_test, axis=1)

        outer = np.outer(x_test_2, x_train_2)
        x_xt = np.ones(np.shape(x_xt)) - np.divide(x_xt, outer)

    else:
        raise NotImplementedError(f"Method '{measure}' is not implemented")

    return x_xt


def knn_ber(d, y_train, y_test, k=1):
    """Calculate the Bayes Error Rate based on knn method and on the precomputed distance matrix d.

    Args:
      d: Distance matrix (MxN) where elemnt i,j is the distance between x_test_i and x_train_j
      y_train: N label vector for the training samples
      y_test: M label vector for the test samples
      k: number of in-class neighbors for every test sample
    Returns:
      Bayes Error Rate based on knn for the k provided
    """

    total_classes = np.unique(np.concatenate((y_train, y_test))).size

    num_elements = np.shape(d)[0]

    if k < 1:
        raise ValueError("No smaller value than '1' allowed for k")

    val_k = k
    cnt = 0
    if val_k == 1:
        indices = np.argmin(d, axis=1)

        for idx, val in enumerate(indices):

            if len(np.shape(y_train)) == 1:
                if y_test[idx] != y_train[val]:
                    cnt += 1
            elif y_test[idx] != y_train[idx, val]:
                cnt += 1

        res = float(cnt) / num_elements
        logging.debug(f"knn error: {res}")

    else:
        indices = np.argpartition(d, val_k - 1, axis=1)
        for i in range(num_elements):

            # Get max vote
            if len(np.shape(y_train)) == 1:
                labels = y_train[indices[i, :val_k]]
            else:
                labels = y_train[i, indices[i, :val_k]]
            keys, counts = np.unique(labels, return_counts=True)

            maxkey = keys[np.argmax(counts)]
            if y_test[i] != maxkey:
                cnt += 1

        res = float(cnt) / num_elements

    return _get_lowerbound(res, k, total_classes), res


def knn_mi(d, y_train, y_test, k=5):
    """Calculate the Mutual Information based on knn method and on the precomputed distance matrix d.

    Args:
      d: Distance matrix (MxN) where elemnt i,j is the distance between
         x_test_i and x_train_j
      y_train: N label vector for the training samples
      y_test: M label vector for the test samples
      k: number of in-class neighbors for every test sample
    Returns:
      Mutual Information based on knn for the k provided
    """

    M, N = d.shape

    keys, counts = np.unique(y_train, return_counts=True)
    _, test_class_counts = np.unique(y_test, return_counts=True)

    dg_N = digamma(N)
    dg_N_Ys = []
    dg_k = digamma(k)
    dg_m_y = []

    for c in keys:
        c = int(c)

        test_indices = np.nonzero(y_test == c)[0]
        train_indices = np.nonzero(y_train == c)[0]

        N_Y = counts[c]
        dg_N_Ys.extend([digamma(N_Y)] * test_class_counts[c])

        assert k < N_Y, "k should be smaller than then number of samples per class"

        # Get only the elements (train and test) with that class
        sub_d = d[test_indices, :][:, train_indices]

        # Get the k-1 nearest index of train samples w.r.t each test samples
        indices = np.argpartition(sub_d, k - 1, axis=1)

        # Get the distance for the index per test sample
        max_d = sub_d[np.arange(len(test_indices)), indices[:, k - 1]]

        # Filter the train samples based on the distance per test sample
        mask = np.less_equal(d[test_indices, :], np.tile(np.expand_dims(max_d, 1), (N)))

        # Count the number of samples with smaller, or equal distance
        m_y = np.count_nonzero(mask, axis=1)

        dg_m_y.extend(digamma(m_y))

    return max(0.0, dg_N - np.mean(dg_N_Ys) + dg_k - np.mean(dg_m_y)) * np.log2(math.e)
