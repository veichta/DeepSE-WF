"""Implementation of the AWF attack model in tensorlfow."""
import tensorflow as tf


def build_awf_cnn_model(input_shape, classes, args):
    """This function builds the AWF cnn attack model.

    Args:
        input_shape: Shape of the input shape i.e. the trace
        classes: Number of classes in the dataset
        args: Arguments

    Returns:
        model: Tensorflow keras sequential model which implements the AWF attack neural network.
    """
    kernel_size = 5
    filters = 32
    pool_size = 4

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dropout(input_shape=input_shape, rate=args.dropout))

    model.add(
        tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation="relu",
            strides=1,
        )
    )

    model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding="valid"))

    model.add(
        tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation="relu",
            strides=1,
        )
    )

    model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding="valid"))

    model.add(tf.keras.layers.Flatten())

    # classification layers of df
    model.add(
        tf.keras.layers.Dense(
            args.embedding_size, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(
        tf.keras.layers.Dense(
            classes, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)
        )
    )
    model.add(tf.keras.layers.Activation("softmax"))

    model.build([None, input_shape])

    return model
