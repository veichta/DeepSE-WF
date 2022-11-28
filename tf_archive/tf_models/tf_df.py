"""Implementation of the DF attack model."""
import tensorflow as tf


def build_df_model(input_shape, classes, args):
    """This function builds the df attack model.

    Args:
        input_shape: Shape of the input shape i.e. the trace
        classes: Number of classes in the dataset
        args: Arguments

    Returns:
        model: Tensorflow keras sequential model which implements the DF attack neural network
    """
    m = tf.keras.Sequential()

    for i, f in enumerate([32, 64, 128, 256]):
        m.add(
            tf.keras.layers.Conv1D(
                filters=f,
                kernel_size=8,
                input_shape=input_shape,
                strides=1,
                padding="same",
            )
        )
        m.add(tf.keras.layers.BatchNormalization(axis=-1))
        if i == 0:
            m.add(tf.keras.layers.ELU(alpha=1.0))
        else:
            m.add(tf.keras.layers.Activation("relu"))
        m.add(
            tf.keras.layers.Conv1D(
                filters=f,
                kernel_size=8,
                input_shape=input_shape,
                strides=1,
                padding="same",
            )
        )
        m.add(tf.keras.layers.BatchNormalization(axis=-1))
        if i == 0:
            m.add(tf.keras.layers.ELU(alpha=1.0))
        else:
            m.add(tf.keras.layers.Activation("relu"))
        m.add(tf.keras.layers.MaxPool1D(8, 4, padding="same"))
        m.add(tf.keras.layers.MaxPool1D(8, 4, padding="same"))
        m.add(tf.keras.layers.Dropout(0.2))

    m.add(tf.keras.layers.Flatten())
    m.add(
        tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))
    )
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation("relu"))

    m.add(tf.keras.layers.Dropout(0.7))

    m.add(
        tf.keras.layers.Dense(
            args.embedding_size, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)
        )
    )
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation("relu"))

    m.add(tf.keras.layers.Dropout(0.5))

    m.add(
        tf.keras.layers.Dense(
            classes, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)
        )
    )
    m.add(tf.keras.layers.Activation("softmax"))

    m.build([None, input_shape])

    return m
