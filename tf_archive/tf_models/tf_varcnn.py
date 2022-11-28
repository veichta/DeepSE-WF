import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model

parameters = {"kernel_initializer": "he_normal"}


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def dilated_basic_1d(
    filters,
    suffix,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    dilations=(1, 1),
):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord("a") + block)

    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(
            filters,
            kernel_size,
            padding="causal",
            strides=stride,
            dilation_rate=dilations[0],
            use_bias=False,
            name=f"res{stage_char}{block_char}_branch2a_{suffix}",
            **parameters,
        )(x)

        y = BatchNormalization(epsilon=1e-5, name=f"bn{stage_char}{block_char}_branch2a_{suffix}")(
            y
        )

        y = Activation("relu", name=f"res{stage_char}{block_char}_branch2a_relu_{suffix}")(y)

        y = Conv1D(
            filters,
            kernel_size,
            padding="causal",
            use_bias=False,
            dilation_rate=dilations[1],
            name=f"res{stage_char}{block_char}_branch2b_{suffix}",
            **parameters,
        )(y)

        y = BatchNormalization(epsilon=1e-5, name=f"bn{stage_char}{block_char}_branch2b_{suffix}")(
            y
        )

        if block == 0:
            shortcut = Conv1D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                name=f"res{stage_char}{block_char}_branch1_{suffix}",
                **parameters,
            )(x)

            shortcut = BatchNormalization(
                epsilon=1e-5, name=f"bn{stage_char}{block_char}_branch1_{suffix}"
            )(shortcut)

        else:
            shortcut = x

        y = Add(name=f"res{stage_char}{block_char}_{suffix}")([y, shortcut])
        y = Activation("relu", name=f"res{stage_char}{block_char}_relu_{suffix}")(y)

        return y

    return f


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def basic_1d(
    filters,
    suffix,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    dilations=(1, 1),
):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    dilations = (1, 1)

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord("a") + block)

    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(
            filters,
            kernel_size,
            padding="same",
            strides=stride,
            dilation_rate=dilations[0],
            use_bias=False,
            name=f"res{stage_char}{block_char}_branch2a_{suffix}",
            **parameters,
        )(x)

        y = BatchNormalization(epsilon=1e-5, name=f"bn{stage_char}{block_char}_branch2a_{suffix}")(
            y
        )

        y = Activation("relu", name=f"res{stage_char}{block_char}_branch2a_relu_{suffix}")(y)

        y = Conv1D(
            filters,
            kernel_size,
            padding="same",
            use_bias=False,
            dilation_rate=dilations[1],
            name=f"res{stage_char}{block_char}_branch2b_{suffix}",
            **parameters,
        )(y)

        y = BatchNormalization(epsilon=1e-5, name=f"bn{stage_char}{block_char}_branch2b_{suffix}")(
            y
        )

        if block == 0:
            shortcut = Conv1D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                name=f"res{stage_char}{block_char}_branch1_{suffix}",
                **parameters,
            )(x)

            shortcut = BatchNormalization(
                epsilon=1e-5, name=f"bn{stage_char}{block_char}_branch1_{suffix}"
            )(shortcut)

        else:
            shortcut = x

        y = Add(name=f"res{stage_char}{block_char}_{suffix}")([y, shortcut])
        y = Activation("relu", name=f"res{stage_char}{block_char}_relu_{suffix}")(y)

        return y

    return f


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def ResNet18(inputs, suffix, blocks=None, block=None, numerical_names=None):
    if blocks is None:
        blocks = [2, 2, 2, 2]
    if block is None:
        block = dilated_basic_1d
    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = ZeroPadding1D(padding=3, name=f"padding_conv1_{suffix}")(inputs)
    x = Conv1D(64, 7, strides=2, use_bias=False, name=f"conv1_{suffix}")(x)
    x = BatchNormalization(epsilon=1e-5, name=f"bn_conv1_{suffix}")(x)
    x = Activation("relu", name=f"conv1_relu_{suffix}")(x)
    x = MaxPooling1D(3, strides=2, padding="same", name=f"pool1_{suffix}")(x)

    features = 64
    outputs = []

    for stage_id, iterations in enumerate(blocks):
        x = block(features, suffix, stage_id, 0, dilations=(1, 2), numerical_name=False)(x)
        for block_id in range(1, iterations):
            x = block(
                features,
                suffix,
                stage_id,
                block_id,
                dilations=(4, 8),
                numerical_name=(block_id > 0 and numerical_names[stage_id]),
            )(x)

        features *= 2
        outputs.append(x)

    x = GlobalAveragePooling1D(name=f"pool5_{suffix}")(x)
    return x


def get_config(timing):
    return {
        "mixture": [["dir"], ["time"]],
        "dir_dilations": not timing,
        "time_dilations": bool(timing),
    }


def build_var_cnn_model(input_shape, classes, time, args):
    config = get_config(time)
    mixture_num = 1 if time else 0

    mixture = config["mixture"]
    use_dir = "dir" in mixture[mixture_num]
    use_time = "time" in mixture[mixture_num]
    use_metadata = "metadata" in mixture[mixture_num]
    dir_dilations = config["dir_dilations"]
    time_dilations = config["time_dilations"]

    # Constructs dir ResNet
    if use_dir:
        dir_input = Input(
            shape=input_shape,
            name="dir_input",
        )
        if dir_dilations:
            dir_output = ResNet18(dir_input, "dir", block=dilated_basic_1d)
        else:
            dir_output = ResNet18(dir_input, "dir", block=basic_1d)

    # Constructs time ResNet
    if use_time:
        time_input = Input(
            shape=input_shape,
            name="time_input",
        )
        if time_dilations:
            time_output = ResNet18(time_input, "time", block=dilated_basic_1d)
        else:
            time_output = ResNet18(time_input, "time", block=basic_1d)

    # Construct MLP for metadata
    if use_metadata:
        metadata_input = Input(shape=(7,), name="metadata_input")
        metadata_output = Dense(32)(
            metadata_input
        )  # consider this the embedding of all the metadata
        metadata_output = BatchNormalization()(metadata_output)
        metadata_output = Activation("relu")(metadata_output)

    # Forms input and output lists and possibly add final dense layer
    input_params = []
    concat_params = []
    if use_dir:
        input_params.append(dir_input)
        concat_params.append(dir_output)
    if use_time:
        input_params.append(time_input)
        concat_params.append(time_output)
    if use_metadata:
        input_params.append(metadata_input)
        concat_params.append(metadata_output)

    if len(concat_params) == 1:
        combined = concat_params[0]
    else:
        combined = Concatenate()(concat_params)

    # Better to have final fc layer if combining multiple models
    if len(concat_params) > 1:
        combined = Dense(1024)(combined)
        combined = BatchNormalization()(combined)
        combined = Activation("relu")(combined)
        combined = Dropout(0.5)(combined)

    embedding = Dense(
        units=args.embedding_size,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        name="embedding",
    )(combined)
    embedding = BatchNormalization()(embedding)
    embedding = Activation("relu")(embedding)
    embedding = Dropout(0.5)(embedding)
    outputs = Dense(classes)(embedding)
    model_output = Activation("softmax", name="outputs")(outputs)

    return Model(inputs=input_params, outputs=model_output)
