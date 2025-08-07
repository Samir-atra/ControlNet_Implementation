"""the ControlNet model architecture"""


import keras
from keras import layers
import tensorflow as tf
import keras_hub


def get_controlnet_model(img_size):
    """
    The ControlNet model.
    This model takes a conditioning image and outputs a list of feature maps
    that can be injected into the skip connections of a larger UNet.
    """
    conditioning_input = keras.Input(shape=img_size + (3,), name="conditioning_input")

    # A convolution layer initialized with zeros.
    def zero_conv(x):
        return layers.Conv2D(
            filters=x.shape[-1], kernel_size=1, kernel_initializer="zeros"
        )(x)

    ### [First half of the network: downsampling inputs] ###
    # This part mirrors the encoder of the main UNet.

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(conditioning_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    block_outputs = []

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # The output of each block is passed through a zero convolution
        # and collected. These will be added to the UNet skip connections.
        block_outputs.append(zero_conv(x))

    # The final output of the encoder, also passed through a zero convolution.
    final_output = zero_conv(x)

    model = keras.Model(inputs=conditioning_input, outputs=[*block_outputs, final_output], name="controlnet")
    return model
