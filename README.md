# ControlNet Implementation

This repository contains a Keras and TensorFlow implementation of the research paper: "Adding Conditional Control to Text-to-Image Diffusion Models". This project provides the building blocks for a ControlNet model, which allows for conditioning a text-to-image diffusion model on an additional input image.

This implementation is a work in progress and is based on the work from [keras-team/keras-hub/pull/2209](https://github.com/keras-team/keras-hub/pull/2209).

## File Descriptions

*   `LICENSE`: The license for this project.
*   `README.md`: This file.
*   `docs/(ControlNet) Planning-oriented Autonomous Driving.pdf`: A research paper on a related topic. The ControlNet implementation in this repository is based on the paper "Adding Conditional Control to Text-to-Image Diffusion Models".
*   `script.sh`: An empty bash script.
*   `src/`: This directory contains the source code for the ControlNet implementation.
    *   `__init__.py`: An empty file that makes the `src` directory a Python package.
    *   `clip_encoder.py`: Contains the `CLIPTextEncoder` class, which is used to encode text prompts into embeddings.
    *   `controlnet.py`: Contains the `get_controlnet_model` function, which creates the ControlNet model.
    - `sd_encoder_block.py`: Implements a U-Net-like architecture, which is the main model that ControlNet is designed to control.

## Dependencies

This project requires the following Python libraries:

*   `tensorflow`
*   `keras`
*   `keras_cv`
*   `keras_hub`

You can install these dependencies using pip:

```bash
pip install tensorflow keras keras_cv keras_hub
```

## Usage

The components in this repository can be used to build a text-to-image model that is conditioned on an additional input image. Here is an example of how you might use the `ControlNet` and `CLIPTextEncoder` models:

```python
import tensorflow as tf
from src.controlnet import get_controlnet_model
from src.clip_encoder import CLIPTextEncoder

# --- Parameters ---
IMG_SIZE = (256, 256)
PROMPT = "a photograph of an astronaut riding a horse"

# --- Models ---

# ControlNet model
controlnet_model = get_controlnet_model(IMG_SIZE)
controlnet_model.summary()

# CLIP Text Encoder
text_encoder = CLIPTextEncoder()
text_embeddings = text_encoder([PROMPT])

# --- Example Usage ---

# A (dummy) conditioning image
conditioning_image = tf.zeros((1, *IMG_SIZE, 3))

# The ControlNet model takes a conditioning image and outputs a list of feature maps
control_outputs = controlnet_model(conditioning_image)

print("Text embeddings shape:", text_embeddings.shape)
print("Number of control outputs:", len(control_outputs))
for i, output in enumerate(control_outputs):
    print(f"Control output {i+1} shape:", output.shape)

```

This example demonstrates how to create the ControlNet model and the CLIP text encoder, and how to get the outputs from each. These outputs would then be injected into a larger diffusion model (like the one in `sd_encoder_block.py`) to guide the image generation process.
