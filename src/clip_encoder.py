"""the clip encoder for the text prompts"""

import tensorflow as tf
import keras_cv
import numpy as np
from keras.preprocessing.image import load_img

#the encoder implementation needs to be wrapped as a custom keras class/model instead of a scrip format.

# Specify the path to your image
image_path = '/home/samer/Desktop/Beedoo/ControlNet_Implementation/Dataset/image.jpg' #"https://laion.ai/blog/laion-5b/"

# Load a preset CLIP model
clip_model = keras_cv.models.CLIPBackbone.from_preset("clip_vit_base_patch16")

# Create a tokenizer
tokenizer = keras_cv.models.CLIPTokenizer.from_preset("clip_vit_base_patch16")

# Create image preprocessor
image_converter = keras_cv.models.CLIPImageConverter(image_size=(512, 512))

# input text prompt
prompt = input("Enter the text prompt: ")

#load input image
image = load_img(image_path)

# Tokenize the text
tokenized_text = tokenizer(prompt)

# Preprocess the image
processed_image = image_converter(image)

# Get the CLIP model output
clip_outputs = clip_model({"images": processed_image, "tokens": tokenized_text})
