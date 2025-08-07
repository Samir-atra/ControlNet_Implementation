"""the clip encoder for the text prompts"""

import tensorflow as tf
import keras
import keras_cv


class CLIPTextEncoder(keras.Model):
    def __init__(self, preset="clip_vit_base_patch16", **kwargs):
        super().__init__(**kwargs)
        # Load a preset CLIP model's text components
        self.tokenizer = keras_cv.models.CLIPTokenizer.from_preset(preset)
        # We only need the text-related layers from the backbone
        clip_backbone = keras_cv.models.CLIPBackbone.from_preset(preset)
        self.text_encoder = clip_backbone.get_layer("text_encoder")

    def call(self, prompts):
        # Tokenize the text
        tokenized_text = self.tokenizer(prompts)
        # Get the text embeddings
        text_embeddings = self.text_encoder({"tokens": tokenized_text})
        return text_embeddings


if __name__ == '__main__':
    # Example usage:
    text_encoder = CLIPTextEncoder()
    prompt = "a photograph of an astronaut riding a horse"
    
    # You can pass a list of prompts
    embeddings = text_encoder([prompt, "a beautiful sunset over the mountains"])

    print("Prompt:", prompt)
    print("Embedding shape:", embeddings.shape)
