from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import numpy as np

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


def describe_scene(image):
    """
    Generates a natural language description of the image.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=40)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
