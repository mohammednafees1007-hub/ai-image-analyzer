from PIL import Image

def load_image(image_path: str):
    """
    Loads an image from disk and returns a PIL Image.
    """
    image = Image.open(image_path).convert("RGB")
    return image
