from .nodes import ImageNoiseExpander

NODE_CLASS_MAPPINGS = {
    "ImageNoiseExpander": ImageNoiseExpander
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNoiseExpander": "Image Noise Expander"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

