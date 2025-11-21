from .nodes import ImageNoiseExpander, ImageExpandMerger

NODE_CLASS_MAPPINGS = {
    "ImageNoiseExpander": ImageNoiseExpander,
    "ImageExpandMerger": ImageExpandMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNoiseExpander": "Image Noise Expander",
    "ImageExpandMerger": "Image Expand Merger",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
