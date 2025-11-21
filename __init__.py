from .nodes import (
    ImageNoiseExpander,
    ImageExpandMerger,
    ImageExpandDirectionOption,
    ImageExpandModeOption,
)

NODE_CLASS_MAPPINGS = {
    "ImageNoiseExpander": ImageNoiseExpander,
    "ImageExpandMerger": ImageExpandMerger,
    "ImageExpandDirectionOption": ImageExpandDirectionOption,
    "ImageExpandModeOption": ImageExpandModeOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNoiseExpander": "Image Noise Expander",
    "ImageExpandMerger": "Image Expand Merger",
    "ImageExpandDirectionOption": "Image Expand Direction Option",
    "ImageExpandModeOption": "Image Expand Mode Option",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
