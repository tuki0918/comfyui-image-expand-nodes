from .nodes import (
    ImageExpandNoiser,
    ImageExpandMerger,
    ImageExpandDirectionOption,
    ImageExpandModeOption,
)

NODE_CLASS_MAPPINGS = {
    "ImageExpandNoiser": ImageExpandNoiser,
    "ImageExpandMerger": ImageExpandMerger,
    "ImageExpandDirectionOption": ImageExpandDirectionOption,
    "ImageExpandModeOption": ImageExpandModeOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageExpandNoiser": "Image Expand Noiser",
    "ImageExpandMerger": "Image Expand Merger",
    "ImageExpandDirectionOption": "Image Expand Direction Option",
    "ImageExpandModeOption": "Image Expand Mode Option",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
