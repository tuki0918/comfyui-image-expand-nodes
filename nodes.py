import torch
import math


class ImageExpandNoiser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_options": ("EXPAND_OPTION",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "Image/Processing"

    def expand_image(self, image, expand_options, mask=None):
        direction = expand_options["direction"]
        mode = expand_options["mode"]
        percentage = float(expand_options.get("percentage", 0.2))

        # image: [Batch, Height, Width, Channels]
        # Output images should be in the same format
        # Masks should be [Batch, Height, Width]

        B, H, W, C = image.shape

        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != (H, W):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=(H, W), mode="nearest"
                ).squeeze(1)
            if mask.shape[0] != B:
                mask = mask.expand(B, -1, -1)
            if mask.device != image.device:
                mask = mask.to(image.device)

        # Determine expansion amount in pixels
        if direction in ["top", "bottom"]:
            expand_size = math.ceil(H * percentage)
        else:  # left, right
            expand_size = math.ceil(W * percentage)

        if mode == "outside":
            new_H, new_W = H, W

            # Initialize with random noise
            out_image = torch.rand((B, H, W, C), dtype=image.dtype, device=image.device)
            # Initialize mask with 1.0 (masked/noise)
            out_mask = torch.ones((B, H, W), dtype=torch.float32, device=image.device)

            # Shift logic
            if direction == "top":
                # Expand top -> Image moves down, Top is noise
                # Visible part of image: 0 to H - expand_size
                # Target position: expand_size to H
                visible_h = H - expand_size
                out_image[:, expand_size:, :, :] = image[:, :visible_h, :, :]
                if mask is not None:
                    out_mask[:, expand_size:, :] = mask[:, :visible_h, :]
                else:
                    out_mask[:, expand_size:, :] = 0.0

            elif direction == "bottom":
                # Expand bottom -> Image moves up, Bottom is noise
                # Visible part of image: expand_size to H
                # Target position: 0 to H - expand_size
                visible_h = H - expand_size
                out_image[:, :visible_h, :, :] = image[:, expand_size:, :, :]
                if mask is not None:
                    out_mask[:, :visible_h, :] = mask[:, expand_size:, :]
                else:
                    out_mask[:, :visible_h, :] = 0.0

            elif direction == "left":
                # Expand left -> Image moves right, Left is noise
                visible_w = W - expand_size
                out_image[:, :, expand_size:, :] = image[:, :, :visible_w, :]
                if mask is not None:
                    out_mask[:, :, expand_size:] = mask[:, :, :visible_w]
                else:
                    out_mask[:, :, expand_size:] = 0.0

            elif direction == "right":
                # Expand right -> Image moves left, Right is noise
                visible_w = W - expand_size
                out_image[:, :, :visible_w, :] = image[:, :, expand_size:, :]
                if mask is not None:
                    out_mask[:, :, :visible_w] = mask[:, :, expand_size:]
                else:
                    out_mask[:, :, :visible_w] = 0.0

        else:  # mode == "inside"
            new_H, new_W = H, W

            # Use original image as base
            out_image = image.clone()
            # Initialize mask with 0.0 (original image)
            if mask is not None:
                out_mask = mask.clone()
            else:
                out_mask = torch.zeros(
                    (B, H, W), dtype=torch.float32, device=image.device
                )

            # Overlay noise logic
            if direction == "top":
                # Noise overlays from top
                # Region to be noise: 0 to expand_size
                out_image[:, :expand_size, :, :] = torch.rand(
                    (B, expand_size, W, C), dtype=image.dtype, device=image.device
                )
                out_mask[:, :expand_size, :] = 1.0

            elif direction == "bottom":
                # Noise overlays from bottom
                # Region to be noise: H - expand_size to H
                noise_height = expand_size
                out_image[:, -noise_height:, :, :] = torch.rand(
                    (B, noise_height, W, C), dtype=image.dtype, device=image.device
                )
                out_mask[:, -noise_height:, :] = 1.0

            elif direction == "left":
                # Noise overlays from left
                # Region to be noise: 0 to expand_size
                out_image[:, :, :expand_size, :] = torch.rand(
                    (B, H, expand_size, C), dtype=image.dtype, device=image.device
                )
                out_mask[:, :, :expand_size] = 1.0

            elif direction == "right":
                # Noise overlays from right
                # Region to be noise: W - expand_size to W
                noise_width = expand_size
                out_image[:, :, -noise_width:, :] = torch.rand(
                    (B, H, noise_width, C), dtype=image.dtype, device=image.device
                )
                out_mask[:, :, -noise_width:] = 1.0

        return (out_image, out_mask)


class ImageExpandMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "expand_options": ("EXPAND_OPTION",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "Image/Processing"

    def _normalize_mask(self, mask, batch, height, width, device):
        if mask is None:
            return None

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        if mask.shape[1:] != (height, width):
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1), size=(height, width), mode="nearest"
            ).squeeze(1)

        if mask.shape[0] != batch:
            mask = mask.expand(batch, -1, -1)

        if mask.device != device:
            mask = mask.to(device)

        return mask.to(torch.float32).clamp(0.0, 1.0)

    def _calc_expand_size(self, height, width, direction, percentage):
        percentage = float(percentage)
        if direction in ["top", "bottom"]:
            return int(max(1, min(height - 1, math.ceil(height * percentage))))
        return int(max(1, min(width - 1, math.ceil(width * percentage))))

    def merge_images(self, image1, image2, expand_options, mask=None):
        direction = expand_options["direction"]
        mode = expand_options["mode"]
        percentage = float(expand_options.get("percentage", 0.2))

        # image1: [B, H1, W1, C] (Original)
        # image2: [B, H2, W2, C] (Expanded/Inpainted)
        # mask: [B, H, W] (Should match image2 usually)

        if image1.device != image2.device:
            image1 = image1.to(image2.device)

        if image1.shape[3] != image2.shape[3]:
            if image1.shape[3] == 3 and image2.shape[3] == 4:
                alpha = torch.ones(
                    (image1.shape[0], image1.shape[1], image1.shape[2], 1),
                    device=image1.device,
                    dtype=image1.dtype,
                )
                image1 = torch.cat((image1, alpha), dim=3)
            elif image1.shape[3] == 4 and image2.shape[3] == 3:
                alpha = torch.ones(
                    (image2.shape[0], image2.shape[1], image2.shape[2], 1),
                    device=image2.device,
                    dtype=image2.dtype,
                )
                image2 = torch.cat((image2, alpha), dim=3)

        B, H1, W1, C1 = image1.shape
        _, H2, W2, _ = image2.shape
        if (H1, W1) != (H2, W2):
            # Merger expects image2 to be the same size as image1 for the shift-based merge.
            # If it differs, fall back to returning image2 (common for already-expanded canvases).
            return (image2,)

        expand_size = self._calc_expand_size(H1, W1, direction, percentage)
        mask = self._normalize_mask(mask, B, H1, W1, image2.device)

        out_image = None

        if mode == "outside":
            # Append generated part of image2 to image1.
            # If mask is provided, apply image2's masked overlap onto image1 (shift-aware)
            # to preserve seam blending.
            merged_original = image1
            if mask is not None:
                mask_e = mask.unsqueeze(-1)  # [B, H, W, 1]
                merged_original = image1.clone()

                if direction == "top":
                    overlap_h = H1 - expand_size
                    img2_overlap = image2[:, expand_size:, :, :]
                    mask_overlap = mask_e[:, expand_size:, :, :]
                    merged_original[:, :overlap_h, :, :] = (
                        merged_original[:, :overlap_h, :, :] * (1.0 - mask_overlap)
                        + img2_overlap * mask_overlap
                    )
                elif direction == "bottom":
                    overlap_h = H1 - expand_size
                    img2_overlap = image2[:, :overlap_h, :, :]
                    mask_overlap = mask_e[:, :overlap_h, :, :]
                    merged_original[:, expand_size:, :, :] = (
                        merged_original[:, expand_size:, :, :] * (1.0 - mask_overlap)
                        + img2_overlap * mask_overlap
                    )
                elif direction == "left":
                    overlap_w = W1 - expand_size
                    img2_overlap = image2[:, :, expand_size:, :]
                    mask_overlap = mask_e[:, :, expand_size:, :]
                    merged_original[:, :, :overlap_w, :] = (
                        merged_original[:, :, :overlap_w, :] * (1.0 - mask_overlap)
                        + img2_overlap * mask_overlap
                    )
                elif direction == "right":
                    overlap_w = W1 - expand_size
                    img2_overlap = image2[:, :, :overlap_w, :]
                    mask_overlap = mask_e[:, :, :overlap_w, :]
                    merged_original[:, :, expand_size:, :] = (
                        merged_original[:, :, expand_size:, :] * (1.0 - mask_overlap)
                        + img2_overlap * mask_overlap
                    )

            if direction == "top":
                new_part = image2[:, :expand_size, :, :]
                out_image = torch.cat((new_part, merged_original), dim=1)
            elif direction == "bottom":
                new_part = image2[:, -expand_size:, :, :]
                out_image = torch.cat((merged_original, new_part), dim=1)
            elif direction == "left":
                new_part = image2[:, :, :expand_size, :]
                out_image = torch.cat((new_part, merged_original), dim=2)
            elif direction == "right":
                new_part = image2[:, :, -expand_size:, :]
                out_image = torch.cat((merged_original, new_part), dim=2)

        else:  # mode == "inside"
            # Overlay masked part of image2 onto image1.
            # If no mask is given, return image2 as-is.
            if mask is None:
                out_image = image2
            else:
                mask_expanded = mask.unsqueeze(-1)  # [B, H, W, 1]
                out_image = image1 * (1.0 - mask_expanded) + image2 * mask_expanded

        return (out_image,)


class ImageExpandOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["top", "bottom", "left", "right"],),
                "mode": (["outside", "inside"],),
                "percentage": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("EXPAND_OPTION",)
    RETURN_NAMES = ("expand_options",)
    FUNCTION = "get_option"
    CATEGORY = "Image/Processing"

    def get_option(self, direction, mode, percentage):
        return (
            {"direction": direction, "mode": mode, "percentage": float(percentage)},
        )


NODE_CLASS_MAPPINGS = {
    "ImageExpandNoiser": ImageExpandNoiser,
    "ImageExpandMerger": ImageExpandMerger,
    "ImageExpandOption": ImageExpandOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageExpandNoiser": "Image Expand Noiser",
    "ImageExpandMerger": "Image Expand Merger",
    "ImageExpandOption": "Image Expand Option",
}
