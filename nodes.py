import torch


class ImageExpandNoiser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["top", "bottom", "left", "right"],),
                "mode": (["outside", "inside"],),
                "percentage": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "Image/Processing"

    def expand_image(self, image, direction, mode, percentage):
        # image: [Batch, Height, Width, Channels]
        # Output images should be in the same format
        # Masks should be [Batch, Height, Width]

        B, H, W, C = image.shape

        # Determine expansion amount in pixels
        if direction in ["top", "bottom"]:
            expand_size = int(H * percentage)
        else:  # left, right
            expand_size = int(W * percentage)

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
                out_mask[:, expand_size:, :] = 0.0

            elif direction == "bottom":
                # Expand bottom -> Image moves up, Bottom is noise
                # Visible part of image: expand_size to H
                # Target position: 0 to H - expand_size
                visible_h = H - expand_size
                out_image[:, :visible_h, :, :] = image[:, expand_size:, :, :]
                out_mask[:, :visible_h, :] = 0.0

            elif direction == "left":
                # Expand left -> Image moves right, Left is noise
                visible_w = W - expand_size
                out_image[:, :, expand_size:, :] = image[:, :, :visible_w, :]
                out_mask[:, :, expand_size:] = 0.0

            elif direction == "right":
                # Expand right -> Image moves left, Right is noise
                visible_w = W - expand_size
                out_image[:, :, :visible_w, :] = image[:, :, expand_size:, :]
                out_mask[:, :, :visible_w] = 0.0

        else:  # mode == "inside"
            new_H, new_W = H, W

            # Use original image as base
            out_image = image.clone()
            # Initialize mask with 0.0 (original image)
            out_mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

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
                "mask": ("MASK",),
                "direction": (["top", "bottom", "left", "right"],),
                "mode": (["outside", "inside"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_images"
    CATEGORY = "Image/Processing"

    def merge_images(self, image1, image2, mask, direction, mode):
        # image1: [B, H1, W1, C] (Original)
        # image2: [B, H2, W2, C] (Expanded/Inpainted)
        # mask: [B, H, W] (Should match image2 usually)

        # Ensure batch dimension for mask if missing
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        B, H, W, C = image2.shape

        # Calculate expand_size from mask
        # Mask is 1.0 at the expanded/generated region
        expand_size = 0
        if direction in ["top", "bottom"]:
            # Check a column (e.g., middle column)
            col_idx = W // 2
            expand_size = int(torch.sum(mask[0, :, col_idx]).item())
        else:
            # Check a row (e.g., middle row)
            row_idx = H // 2
            expand_size = int(torch.sum(mask[0, row_idx, :]).item())

        out_image = None

        if mode == "outside":
            # Append the generated part of image2 to image1
            if direction == "top":
                new_part = image2[:, :expand_size, :, :]
                out_image = torch.cat((new_part, image1), dim=1)
            elif direction == "bottom":
                new_part = image2[:, -expand_size:, :, :]
                out_image = torch.cat((image1, new_part), dim=1)
            elif direction == "left":
                new_part = image2[:, :, :expand_size, :]
                out_image = torch.cat((new_part, image1), dim=2)
            elif direction == "right":
                new_part = image2[:, :, -expand_size:, :]
                out_image = torch.cat((image1, new_part), dim=2)

        else:  # mode == "inside"
            # Overlay masked part of image2 onto image1
            mask_expanded = mask.unsqueeze(-1)  # [B, H, W, 1]

            # Ensure image1 is on same device as image2
            if image1.device != image2.device:
                image1 = image1.to(image2.device)

            # Ensure mask is on same device as image2
            if mask_expanded.device != image2.device:
                mask_expanded = mask_expanded.to(image2.device)

            # Basic overlay: image1 (background) + image2 (foreground) * mask
            out_image = image1 * (1.0 - mask_expanded) + image2 * mask_expanded

        return (out_image,)


class ImageExpandDirectionOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["top", "bottom", "left", "right"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("direction",)
    FUNCTION = "get_option"
    CATEGORY = "Image/Processing"

    def get_option(self, direction):
        return (direction,)


class ImageExpandModeOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["outside", "inside"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mode",)
    FUNCTION = "get_option"
    CATEGORY = "Image/Processing"

    def get_option(self, mode):
        return (mode,)
