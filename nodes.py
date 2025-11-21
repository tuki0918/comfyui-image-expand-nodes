import torch

class ImageNoiseExpander:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["top", "bottom", "left", "right"],),
                "mode": (["outside", "inside"],),
                "percentage": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.5, "step": 0.01}),
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
        else: # left, right
            expand_size = int(W * percentage)
            
        if mode == "outside":
            if direction in ["top", "bottom"]:
                new_H = H + expand_size
                new_W = W
            else:
                new_H = H
                new_W = W + expand_size
                
            # Initialize output tensor with random noise (0-1)
            out_image = torch.rand((B, new_H, new_W, C), dtype=image.dtype, device=image.device)
            # Initialize mask with 1.0 (masked/noise)
            out_mask = torch.ones((B, new_H, new_W), dtype=torch.float32, device=image.device)
            
            # Place original image and clear mask area (set to 0.0)
            if direction == "top":
                # Noise at top, Image at bottom
                out_image[:, expand_size:, :, :] = image
                out_mask[:, expand_size:, :] = 0.0
            elif direction == "bottom":
                # Image at top, Noise at bottom
                out_image[:, :H, :, :] = image
                out_mask[:, :H, :] = 0.0
            elif direction == "left":
                # Noise at left, Image at right
                out_image[:, :, expand_size:, :] = image
                out_mask[:, :, expand_size:] = 0.0
            elif direction == "right":
                # Image at left, Noise at right
                out_image[:, :, :W, :] = image
                out_mask[:, :, :W] = 0.0
                
        else: # mode == "inside"
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

        return (out_image, out_mask)

