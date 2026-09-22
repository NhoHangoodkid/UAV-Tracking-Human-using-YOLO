"""Coordinate normalizer module to convert pixel bounding boxes into YOLO normalized format."""

import struct
from pathlib import Path


def get_image_size(image_path):
    """Read image dimensions width and height directly from PNG or JPEG file headers."""
    try:
        with open(image_path, "rb") as file:
            header = file.read(24)

            # PNG format
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                width = struct.unpack(">I", header[16:20])[0]
                height = struct.unpack(">I", header[20:24])[0]
                return width, height

            # JPEG format
            if header[:2] == b"\xff\xd8":
                file.seek(2)
                for _ in range(1000):
                    marker = file.read(2)
                    if len(marker) < 2:
                        break

                    # Start Of Frame markers containing dimension info
                    if marker[0] == 0xFF and marker[1] in (
                        0xC0,
                        0xC1,
                        0xC2,
                        0xC3,
                        0xC5,
                        0xC6,
                        0xC7,
                        0xC9,
                        0xCA,
                        0xCB,
                        0xCD,
                        0xCE,
                        0xCF,
                    ):
                        file.read(3)
                        height = struct.unpack(">H", file.read(2))[0]
                        width = struct.unpack(">H", file.read(2))[0]
                        return width, height
                    else:
                        seg_len = struct.unpack(">H", file.read(2))[0]
                        file.seek(seg_len - 2, 1)

    except (OSError, struct.error):
        return None

    return None


def clip(value, min_val=0.0, max_val=1.0):
    """Clip a numeric value to a bounded range."""
    return min(max(value, min_val), max_val)


def normalize_coordinates(mapped_objects, images_dirs):
    """Normalize pixel bounding boxes into relative YOLO coordinates between 0 and 1."""
    if isinstance(images_dirs, (str, Path)):
        images_dirs = [images_dirs]

    image_index = {}
    for folder in images_dirs:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_path in folder_path.glob(ext):
                image_index[img_path.stem] = img_path

    normalized_data = {}

    for image_key, bboxes in mapped_objects.items():
        parts = image_key.split("_", 1)
        if len(parts) != 2:
            continue

        _, image_stem = parts
        image_path = image_index.get(image_stem)
        if image_path is None:
            continue

        size = get_image_size(image_path)
        if size is None or size[0] <= 0 or size[1] <= 0:
            continue

        img_width, img_height = size
        normalized_bboxes = []

        for class_id, x_min, y_min, width, height in bboxes:
            x_max = x_min + width
            y_max = y_min + height

            # Clip pixel coordinates to image boundary
            x_min_clipped = max(0.0, min(float(img_width), x_min))
            y_min_clipped = max(0.0, min(float(img_height), y_min))
            x_max_clipped = max(0.0, min(float(img_width), x_max))
            y_max_clipped = max(0.0, min(float(img_height), y_max))

            clipped_width = x_max_clipped - x_min_clipped
            clipped_height = y_max_clipped - y_min_clipped

            # Discard tiny bounding boxes
            if clipped_width <= 2.0 or clipped_height <= 2.0:
                continue

            # Calculate center coordinate in pixel space
            x_center = x_min_clipped + clipped_width / 2.0
            y_center = y_min_clipped + clipped_height / 2.0

            # Normalize to 0 to 1 range
            x_center_norm = clip(x_center / img_width)
            y_center_norm = clip(y_center / img_height)
            width_norm = clip(clipped_width / img_width)
            height_norm = clip(clipped_height / img_height)

            normalized_bboxes.append(
                (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
            )

        if normalized_bboxes:
            normalized_data[image_key] = normalized_bboxes

    return normalized_data
