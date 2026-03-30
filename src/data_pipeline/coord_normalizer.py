from pathlib import Path
import struct

# The YOLO input format is  <class_label> <x_center> <y_center> <width> <height>, where the coordinates are normalized to the range [0, 1] relative to the
# image dimensions.
# This file uses the output of the class_mapping.py to convert the original annotation files into the YOLO format,
#  and then normalizes the coordinates based on the image dimensions.

def get_image_size(image_path):
    """
    Get the dimensions of an image (width, height) using the struct module to read the image header for  normalizings.
    This function supports JPEG and PNG formats.
    """
    try:
        with open(image_path, 'rb') as f:
            header = f.read(24)

            # PNG format
            if header[:8] == b'\x89PNG\r\n\x1a\n':
                width = struct.unpack('>I', header[16:20])[0]
                height = struct.unpack('>I', header[20:24])[0]
                return width, height

            # JPEG format
            if header[:2] == b'\xff\xd8':
                f.seek(2)
                for _ in range(1000):  # prevent infinite loop.
                    marker = f.read(2)
                    if len(marker) < 2:
                        break

                    # Start Of Frame markers contain size info.
                    if marker[0] == 0xFF and marker[1] in (
                        0xC0, 0xC1, 0xC2, 0xC3,
                        0xC5, 0xC6, 0xC7, 0xC9,
                        0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        f.read(3)
                        height = struct.unpack('>H', f.read(2))[0]
                        width = struct.unpack('>H', f.read(2))[0]
                        return width, height
                    else:
                        seg_len = struct.unpack('>H', f.read(2))[0]
                        f.seek(seg_len - 2, 1)

    except (OSError, struct.error):
        return None

    return None

def clip(value):
    """
    Clip a value to the range [0, 1]. This is used to ensure that the normalized coordinates do not exceed the valid range.
    """
    return max(0, min(1, value))

def normalize_coordinates(mapped_objects, images_dir):
    """
    Normalize the coordinates of the bounding boxes in the mapped_objects based on the dimensions of the corresponding images.
    The function iterates through each file in the mapped_objects, retrieves the image dimensions using get_image_size(),
    and then normalizes the coordinates to the YOLO format.
    """
    if isinstance(image_dirs, (str, Path)):
        image_dirs = [image_dirs]

    image_index = {}
    
    for folder in image_dirs:
        folder_path = Path(folder)

        if not folder_path.exists():
            continue

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for p in folder_path.glob(ext):
                image_index[p.stem] = p
    
    normalized_data = {}

    for image_key, bboxes in mapped_objects.items():
        
        # "visdrone_00001" -> _, "00001"
        parts = image_key.split('_', 1)
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
            
            if x_min >= img_width or y_min >= img_height or (x_min + width) <= 0 or (y_min + height) <= 0:
                continue

            w_norm = clip(width / img_width)
            h_norm = clip(height / img_height)

            if w_norm == 0.0 or h_norm == 0.0:
                continue

            x_center_norm = clip((x_min + width / 2.0) / img_width)
            y_center_norm = clip((y_min + height / 2.0) / img_height)

    
            normalized_bboxes.append((class_id, x_center_norm, y_center_norm, w_norm, h_norm))

        # Only save the normalized bounding boxes if there are valid ones after normalization.
        if normalized_bboxes:
            # Still use the original image key (e.g., "visdrone_00001") for the normalized data.
            normalized_data[image_key] = normalized_bboxes

    return normalized_data