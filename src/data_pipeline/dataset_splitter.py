from pathlib import Path
import random
import shutil

# This module handles the stratified splitting of the unified dataset into train, validation, and test sets.
# It ensures that images from different sources (e.g., VisDrone, AFO) are proportionally distributed across all splits.
# It also constructs the final directory structure required by YOLO and copies/renames the files accordingly.


def split_dataset(normalized_data, images_dirs, output_dir, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, seed = 42):
    """
    Split the dataset into train/val/test and copy images and labels to the appropriate directories.
    The output structure matches the YOLO format defined in dataset.yaml.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError(f"Total ratio must be 1.0, got: {train_ratio + val_ratio + test_ratio:.4f}")

    # Build image index                                                   
    if isinstance(images_dirs, (str, Path)):
        images_dirs = [images_dirs]

    image_index = {}
    for folder in images_dirs:
        folder_path = Path(folder)

        if not folder_path.exists():
            continue

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in folder_path.glob(ext):
                if p.stem not in image_index:
                    image_index[p.stem] = p

    # Group keys by dataset prefix for stratified split: "visdrone_img_001" -> prefix "visdrone"; "afo_img_001" -> prefix "afo".
    groups: dict[str, list[str]] = {}
    for key in normalized_data:
        parts = key.split('_', 1)
        if len(parts) != 2:
            continue
        prefix = parts[0]
        groups.setdefault(prefix, []).append(key)

    # Stratified split - each group uses its own rng.          
    # Using a shared rng means shuffle results depend on iteration order of groups.items(). 
    # Adding a 3rd dataset wouldchange the shuffle of existing groups even with the same seed.
    # Using f"{seed}_{prefix}" gives each group a stable, independent seed.
    train_keys, val_keys, test_keys = [], [], []

    for prefix, keys in groups.items():
        rng = random.Random(f"{seed}_{prefix}")  # stable per-group seed.
        rng.shuffle(keys)

        n       = len(keys)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_keys.extend(keys[:n_train])
        val_keys.extend(keys[n_train : n_train + n_val])
        test_keys.extend(keys[n_train + n_val :])


    # Create YOLO directory structure.

    output_dir = Path(output_dir)
    splits     = {"train": train_keys, "val": val_keys, "test": test_keys}

    for split_name in splits:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # Copy images and write YOLO formatted labels.
    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    for split_name, keys in splits.items():
        img_out_dir = output_dir / "images" / split_name
        lbl_out_dir = output_dir / "labels" / split_name

        for image_key in keys:
            parts = image_key.split('_', 1)
            if len(parts) != 2:
                continue

            _, image_stem = parts
            image_path = image_index.get(image_stem)

            if image_path is None:
                stats["skipped"] += 1
                continue

            # Rename image to match label name so YOLO can pair them correctly e.g "00001.jpg" → "visdrone_00001.jpg" ↔ "visdrone_00001.txt"
            shutil.copy2(image_path, img_out_dir / f"{image_key}{image_path.suffix}")

            # Write YOLO label file.
            label_path = lbl_out_dir / f"{image_key}.txt"
            with label_path.open('w', encoding='utf-8') as f:
                for class_id, x_c, y_c, w, h in normalized_data[image_key]:
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            stats[split_name] += 1

    return stats