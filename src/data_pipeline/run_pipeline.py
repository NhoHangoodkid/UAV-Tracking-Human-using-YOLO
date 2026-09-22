"""Data pipeline script to build the complete YOLO human dataset from VisDrone and AFO."""

import shutil
import time
import zipfile
from pathlib import Path

import yaml
from src.data_pipeline.class_mapping import mapping_afoninja, mapping_visdrone
from src.data_pipeline.coord_normalizer import normalize_coordinates

# Path configuration
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DATASET_DIR = DATA_DIR / "rawDataset"
OUTPUT_DATASET_DIR = DATA_DIR / "dataset"
OUTPUT_ZIP = DATA_DIR / "uav_human_dataset.zip"
DATASET_CONFIG_TEMPLATE = Path(__file__).resolve().parent / "dataset.yaml"

# Dataset source configuration mapping raw splits to output train, val, and test splits
DATASETS = [
    {
        "name": "visdrone",
        "mapper": mapping_visdrone,
        "raw_path": RAW_DATASET_DIR / "VisDrone2019-DET-train",
        "img_dir": "images",
        "output_split": "train",
    },
    {
        "name": "visdrone",
        "mapper": mapping_visdrone,
        "raw_path": RAW_DATASET_DIR / "VisDrone2019-DET-val",
        "img_dir": "images",
        "output_split": "val",
    },
    {
        "name": "visdrone",
        "mapper": mapping_visdrone,
        "raw_path": RAW_DATASET_DIR / "VisDrone2019-DET-test-dev",
        "img_dir": "images",
        "output_split": "test",
    },
    {
        "name": "afo",
        "mapper": mapping_afoninja,
        "raw_path": RAW_DATASET_DIR / "afo" / "train",
        "img_dir": "img",
        "output_split": "train",
    },
    {
        "name": "afo",
        "mapper": mapping_afoninja,
        "raw_path": RAW_DATASET_DIR / "afo" / "validation",
        "img_dir": "img",
        "output_split": "val",
    },
    {
        "name": "afo",
        "mapper": mapping_afoninja,
        "raw_path": RAW_DATASET_DIR / "afo" / "test",
        "img_dir": "img",
        "output_split": "test",
    },
]


def create_zip(dataset_dir, zip_path):
    """Package the dataset directory into an uncompressed zip archive for fast transfer."""
    print(f"\n[PIPELINE] Creating archive: {zip_path.name}...")

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zip_file:
        for file_path in sorted(dataset_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(dataset_dir)
                zip_file.write(file_path, arcname)
                file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[PIPELINE] Archive created: {file_count} files ({size_mb:.1f} MB)")


def run():
    """Execute the end-to-end dataset extraction, normalization, and assembly workflow."""
    start_time = time.time()

    # Prepare clean output directories
    if OUTPUT_DATASET_DIR.exists():
        print(f"[PIPELINE] Cleaning existing dataset directory: {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)

    for split in ("train", "val", "test"):
        (OUTPUT_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Initialize statistics counters
    stats = {
        "train": 0,
        "val": 0,
        "test": 0,
        "skipped_normalize": 0,
        "skipped_image": 0,
    }
    source_stats = {}

    # Process each dataset split independently
    for entry in DATASETS:
        name = entry["name"]
        mapper = entry["mapper"]
        raw_path = entry["raw_path"]
        img_dir_name = entry["img_dir"]
        output_split = entry["output_split"]

        print(
            f"\n[PIPELINE] Processing {name.upper()} -> {output_split} ({raw_path.name})"
        )

        if not raw_path.exists():
            print(f"  [WARNING] Directory not found: {raw_path}")
            continue

        img_dir = raw_path / img_dir_name
        if not img_dir.exists():
            print(f"  [WARNING] Image directory not found: {img_dir}")
            continue

        # Step 1: Class Mapping
        print("  [STEP 1/3] Class Mapping: Extracting human annotations...")
        try:
            mapped = mapper(raw_path)
        except Exception as error:
            print(f"  [ERROR] Mapping failed: {error}")
            continue

        print(f"    Mapped: {len(mapped)} files containing target human objects")
        if not mapped:
            print("    No human objects found, skipping split.")
            continue

        # Step 2: Coordinate Normalization
        print("  [STEP 2/3] Coordinate Normalization: Converting to YOLO format...")
        normalized = normalize_coordinates(mapped, img_dir)
        skipped_norm = len(mapped) - len(normalized)
        stats["skipped_normalize"] += skipped_norm

        print(f"    Normalized: {len(normalized)} files")
        if skipped_norm > 0:
            print(
                f"    Skipped: {skipped_norm} files due to invalid bbox or missing image"
            )

        if not normalized:
            print("    No valid normalized coordinates, skipping split.")
            continue

        # Step 3: Assemble Output
        print(f"  [STEP 3/3] Assembling output into {output_split}/...")
        img_out_dir = OUTPUT_DATASET_DIR / "images" / output_split
        lbl_out_dir = OUTPUT_DATASET_DIR / "labels" / output_split

        image_index = {}
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_path in img_dir.glob(ext):
                image_index[img_path.stem] = img_path

        copied = 0
        skipped_img = 0

        for image_key, bboxes in normalized.items():
            parts = image_key.split("_", 1)
            if len(parts) != 2:
                skipped_img += 1
                continue

            _, image_stem = parts
            image_path = image_index.get(image_stem)
            if image_path is None:
                skipped_img += 1
                continue

            # Copy image with dataset prefix to ensure unique filenames
            shutil.copy2(image_path, img_out_dir / f"{image_key}{image_path.suffix}")

            # Write YOLO label format
            label_path = lbl_out_dir / f"{image_key}.txt"
            with label_path.open("w", encoding="utf-8") as label_file:
                for class_id, x_c, y_c, width, height in bboxes:
                    label_file.write(
                        f"{class_id} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}\n"
                    )

            copied += 1

        stats[output_split] += copied
        stats["skipped_image"] += skipped_img

        if name not in source_stats:
            source_stats[name] = {"train": 0, "val": 0, "test": 0}
        source_stats[name][output_split] += copied

        print(f"    Assembled: {copied} image and label pairs")
        if skipped_img > 0:
            print(f"    Skipped: {skipped_img} (image files not found)")

    # Write a self-contained config beside the generated dataset.
    with DATASET_CONFIG_TEMPLATE.open("r", encoding="utf-8") as config_file:
        dataset_config = yaml.safe_load(config_file) or {}
    dataset_config["path"] = "."
    yaml_path = OUTPUT_DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"\n[PIPELINE] Configuration written to: {yaml_path}")

    # Create zip archive for Kaggle upload
    create_zip(OUTPUT_DATASET_DIR, OUTPUT_ZIP)

    # Final summary display
    elapsed = time.time() - start_time
    total = stats["train"] + stats["val"] + stats["test"]

    print("\n[PIPELINE] Pipeline Execution Complete")
    print(f"  Dataset destination: {OUTPUT_DATASET_DIR}")
    print("  Split statistics:")
    print(f"    Train: {stats['train']:>6} samples")
    print(f"    Val:   {stats['val']:>6} samples")
    print(f"    Test:  {stats['test']:>6} samples")
    print(f"    Total: {total:>6} samples")

    if stats["skipped_normalize"] > 0:
        print(f"    Skipped (normalization): {stats['skipped_normalize']} samples")
    if stats["skipped_image"] > 0:
        print(f"    Skipped (missing image): {stats['skipped_image']} samples")

    print("  Source distribution:")
    for src_name, src_splits in source_stats.items():
        src_total = sum(src_splits.values())
        print(
            f"    {src_name:>10}: train={src_splits['train']}, "
            f"val={src_splits['val']}, test={src_splits['test']} (total={src_total})"
        )

    print(f"  Kaggle package: {OUTPUT_ZIP}")
    print(f"  Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    run()
