import time
import shutil
from pathlib import Path


from src.data_pipeline.class_mapping import mapping_visdrone, mapping_afoninja
from src.data_pipeline.coord_normalizer import normalize_coordinates
from src.data_pipeline.dataset_splitter import split_dataset


# Path config.

ROOT = Path(__file__).resolve().parents[2] 
DATA_DIR = ROOT / "data"
RAW_DATASET_DIR = DATA_DIR / "rawDataset"
OUTPUT_DATASET_DIR = DATA_DIR / "dataset"

DATASETS = [
    {
        "name": "visdrone",
        "mapper": mapping_visdrone,
        "splits": [
            RAW_DATASET_DIR / "VisDrone2019-DET-train",
            RAW_DATASET_DIR / "VisDrone2019-DET-val",
            RAW_DATASET_DIR / "VisDrone2019-DET-test-dev",
        ],
        "img_dir": "images",  # annotation annotations/ + images/ subdirs.
    },
    {
        "name": "afo",
        "mapper": mapping_afoninja,
        "splits": [
            RAW_DATASET_DIR / "afo" / "train",
            RAW_DATASET_DIR / "afo" / "validation",
            RAW_DATASET_DIR / "afo" / "test",
        ],
        "img_dir": "img",  # annotation ann/ + img/ subdirs.
    },
    
    # Add more datasets here as needed, following the same structure.

]


# Split config.

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def run():
    """
    Main data pipeline orchestrator.
    
    Steps:
    1. Class Mapping: Read annotations from each dataset split, filter human objects
    2. Coordinate Normalization: Convert pixel coordinates to YOLO format [0, 1]
    3. Dataset Split: Stratified split into train/val/test with proper directory structure
    """
    
    start_time = time.time()

    # Step 1: Class Mapping.
    print("\n[STEP 1/3] Class Mapping — Extracting human objects from annotations...\n")

    all_mapped = {}  # Consolidate results from all datasets.
    all_image_dirs = []  # Collect all image directories.

    for dataset in DATASETS:
        name = dataset["name"]
        mapper = dataset["mapper"]
        splits = dataset["splits"]
        img_dir_name = dataset["img_dir"]

        dataset_mapped = {}
        dataset_img_dirs = []

        print(f" Processing {name.upper()}:")

        for split_path in splits:
            # Validate directory exists
            if not split_path.exists():
                print(f"[{split_path.name}] Directory not found: {split_path}")
                continue

            try:
                split_mapped = mapper(split_path)
                dataset_mapped.update(split_mapped)
                print(f"[{split_path.name}] {len(split_mapped):>5} files mapped")

            except Exception as e:
                print(f"[{split_path.name}] Error: {e}")
                continue

            # Collect image directory path for later normalization.
            img_dir = split_path / img_dir_name
            if img_dir.exists():
                dataset_img_dirs.append(img_dir)

        if dataset_mapped:
            print(f"Total: {len(dataset_mapped)} files")
            all_mapped.update(dataset_mapped)
            all_image_dirs.extend(dataset_img_dirs)

    print(f"\nSummary: {len(all_mapped):>5} total files mapped across all datasets")

    if not all_mapped:
        print("\n NO DATA FOUND! Cannot proceed. Check your data directories:")
        for dataset in DATASETS:
            print(f" {dataset['name']}: {dataset['splits']}")
        return

    # Step 2: Coordinate normalization.
    # Verify image directories exist
    print("  Checking image directories:")
    for img_dir in all_image_dirs:
        status = "Y" if img_dir.exists() else "X"
        print(f" {status}  {img_dir}")

    normalized_data = normalize_coordinates(all_mapped, all_image_dirs)

    skipped = len(all_mapped) - len(normalized_data)
    print(f"\n Normalized: {len(normalized_data):>5} files")
    if skipped > 0:
        print(f" Skipped:    {skipped:>5} files (missing images / invalid bbox)")

    if not normalized_data:
        print("\n NO NORMALIZED DATA! Cannot proceed.")
        return

    # Step 3: Dataset splitting.
    print("\n[STEP 3/3] Splitting dataset into train/val/test (stratified)...\n")

    # Remove old dataset if exists.
    if OUTPUT_DATASET_DIR.exists():
        print(f"  Cleaning old dataset: {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)

    try:
        stats = split_dataset(
            normalized_data=normalized_data,
            images_dirs=all_image_dirs,
            output_dir=OUTPUT_DATASET_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=SEED)
        
    except Exception as e:
        print(f"\nError during split: {e}")
        return

    elapsed = time.time() - start_time

    print(f"  Split Statistics:")
    print(f"Train: {stats['train']:>5} files")
    print(f"Val:   {stats['val']:>5} files")
    print(f"Test:  {stats['test']:>5} files")
    if stats.get("skipped", 0) > 0:
        print(f"Skipped: {stats['skipped']:>5} files")

    print(f"  Time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run()
