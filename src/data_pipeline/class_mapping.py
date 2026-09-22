"""Class mapping module to extract human annotations from VisDrone and AFO datasets."""

import json
from pathlib import Path


def mapping_visdrone(label_path):
    """Read VisDrone annotations and map pedestrian and people categories to human class 0."""
    visdrone_human_labels = {1, 2}
    target_class_label = 0
    mapped_objects = {}

    annotation_path = Path(label_path) / "annotations"

    for file_path in sorted(annotation_path.glob("*.txt")):
        file_objects = []

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) != 8:
                    continue

                try:
                    score = int(parts[4])
                    category_id = int(parts[5])
                except ValueError:
                    continue

                if score == 0 or category_id not in visdrone_human_labels:
                    continue

                x_min = float(parts[0])
                y_min = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])

                if width <= 0 or height <= 0:
                    continue

                file_objects.append((target_class_label, x_min, y_min, width, height))

        if file_objects:
            mapped_objects[f"visdrone_{file_path.stem}"] = file_objects

    return mapped_objects


def mapping_afoninja(label_paths):
    """Read AFO Supervisely JSON annotations and extract human bounding boxes for class 0."""
    afo_human_labels = {"human"}
    target_class_label = 0
    mapped_objects = {}

    annotations_path = Path(label_paths) / "ann"

    for json_file in sorted(annotations_path.glob("*.json")):
        file_objects = []

        try:
            with json_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            for obj in data.get("objects", []):
                class_title = obj.get("classTitle", "").lower()
                if class_title not in afo_human_labels:
                    continue

                points = obj.get("points", {}).get("exterior", [])
                if len(points) != 2:
                    continue

                x1, y1 = points[0]
                x2, y2 = points[1]

                x_min = float(min(x1, x2))
                y_min = float(min(y1, y2))
                width = float(abs(x2 - x1))
                height = float(abs(y2 - y1))

                if width <= 0 or height <= 0:
                    continue

                file_objects.append((target_class_label, x_min, y_min, width, height))

        except (json.JSONDecodeError, KeyError, IndexError) as error:
            print(f"[ERROR] Failed to parse {json_file.name}: {error}")
            continue

        if file_objects:
            image_name = Path(json_file.stem).stem
            mapped_objects[f"afo_{image_name}"] = file_objects

    return mapped_objects
