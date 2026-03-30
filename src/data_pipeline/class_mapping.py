from pathlib import Path
import json

# EXPLAINATION: The Input of YOLO is formatted: <class> <x_center> <y_center> <width> <height> (normalized by image width and height). 
# We better transform the annotation files in the datasets to the YOLO format for better training performance.

def mapping_visdrone(label_path):   
    """
    Read the annotation files in the VisDrone for mapping class human: {'pedestrian', 'people'} -> {'human'}.
    """
    VISDRONE_HUMAN_LABELS = {1, 2} # pedestrian, people in VisDrone dataset.
    TARGET_CLASS_LABEL = 0 # Human in the target dataset.

    mapped_objects = {}

    annotation_path = Path(label_path) / "annotations"

    for file_path in annotation_path.glob("*.txt"):
        file_objects = []  

        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',') # Split the line into parts.
    
                # Check valid 
                if len(parts) != 8:
                    continue

                try:
                    score = int(parts[4])
                    category_ID = int(parts[5])
                except ValueError:
                    continue

                if score == 0: 
                    continue
                
                if category_ID not in VISDRONE_HUMAN_LABELS:
                    continue

    
                # FLOAT is prepared for calculating coordinates <x_center>, <y_center>, <width>, <height> in the YOLO format.
                x_min = float(parts[0])
                y_min = float(parts[1]) 
                width = float(parts[2])
                height = float(parts[3])

                if width <= 0 or height <= 0:
                    continue

                # Sử dụng Tuple () để tối ưu bộ nhớ
                file_objects.append((TARGET_CLASS_LABEL, x_min, y_min, width, height))
        
        # Only add the file to the mapped_objects if there are valid objects in it (Human in this case).
        if file_objects:
            mapped_objects[f"visdrone_{file_path.stem}"] = file_objects

    return mapped_objects


def mapping_afoninja(label_paths):
    """
    Read the annotation files in the AFO for mapping class human: {'person', 'human'} -> {'human'}.
    """
    AFONINJA_HUMAN_LABELS = {"person", "human"} # person, human in AFO dataset.
    TARGET_CLASS_LABEL = 0 # Human in the target dataset.

    mapped_objects = {}

    annotations_path = Path(label_paths) / "ann"

    for json_file in annotations_path.glob("*.json"):
        file_objects = []

        try:
            # SỬA LỖI CÚ PHÁP: Dùng dấu chấm '.' và ép chuẩn utf-8
            with json_file.open('r', encoding='utf-8') as file:
                data = json.load(file)
        
            for obj in data.get('objects', []):
                class_title = obj.get('classTitle', '').lower()

                if class_title not in AFONINJA_HUMAN_LABELS:
                    continue

                points = obj.get('points', {}).get('exterior', [])
                if len(points) != 2:
                    continue

                x1, y1 = points[0]
                x2, y2 = points[1]
                
                x_min  = float(min(x1, x2))
                y_min  = float(min(y1, y2))
                width  = float(abs(x2 - x1))
                height = float(abs(y2 - y1))


                if width <= 0 or height <= 0:
                    continue

                file_objects.append((TARGET_CLASS_LABEL, x_min, y_min, width, height))

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing {json_file.name}: {e}")
            continue

        if file_objects:
            # "img_001.jpg.json" -> json_file.stem = "img_001.jpg" -> need one more .stem to get "img_001" as the image name.
            image_name = Path(json_file.stem).stem
            mapped_objects[f"afo_{image_name}"] = file_objects

    return mapped_objects