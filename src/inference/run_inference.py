"""Inference runner script for UAV human detection on images and videos."""

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "v1_yolo11s_1280" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "output" / "inference"


def run_inference(
    source,
    weights=DEFAULT_WEIGHTS,
    imgsz=1280,
    conf=0.25,
    iou=0.5,
    save=True,
    device="0",
):
    """Run YOLO human detection inference on specified source."""
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFERENCE] Running detection on: {source}")
    print(f"  Weights:    {weights_path.name}")
    print(f"  Resolution: {imgsz}")
    print(f"  Confidence: {conf}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True,
    )

    total_detections = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(
        f"\n[INFERENCE] Completed: {len(results)} images processed, {total_detections} humans detected"
    )
    if save:
        print(f"  Output saved to: {OUTPUT_DIR / 'predict'}")

    return results


def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run UAV Human Detection Inference")
    parser.add_argument(
        "--source", type=str, required=True, help="Path to image, directory, or video"
    )
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to YOLO weights"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference image resolution"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--device", type=str, default="0", help="CUDA device index or 'cpu'"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save output images"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        source=args.source,
        weights=Path(args.weights),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=not args.no_save,
        device=args.device,
    )
