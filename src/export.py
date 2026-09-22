"""Model export module to convert trained PyTorch YOLO checkpoints to ONNX or TensorRT."""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "v1_yolo11s_1280" / "weights" / "best.pt"


def export_model(
    weights_path, imgsz=1280, dynamic=False, half=False, format_type="onnx", opset=17
):
    """Export PyTorch YOLO model to ONNX or TensorRT engine format."""
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Source weights not found: {weights}")
        sys.exit(1)

    print(f"\n[EXPORT] Preparing export: {weights.name} -> {format_type.upper()}")
    print(f"  Source path:    {weights}")
    print(f"  Input size:     {imgsz}x{imgsz}")
    print(f"  FP16 precision: {half}")
    print(f"  Dynamic shapes: {dynamic}")
    print(f"  ONNX opset:     {opset}")

    model = YOLO(str(weights))

    try:
        exported_path = model.export(
            format=format_type,
            imgsz=imgsz,
            dynamic=dynamic,
            half=half,
            simplify=True,
            opset=opset,
            device=0,
        )
        print(f"\n[EXPORT] Export completed successfully!")
        print(f"  Artifact: {exported_path}")

        if format_type == "onnx":
            import onnx

            onnx_model = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)
            print("  ONNX verification: Graph structure validated successfully")

        return exported_path
    except Exception as error:
        print(f"\n[ERROR] Export execution failed: {error}")
        return None


def parse_args():
    """Parse command line arguments for model export."""
    parser = argparse.ArgumentParser(description="Export YOLO Model to Edge AI Formats")
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to .pt weights"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Square input resolution"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "engine"],
        help="Export format",
    )
    parser.add_argument(
        "--half", action="store_true", default=False, help="Enable FP16 half precision"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic batch/shape",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    export_model(
        weights_path=cli_args.weights,
        imgsz=cli_args.imgsz,
        dynamic=cli_args.dynamic,
        half=cli_args.half,
        format_type=cli_args.format,
        opset=cli_args.opset,
    )
