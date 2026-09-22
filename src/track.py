"""UAV human tracking script using YOLO and BoT-SORT or ByteTrack.

Includes ID remapping to ensure displayed track IDs are sequential (1, 2, 3...)
instead of fragmented raw tracker IDs (1, 5, 47, 200...).
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics.trackers.basetrack import BaseTrack

try:
    from src.utils.track_id import ActiveTrackIdMapper
except ModuleNotFoundError:
    from utils.track_id import ActiveTrackIdMapper

ROOT = Path(__file__).resolve().parents[1]
BEST_WEIGHTS = ROOT / "runs" / "detect" / "v1_yolo11s_1280" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "output" / "tracking"
SAMPLE_VIDEO = "https://ultralytics.com/assets/people-walking.mp4"
UAV_BOTSORT = ROOT / "src" / "uav_botsort.yaml"
UAV_BYTETRACK = ROOT / "src" / "uav_bytetrack.yaml"


def _remap_and_annotate(frame, boxes, id_mapper):
    """Remap active raw tracker IDs to compact display IDs and draw on frame.

    Args:
        frame: The video frame (numpy array, BGR).
        boxes: The detection boxes result object.
        id_mapper: ActiveTrackIdMapper that releases IDs for inactive tracks.

    Returns:
        Tuple of (annotated_frame, list_of_display_ids).
    """
    if boxes is None or boxes.id is None:
        id_mapper.update([])
        return frame, []

    raw_ids = boxes.id.int().cpu().tolist()
    bboxes = boxes.xyxy.int().cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    display_ids = id_mapper.update(raw_ids)

    # Draw remapped IDs on frame
    for (x1, y1, x2, y2), disp_id, conf in zip(bboxes, display_ids, confs):
        # Bounding box
        color = _id_color(disp_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with sequential ID
        label = f"ID:{disp_id} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame, display_ids


def _id_color(track_id):
    """Generate a deterministic color for a given track ID."""
    hue = (track_id * 47) % 180  # Spread hues across the spectrum
    import numpy as np

    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def track(args):
    """Run human tracking on video or stream source."""
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights file not found: {weights}")
        print("  Execute training first: python -m src.train")
        sys.exit(1)

    # Reset global tracker counter for clean starting IDs
    BaseTrack.reset_id()

    # Resolve tracker configuration
    tracker_arg = args.tracker.lower()
    if "botsort" in tracker_arg and UAV_BOTSORT.exists():
        tracker_config = str(UAV_BOTSORT)
    elif "bytetrack" in tracker_arg and UAV_BYTETRACK.exists():
        tracker_config = str(UAV_BYTETRACK)
    else:
        tracker_config = args.tracker

    print(f"\n[TRACK] Initializing tracker on source: {args.source}")
    print(f"  Model:      {weights.name}")
    print(f"  Tracker:    {tracker_config}")
    print(f"  Resolution: {args.imgsz}")
    print(f"  Confidence: {args.conf}")

    model = YOLO(str(weights))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run tracker WITHOUT built-in save/show — we handle rendering ourselves
    # so that remapped IDs appear on the output video
    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        tracker=tracker_config,
        conf=args.conf,
        iou=args.iou,
        persist=True,
        show=False,  # We handle display ourselves
        save=False,  # We handle saving ourselves
        project=str(OUTPUT_DIR),
        name="result",
        exist_ok=True,
        device=0,
        stream=True,
        vid_stride=args.vid_stride,
    )

    frame_count = 0
    max_active_people = 0
    id_mapper = ActiveTrackIdMapper()
    video_writer = None
    save_path = OUTPUT_DIR / "result"
    save_path.mkdir(parents=True, exist_ok=True)

    for frame_result in results:
        frame_count += 1
        frame = frame_result.orig_img.copy()

        # Remap IDs and draw custom annotations
        annotated, display_ids = _remap_and_annotate(
            frame, frame_result.boxes, id_mapper
        )
        max_active_people = max(max_active_people, len(display_ids))

        # Initialize video writer on first frame
        if args.save and video_writer is None:
            h, w = annotated.shape[:2]
            out_file = str(save_path / "tracking_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Try to get source FPS, default to 30
            fps = getattr(frame_result, "fps", 30) or 30
            video_writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))

        if args.save and video_writer is not None:
            video_writer.write(annotated)

        if args.show:
            cv2.imshow("UAV Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 30 == 1:
            active_count = len(display_ids)
            max_raw = (
                max(frame_result.boxes.id.int().cpu().tolist()) if display_ids else 0
            )
            print(
                f"  Frame {frame_count:>5} | "
                f"Active: {active_count:>3} | "
                f"Active IDs: {len(display_ids):>3} | "
                f"Max raw tracker ID: {max_raw:>3}"
            )

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("\n[TRACK] Video tracking completed")
    print(f"  Frames processed:       {frame_count}")
    print(f"  Max active people:       {max_active_people}")
    print(f"  Raw track instances:     {len(id_mapper.seen_raw_ids)}")

    if args.save:
        print(f"  Saved output to:         {save_path / 'tracking_result.mp4'}")

    return results


def parse_args():
    """Parse command-line arguments for tracking module."""
    parser = argparse.ArgumentParser(description="UAV Aerial Human Tracking")
    parser.add_argument(
        "--source",
        type=str,
        default=SAMPLE_VIDEO,
        help="Video source: file path, webcam index (0), or video stream URL",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(BEST_WEIGHTS),
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image resolution",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack", "botsort.yaml", "bytetrack.yaml"],
        help="Tracking algorithm configuration",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detecting small UAV humans",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS and association",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Video frame stride (process every N-th frame)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video output in interactive window",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated tracking video to disk",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    track(cli_args)
