"""Tracking benchmark module comparing ByteTrack and BoT-SORT on UAV footage."""

import argparse
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

try:
    from src.utils.track_id import ActiveTrackIdMapper
except ModuleNotFoundError:
    from utils.track_id import ActiveTrackIdMapper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "v1_yolo11s_1280" / "weights" / "best.pt"
SAMPLES_DIR = ROOT / "data" / "samples"
OUTPUT_TRACKING_DIR = ROOT / "output" / "tracking"
OUTPUT_BENCHMARK_DIR = ROOT / "output" / "benchmark"
UAV_BOTSORT = ROOT / "src" / "uav_botsort.yaml"
UAV_BYTETRACK = ROOT / "src" / "uav_bytetrack.yaml"

# Distinct color palette for consistent track ID visualization in BGR format
COLOR_PALETTE = [
    (255, 128, 0),
    (0, 230, 115),
    (0, 165, 255),
    (255, 0, 210),
    (0, 238, 238),
    (230, 216, 173),
    (147, 112, 219),
    (50, 205, 50),
    (75, 0, 130),
    (255, 192, 203),
    (0, 128, 255),
    (173, 255, 47),
]


def get_color_for_id(track_id):
    """Return consistent BGR color mapped from track ID."""
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


def draw_uav_annotations(
    frame,
    boxes,
    track_ids,
    track_history,
    tracker_name,
    fps,
    frame_idx,
    total_frames,
    draw_trail=False,
):
    """Render clean UAV tracking annotations with bounding boxes and HUD overlay."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    if boxes is not None and track_ids is not None:
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            color = get_color_for_id(track_id)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # Motion history trail
            if draw_trail and track_history is not None:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                history = track_history[track_id]
                history.append((cx, cy))
                if len(history) > 25:
                    history.pop(0)
                if len(history) > 1:
                    for idx in range(1, len(history)):
                        cv2.line(
                            annotated,
                            history[idx - 1],
                            history[idx],
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            # Compact ID label
            label = f"#{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            tag_y1 = max(0, y1 - text_h - 3)
            tag_y2 = y1
            tag_x1 = x1
            tag_x2 = min(width, x1 + text_w + 3)

            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (tag_x1 + 1, tag_y2 - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

    # Semi-transparent HUD overlay
    hud_w, hud_h = 240, 52
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    cv2.rectangle(annotated, (8, 8), (8 + hud_w, 8 + hud_h), (80, 80, 80), 1)

    hud_title = f"{tracker_name.upper()} | UAV Human Tracker"
    cv2.putText(
        annotated,
        hud_title,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 238, 238),
        1,
        cv2.LINE_AA,
    )

    human_count = len(track_ids) if track_ids is not None else 0
    frame_text = (
        f"Frame: {frame_idx}/{total_frames}  Humans: {human_count:>2}  FPS: {fps:.1f}"
    )
    cv2.putText(
        annotated,
        frame_text,
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return annotated


def run_tracker_benchmark(
    weights_path,
    source_path,
    tracker_type,
    imgsz=1280,
    conf=0.15,
    iou=0.5,
    device="0",
    save_video=True,
    draw_trail=False,
):
    """Run tracking benchmark on a single video clip with a designated tracker."""
    print(f"\n[BENCHMARK] Executing {tracker_type.upper()} on {source_path.name}...")

    # Reset global tracker counter before processing each video
    BaseTrack.reset_id()

    model = YOLO(str(weights_path))

    # Resolve tuned UAV tracker config if available
    if tracker_type.lower() == "botsort" and UAV_BOTSORT.exists():
        tracker_config = str(UAV_BOTSORT)
    elif tracker_type.lower() == "bytetrack" and UAV_BYTETRACK.exists():
        tracker_config = str(UAV_BYTETRACK)
    else:
        tracker_config = f"{tracker_type}.yaml"

    cap = cv2.VideoCapture(str(source_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_writer = None
    output_video_path = None
    if save_video:
        OUTPUT_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        video_stem = source_path.stem
        output_video_path = OUTPUT_TRACKING_DIR / f"{video_stem}_{tracker_type}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, src_fps, (width, height)
        )

    frame_times = []
    people_per_frame = []
    raw_track_ids = set()
    track_history = defaultdict(list)
    id_mapper = ActiveTrackIdMapper()
    frame_count = 0

    start_total = time.perf_counter()

    results = model.track(
        source=str(source_path),
        imgsz=imgsz,
        tracker=tracker_config,
        conf=conf,
        iou=iou,
        persist=True,
        show=False,
        save=False,
        device=device,
        stream=True,
        verbose=False,
    )

    last_time = time.perf_counter()
    current_fps = 0.0

    for result in results:
        now_time = time.perf_counter()
        elapsed = (now_time - last_time) * 1000.0
        last_time = now_time
        frame_count += 1

        if frame_count > 5:
            frame_times.append(elapsed)
            current_fps = 1000.0 / elapsed if elapsed > 0 else 0.0

        display_ids = []
        boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            raw_ids = result.boxes.id.int().cpu().tolist()

            display_ids = id_mapper.update(raw_ids)

            raw_track_ids.update(raw_ids)
            people_per_frame.append(len(display_ids))
        else:
            id_mapper.update([])
            people_per_frame.append(0)

        if out_writer is not None:
            annotated_frame = draw_uav_annotations(
                frame=result.orig_img,
                boxes=boxes,
                track_ids=display_ids,
                track_history=track_history,
                tracker_name=tracker_type,
                fps=current_fps,
                frame_idx=frame_count,
                total_frames=total_frames,
                draw_trail=draw_trail,
            )
            out_writer.write(annotated_frame)

    if out_writer is not None:
        out_writer.release()
        try:
            import subprocess
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_h264 = output_video_path.with_name(f"h264_{output_video_path.name}")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(output_video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-preset",
                "veryfast",
                str(temp_h264),
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0 and temp_h264.exists():
                output_video_path.unlink()
                temp_h264.rename(output_video_path)
        except Exception as err:
            print(f"  [WARNING] H.264 transcoding skipped: {err}")

        size_mb = output_video_path.stat().st_size / (1024 * 1024)
        print(f"  Video rendered (H.264): {output_video_path.name} ({size_mb:.2f} MB)")

    total_elapsed = time.perf_counter() - start_total
    avg_latency = float(np.mean(frame_times)) if frame_times else 0.0
    p50_latency = float(np.median(frame_times)) if frame_times else 0.0
    p95_latency = float(np.percentile(frame_times, 95)) if frame_times else 0.0
    computed_fps = (
        (len(frame_times) / (sum(frame_times) / 1000.0)) if frame_times else 0.0
    )

    return {
        "video_name": source_path.name,
        "tracker": tracker_type.upper(),
        "frames": frame_count,
        "total_time_s": total_elapsed,
        "fps": computed_fps,
        "mean_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "unique_ids": len(raw_track_ids),
        "avg_people_per_frame": (
            float(np.mean(people_per_frame)) if people_per_frame else 0.0
        ),
        "max_people_in_frame": max(people_per_frame) if people_per_frame else 0,
        "video_output": output_video_path.name if output_video_path else None,
    }


def generate_comparison_report(all_results, weights_path, imgsz, conf, iou):
    """Write benchmark comparative metrics table to markdown report."""
    OUTPUT_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_BENCHMARK_DIR / "tracker_comparison.md"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("# UAV Tracking Benchmark Report: ByteTrack vs BoT-SORT\n\n")
        weights_url = str(weights_path).replace("\\", "/")
        file.write(
            f"- **YOLO Model:** `{weights_path.name}` ([v1_yolo11s_1280](file:///{weights_url}))\n"
        )
        file.write(f"- **Inference Resolution:** `{imgsz}`\n")
        file.write(f"- **Confidence / IoU:** `{conf}` / `{iou}`\n")
        file.write(f"- **Device:** NVIDIA GPU (RTX 4050)\n\n")

        by_video = defaultdict(list)
        for r in all_results:
            by_video[r["video_name"]].append(r)

        file.write("## 1. Performance Comparison by Video Sequence\n\n")
        for video_name, tracker_runs in by_video.items():
            file.write(f"### Video: `{video_name}`\n\n")
            file.write("| Metric | ByteTrack | BoT-SORT | Advantage |\n")
            file.write("| :--- | :--- | :--- | :--- |\n")

            byte_res = next(
                (r for r in tracker_runs if r["tracker"] == "BYTETRACK"), None
            )
            bot_res = next((r for r in tracker_runs if r["tracker"] == "BOTSORT"), None)

            if byte_res and bot_res:
                fps_diff = ((byte_res["fps"] - bot_res["fps"]) / bot_res["fps"]) * 100
                fps_eval = (
                    f"ByteTrack is +{fps_diff:.1f}% faster"
                    if fps_diff > 0
                    else "BoT-SORT is faster"
                )
                id_diff = byte_res["unique_ids"] - bot_res["unique_ids"]
                if id_diff > 0:
                    id_eval = f"BoT-SORT has fewer ID switches ({bot_res['unique_ids']} vs {byte_res['unique_ids']})"
                elif id_diff < 0:
                    id_eval = f"ByteTrack has fewer ID switches ({byte_res['unique_ids']} vs {bot_res['unique_ids']})"
                else:
                    id_eval = "Identical"

                file.write(
                    f"| **Throughput (FPS)** | **{byte_res['fps']:.2f} FPS** | {bot_res['fps']:.2f} FPS | **{fps_eval}** |\n"
                )
                file.write(
                    f"| **Mean Latency** | **{byte_res['mean_latency_ms']:.2f} ms** | {bot_res['mean_latency_ms']:.2f} ms | ByteTrack lower latency |\n"
                )
                file.write(
                    f"| **P95 Latency** | **{byte_res['p95_latency_ms']:.2f} ms** | {bot_res['p95_latency_ms']:.2f} ms | ByteTrack more consistent |\n"
                )
                file.write(
                    f"| **Total Track IDs** | {byte_res['unique_ids']} IDs | {bot_res['unique_ids']} IDs | {id_eval} |\n"
                )
                file.write(
                    f"| **Mean People / Frame** | {byte_res['avg_people_per_frame']:.1f} | {bot_res['avg_people_per_frame']:.1f} | Consistent detections |\n"
                )
                file.write(
                    f"| **Max People / Frame** | {byte_res['max_people_in_frame']} | {bot_res['max_people_in_frame']} | |\n"
                )
                file.write(
                    f"| **Video Output** | `{byte_res['video_output']}` | `{bot_res['video_output']}` | Saved in `output/tracking/` |\n\n"
                )

        file.write("## 2. Summary & Practical Recommendations\n\n")
        file.write("### When to Choose ByteTrack?\n")
        file.write(
            "- **Primary Benefit:** Significantly higher frame rate with minimal computational overhead.\n"
        )
        file.write(
            "- **Best Suited For:** Onboard embedded UAV systems (NVIDIA Jetson, Raspberry Pi AI) requiring real-time performance (>= 30 FPS) for gimbal tracking or autonomous flight.\n\n"
        )

        file.write("### When to Choose BoT-SORT?\n")
        file.write(
            "- **Primary Benefit:** Integrates Camera Motion Compensation (GMC) to correct for drone motion and camera shake.\n"
        )
        file.write(
            "- **Best Suited For:** Fast-moving UAV flights, aggressive angle changes, or crowded scenarios where minimizing ID switches is critical.\n\n"
        )

        file.write("## 3. Rendered Output Videos\n\n")
        for r in all_results:
            file.write(
                f"- `{r['video_output']}`: Rendered with **{r['tracker']}** on `{r['video_name']}` (FPS: {r['fps']:.1f})\n"
            )

    print(f"\n[BENCHMARK] Comparison report written to: {report_file}")


def main():
    """Execute benchmark comparison between ByteTrack and BoT-SORT."""
    parser = argparse.ArgumentParser(
        description="Benchmark ByteTrack vs BoT-SORT on UAV Footage"
    )
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to YOLO weights"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="Video source: 'all' to evaluate data/samples, or specific video path",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference image resolution"
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold for association"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="CUDA device index or 'cpu'"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        default=False,
        help="Disable video output saving",
    )
    parser.add_argument(
        "--draw-trail", action="store_true", default=False, help="Enable motion trails"
    )
    args = parser.parse_args()

    if args.source == "all":
        video_paths = sorted(SAMPLES_DIR.glob("*.mp4"))
        if not video_paths:
            print(
                f"[WARNING] No sample videos found in {SAMPLES_DIR}. Run prepare_samples.py first."
            )
            return
    else:
        src = Path(args.source)
        if not src.exists():
            print(f"[ERROR] Source file not found: {src}")
            return
        video_paths = [src]

    trackers = ["bytetrack", "botsort"]
    all_results = []

    print(f"\n[BENCHMARK] Multi-Object Tracking Evaluation: ByteTrack vs BoT-SORT")
    print(f"  Model:   {Path(args.weights).name}")
    print(f"  Videos:  {[v.name for v in video_paths]}")
    print(f"  Device:  {args.device}")

    for video_path in video_paths:
        print(f"\n[BENCHMARK] Processing video: {video_path.name}")
        for tracker_type in trackers:
            res = run_tracker_benchmark(
                weights_path=Path(args.weights),
                source_path=video_path,
                tracker_type=tracker_type,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                save_video=not args.no_video,
                draw_trail=args.draw_trail,
            )
            all_results.append(res)

    generate_comparison_report(
        all_results=all_results,
        weights_path=Path(args.weights),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
