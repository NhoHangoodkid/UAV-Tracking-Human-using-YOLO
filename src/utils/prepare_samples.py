"""Sample video preparation utility converting VisDrone-VID image sequences to MP4."""

from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
RAW_SEQ_DIR = ROOT / "data" / "rawDataset" / "VisDrone2019-VID-test-dev" / "sequences"
OUTPUT_DIR = ROOT / "data" / "samples"

SAMPLES = [
    {
        "name": "uav_street_009.mp4",
        "seq_id": "uav0000009_03358_v",
        "desc": "Street scene, mixed traffic and pedestrians (219 frames)",
    },
    {
        "name": "uav_high_119.mp4",
        "seq_id": "uav0000119_02301_v",
        "desc": "High-altitude UAV view, tiny pedestrians (179 frames)",
    },
    {
        "name": "uav_crowd_088.mp4",
        "seq_id": "uav0000088_00290_v",
        "desc": "Crowded pedestrian plaza, heavy occlusion (296 frames)",
    },
]


def create_sample_videos(fps=30, overwrite=False):
    """Assemble VisDrone image directories into standalone H.264 video files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created_files = []

    print("\n[SAMPLES] Creating UAV Test Sample Videos from VisDrone-VID")

    for item in SAMPLES:
        out_path = OUTPUT_DIR / item["name"]
        seq_dir = RAW_SEQ_DIR / item["seq_id"]

        if out_path.exists() and not overwrite:
            print(f"  [OK] Already exists: {item['name']} ({item['desc']})")
            created_files.append(out_path)
            continue

        if not seq_dir.exists():
            print(f"  [WARNING] Sequence directory not found: {seq_dir}")
            continue

        images = sorted(seq_dir.glob("*.jpg"))
        if not images:
            print(f"  [WARNING] No images found in: {seq_dir}")
            continue

        first_frame = cv2.imread(str(images[0]))
        h, w, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        print(
            f"  [INFO] Compiling {item['seq_id']} -> {item['name']} ({len(images)} frames, {w}x{h})..."
        )
        for img_path in images:
            frame = cv2.imread(str(img_path))
            writer.write(frame)
        writer.release()

        # Transcode to universal H.264 profile using ffmpeg
        try:
            import subprocess
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_h264 = out_path.with_name(f"h264_{out_path.name}")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(out_path),
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
                out_path.unlink()
                temp_h264.rename(out_path)
        except Exception as err:
            print(f"  [WARNING] H.264 encoding fallback: {err}")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Rendered (H.264): {item['name']} ({size_mb:.2f} MB)")
        created_files.append(out_path)

    print("[SAMPLES] Sample video preparation completed\n")
    return created_files


if __name__ == "__main__":
    create_sample_videos(fps=30)
