"""Training pipeline script for YOLO11 human detection on UAV datasets."""

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "dataset" / "dataset.yaml"
if not DATA_YAML.exists():
    DATA_YAML = ROOT / "src" / "data_pipeline" / "dataset.yaml"

TRAIN_YAML = ROOT / "src" / "train.yaml"
PROJECT_DIR = ROOT / "runs" / "detect"
HISTORY_CSV = PROJECT_DIR / "history.csv"


def check_environment():
    """Verify GPU availability, CUDA support, and essential dependencies."""
    print("[INFO] Checking training environment...")

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed.")
        sys.exit(1)

    print(f"  PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. GPU is required for training.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  Compute Device:  {gpu_name}")
    print(f"  Dedicated VRAM:  {vram_gb:.1f} GB")

    try:
        import ultralytics

        print(f"  Ultralytics:     {ultralytics.__version__}")
    except ImportError:
        print(
            "[ERROR] Ultralytics is not installed. Install via: pip install ultralytics"
        )
        sys.exit(1)

    return vram_gb


def load_train_config():
    """Load hyperparameters from the training YAML configuration file."""
    if not TRAIN_YAML.exists():
        print(
            f"[WARNING] Config file not found at {TRAIN_YAML}. Falling back to defaults."
        )
        return {}

    with open(TRAIN_YAML, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    print(f"[INFO] Configuration loaded from {TRAIN_YAML}")
    return config


def get_next_version():
    """Determine the next sequential run version number."""
    if not PROJECT_DIR.exists():
        return 1

    versions = []
    for entry in PROJECT_DIR.iterdir():
        if entry.is_dir() and entry.name.startswith("v"):
            try:
                version_num = int(entry.name.split("_")[0].lstrip("v"))
                versions.append(version_num)
            except (ValueError, IndexError):
                continue

    return max(versions, default=0) + 1


def count_epochs(save_dir):
    """Count total completed training epochs from results.csv."""
    results_csv = Path(save_dir) / "results.csv"
    if not results_csv.exists():
        return 0

    with open(results_csv, "r", encoding="utf-8") as file:
        return sum(1 for _ in file) - 1


def generate_summary(
    save_dir, config_snapshot, model_name, imgsz, test_results, elapsed_seconds
):
    """Generate summary markdown document containing run configurations and metrics."""
    save_dir = Path(save_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    duration_hours = elapsed_seconds / 3600
    epochs_trained = count_epochs(save_dir)
    version = save_dir.name.split("_")[0]

    lines = [
        f"# Training Run {version} - Summary\n",
        f"| Attribute | Value |",
        f"|---|---|",
        f"| **Date** | {timestamp} |",
        f"| **Duration** | {duration_hours:.1f} hours |",
        f"| **Model Architecture** | `{model_name}` |",
        f"| **Image Resolution** | {imgsz} |",
        f"| **Epochs Completed** | {epochs_trained} / {config_snapshot.get('epochs', 'N/A')} |\n",
    ]

    if test_results and hasattr(test_results, "box"):
        box = test_results.box
        lines.extend(
            [
                "## Test Split Evaluation Results\n",
                "| Metric | Value |",
                "|---|---|",
                f"| **mAP50** | **{box.map50:.4f}** |",
                f"| **mAP50-95** | **{box.map:.4f}** |",
                f"| **Precision** | **{box.mp:.4f}** |",
                f"| **Recall** | **{box.mr:.4f}** |\n",
            ]
        )

    lines.extend(
        [
            "## Training Configuration\n",
            "| Parameter | Value |",
            "|---|---|",
        ]
    )

    config_display = [
        ("Model", f"`{model_name}`"),
        ("Image Size", imgsz),
        ("Batch Size", config_snapshot.get("batch", "auto")),
        ("Epochs (max)", config_snapshot.get("epochs", "N/A")),
        ("Patience", config_snapshot.get("patience", "N/A")),
        ("Optimizer", config_snapshot.get("optimizer", "auto")),
        (
            "Learning Rate",
            f"{config_snapshot.get('lr0', 0.01)} -> {config_snapshot.get('lrf', 0.01)}",
        ),
        ("Cosine LR Scheduler", config_snapshot.get("cos_lr", False)),
        ("Warmup Epochs", config_snapshot.get("warmup_epochs", 3.0)),
        ("Mosaic Augmentation", config_snapshot.get("mosaic", 1.0)),
        ("MixUp Augmentation", config_snapshot.get("mixup", 0.0)),
        ("Scale Jitter", config_snapshot.get("scale", 0.5)),
    ]

    for param, value in config_display:
        lines.append(f"| {param} | {value} |")

    lines.extend(
        [
            "\n## Saved Artifacts\n",
            "| File | Description |",
            "|---|---|",
            "| `weights/best.pt` | Best model checkpoint weights |",
            "| `results.csv` | Epoch-level metrics history |",
            "| `results.png` | Training loss and mAP progression plots |",
            "| `confusion_matrix.png` | Normalized confusion matrix |",
            "| `BoxPR_curve.png` | Precision-Recall curve |",
            "| `val_batch0_pred.jpg` | Validation batch sample predictions |",
            "| `val_batch0_labels.jpg` | Ground truth annotation references |\n",
        ]
    )

    summary_path = save_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Run summary written to {summary_path}")


def cleanup_run_dir(save_dir):
    """Remove non-essential temporary checkpoint files to save storage space."""
    keep_files = {
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "BoxPR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
        "summary.md",
    }
    keep_weights = {"best.pt"}

    save_dir = Path(save_dir)
    removed_count = 0

    for file_path in save_dir.iterdir():
        if file_path.is_file() and file_path.name not in keep_files:
            file_path.unlink()
            removed_count += 1

    weights_dir = save_dir / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.iterdir():
            if weight_file.name not in keep_weights:
                weight_file.unlink()
                removed_count += 1

    print(
        f"[INFO] Storage cleanup: removed {removed_count} temporary files from {save_dir.name}"
    )


def update_history(
    save_dir, model_name, imgsz, config_snapshot, test_results, elapsed_seconds
):
    """Append training results to history.csv for record keeping."""
    version = Path(save_dir).name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    duration_hours = elapsed_seconds / 3600
    epochs_trained = count_epochs(save_dir)

    map50 = map5095 = precision = recall = ""
    if test_results and hasattr(test_results, "box"):
        box = test_results.box
        map50 = f"{box.map50:.4f}"
        map5095 = f"{box.map:.4f}"
        precision = f"{box.mp:.4f}"
        recall = f"{box.mr:.4f}"

    header = [
        "version",
        "date",
        "model",
        "imgsz",
        "batch",
        "epochs",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "hours",
    ]
    row = [
        version,
        timestamp,
        model_name,
        imgsz,
        config_snapshot.get("batch", "auto"),
        epochs_trained,
        map50,
        map5095,
        precision,
        recall,
        f"{duration_hours:.1f}",
    ]

    write_header = not HISTORY_CSV.exists()
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[INFO] History ledger updated at {HISTORY_CSV}")


def cleanup_test_eval():
    """Remove redundant validation evaluation folders created during test evaluation."""
    for dir_entry in PROJECT_DIR.iterdir():
        if dir_entry.is_dir() and dir_entry.name.startswith("val"):
            shutil.rmtree(dir_entry)


def create_progress_callbacks(training_start_time, val_interval=1):
    """Create progress callbacks for terminal monitoring."""
    state = {
        "best_map50": 0.0,
        "best_epoch": 0,
        "header_printed": False,
        "val_interval": val_interval,
        "val_ran_this_epoch": True,
    }

    def on_train_start(trainer):
        import torch

        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_used = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[TRAIN] Active VRAM allocation: {vram_used:.1f} / {vram_total:.1f} GB")
        if val_interval > 1:
            print(f"[TRAIN] Validation interval: every {val_interval} epochs")

    def on_train_epoch_end(trainer):
        if val_interval <= 1:
            return

        epoch = trainer.epoch + 1
        epochs = trainer.epochs
        should_val = (epoch == 1) or (epoch % val_interval == 0) or (epoch >= epochs)
        trainer.args.val = should_val
        state["val_ran_this_epoch"] = should_val

    def on_fit_epoch_end(trainer):
        import torch

        epoch = trainer.epoch + 1
        epochs = trainer.epochs
        val_ran = state["val_ran_this_epoch"]

        if not state["header_printed"]:
            print(
                f"  {'':>2} {'Epoch':>8}  "
                f"{'Box':>7} {'Cls':>7} {'DFL':>7}  "
                f"{'mAP50':>7} {'mAP50-95':>10}  "
                f"{'VRAM':>6}  {'ETA':>8}"
            )
            state["header_printed"] = True

        box_loss = cls_loss = dfl_loss = 0.0
        if trainer.tloss is not None:
            try:
                if isinstance(trainer.tloss, torch.Tensor):
                    loss_vals = trainer.tloss.cpu().tolist()
                    if isinstance(loss_vals, list) and len(loss_vals) >= 3:
                        box_loss, cls_loss, dfl_loss = (
                            loss_vals[0],
                            loss_vals[1],
                            loss_vals[2],
                        )
                    elif isinstance(loss_vals, (int, float)):
                        box_loss = loss_vals
            except (IndexError, TypeError, RuntimeError):
                pass

        if val_ran:
            metrics = trainer.metrics or {}
            map50 = metrics.get("metrics/mAP50(B)", 0)
            map5095 = metrics.get("metrics/mAP50-95(B)", 0)
            map50_str = f"{map50:>7.4f}"
            map5095_str = f"{map5095:>10.4f}"
        else:
            map50 = 0
            map50_str = f"{'-':>7}"
            map5095_str = f"{'-':>10}"

        vram_used = torch.cuda.memory_reserved(0) / (1024**3)
        elapsed = time.time() - training_start_time
        avg_per_epoch = elapsed / epoch
        remaining_secs = avg_per_epoch * (epochs - epoch)
        hours = int(remaining_secs // 3600)
        minutes = int((remaining_secs % 3600) // 60)
        eta_str = f"{hours}h{minutes:02d}m" if hours > 0 else f"{minutes}m"

        is_best = val_ran and map50 > state["best_map50"] and map50 > 0
        if is_best:
            state["best_map50"] = map50
            state["best_epoch"] = epoch
        marker = "*" if is_best else " "

        print(
            f"  {marker} {epoch:>4}/{epochs:<4}  "
            f"{box_loss:>7.4f} {cls_loss:>7.4f} {dfl_loss:>7.4f}  "
            f"{map50_str} {map5095_str}  "
            f"{vram_used:>5.1f}G  {eta_str:>8}"
        )

        state["val_ran_this_epoch"] = True

    def on_train_end(trainer):
        elapsed = time.time() - training_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        time_str = f"{hours}h{minutes:02d}m" if hours > 0 else f"{minutes}m"
        print(f"\n[TRAIN] Training finished in {time_str}")
        print(
            f"[TRAIN] Optimal epoch: {state['best_epoch']} (mAP50 = {state['best_map50']:.4f})"
        )

    return {
        "on_train_start": on_train_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def train(args):
    """Execute YOLO training workflow with auto-versioning and evaluation."""
    from ultralytics import YOLO

    vram_gb = check_environment()
    config = load_train_config()
    config_snapshot = dict(config)

    model_name = args.model or config.pop("model", "yolo11s.pt")
    imgsz = args.imgsz or config.pop("imgsz", 1280)

    config_snapshot["model"] = model_name
    config_snapshot["imgsz"] = imgsz

    # Dynamic VRAM scaling
    if vram_gb < 4.0:
        print(
            f"[WARNING] Limited VRAM ({vram_gb:.1f} GB). Switching to yolo11n.pt at imgsz 960."
        )
        model_name = "yolo11n.pt"
        imgsz = 960
    elif vram_gb < 5.5:
        print(f"[WARNING] Moderate VRAM ({vram_gb:.1f} GB). Scaling imgsz to 960.")
        imgsz = 960

    # Auto-versioning run naming
    if args.resume:
        existing = (
            sorted(
                [
                    d
                    for d in PROJECT_DIR.iterdir()
                    if d.is_dir() and d.name.startswith("v")
                ],
                key=lambda d: d.stat().st_mtime,
            )
            if PROJECT_DIR.exists()
            else []
        )

        if existing:
            run_name = existing[-1].name
            last_checkpoint = existing[-1] / "weights" / "last.pt"
            if last_checkpoint.exists():
                model_name = str(last_checkpoint)
                print(f"[INFO] Resuming training checkpoint: {last_checkpoint}")
            else:
                print("[WARNING] Checkpoint last.pt not found. Initializing new run.")
                version = get_next_version()
                tag = Path(model_name).stem
                run_name = f"v{version}_{tag}_{imgsz}"
        else:
            version = get_next_version()
            tag = Path(model_name).stem
            run_name = f"v{version}_{tag}_{imgsz}"
    else:
        version = get_next_version()
        tag = Path(model_name).stem
        run_name = f"v{version}_{tag}_{imgsz}"

    print(
        f"\n[TRAIN] Setup: Run={run_name} Model={model_name} ImgSize={imgsz} Batch={config.get('batch', 8)}"
    )
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Output:   {PROJECT_DIR / run_name}")

    model = YOLO(model_name)
    train_args = {
        "data": str(DATA_YAML),
        "imgsz": imgsz,
        "project": str(PROJECT_DIR),
        "name": run_name,
        "exist_ok": True,
        "device": 0,
        "verbose": False,
        "deterministic": False,
        "amp": True,
        "cache": "disk",
        "resume": args.resume,
    }
    train_args.update(config)

    start_time = time.time()
    callbacks = create_progress_callbacks(start_time, val_interval=2)
    for event_name, callback_fn in callbacks.items():
        model.add_callback(event_name, callback_fn)

    print("\n[TRAIN] Initiating training loop...")
    model.train(**train_args)
    elapsed = time.time() - start_time

    # Evaluate best checkpoint on test set
    print("\n[EVAL] Evaluating best checkpoint on test split...")
    best_weights_path = model.trainer.best
    best_model = YOLO(best_weights_path)
    test_results = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        device=0,
        verbose=True,
        plots=False,
    )

    save_dir = model.trainer.save_dir
    print("\n[INFO] Saving run summary, pruning artifacts, and archiving metrics...")
    generate_summary(
        save_dir, config_snapshot, model_name, imgsz, test_results, elapsed
    )
    cleanup_run_dir(save_dir)
    update_history(save_dir, model_name, imgsz, config_snapshot, test_results, elapsed)
    cleanup_test_eval()

    print(f"\n[TRAIN] Run complete: {run_name}")
    print(f"  Best model checkpoint: {Path(save_dir) / 'weights' / 'best.pt'}")
    print(f"  Markdown summary:     {Path(save_dir) / 'summary.md'}")
    print(f"  Run duration:         {elapsed / 3600:.2f} hours")

    if test_results and hasattr(test_results, "box"):
        box = test_results.box
        print("\n[EVAL] Final Test Metrics:")
        print(f"  mAP50:     {box.map50:.4f}")
        print(f"  mAP50-95:  {box.map:.4f}")
        print(f"  Precision: {box.mp:.4f}")
        print(f"  Recall:    {box.mr:.4f}")


def parse_args():
    """Parse command-line arguments for training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11 model on UAV human dataset"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last.pt"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model checkpoint or architecture name"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Input image resolution"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
