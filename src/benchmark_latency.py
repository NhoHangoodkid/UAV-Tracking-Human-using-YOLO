"""Latency and throughput benchmark module for PyTorch and ONNX Runtime inference engines."""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PT = ROOT / "runs" / "detect" / "v1_yolo11s_1280" / "weights" / "best.pt"
DEFAULT_ONNX = ROOT / "runs" / "detect" / "v1_yolo11s_1280" / "weights" / "best.onnx"
OUTPUT_DIR = ROOT / "output" / "benchmark"


def benchmark_pytorch(
    weights_path, imgsz=1280, half=False, iterations=100, warmup=20, device="cuda:0"
):
    """Benchmark PyTorch model inference speed using CUDA event timers."""
    precision_label = "FP16" if half else "FP32"
    print(f"\n[BENCHMARK] Testing PyTorch {precision_label} on {device}...")

    model = YOLO(str(weights_path))
    model.to(device)

    dummy_input = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        dtype=torch.float16 if half else torch.float32,
        device=device,
    )
    raw_model = model.model
    if half:
        raw_model.half()
    raw_model.eval()

    # GPU warmup phase
    with torch.no_grad():
        for _ in range(warmup):
            _ = raw_model(dummy_input)
    torch.cuda.synchronize()

    # Timed benchmarking phase with CUDA events
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(iterations):
            starter.record()
            _ = raw_model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    file_size_mb = Path(weights_path).stat().st_size / (1024 * 1024)
    mean_latency = float(np.mean(timings))
    fps = (1000.0 / mean_latency) if mean_latency > 0 else 0.0

    return {
        "Engine": f"PyTorch {precision_label} (CUDA)",
        "Precision": precision_label,
        "Size (MB)": f"{file_size_mb:.1f} MB",
        "Mean (ms)": mean_latency,
        "P50 (ms)": float(np.median(timings)),
        "P95 (ms)": float(np.percentile(timings, 95)),
        "P99 (ms)": float(np.percentile(timings, 99)),
        "Throughput (FPS)": fps,
    }


def benchmark_onnx(onnx_path, imgsz=1280, iterations=100, warmup=20):
    """Benchmark ONNX Runtime execution provider latency and throughput."""
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    selected_provider = (
        ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else ["CPUExecutionProvider"]
    )
    provider_name = "CUDA" if "CUDAExecutionProvider" in selected_provider else "CPU"
    print(f"\n[BENCHMARK] Testing ONNX Runtime with provider: {provider_name}...")

    session = ort.InferenceSession(str(onnx_path), providers=selected_provider)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup phase
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Timed inference loops
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    mean_latency = float(np.mean(timings))
    fps = (1000.0 / mean_latency) if mean_latency > 0 else 0.0

    return {
        "Engine": f"ONNX Runtime ({provider_name})",
        "Precision": "FP32",
        "Size (MB)": f"{file_size_mb:.1f} MB",
        "Mean (ms)": mean_latency,
        "P50 (ms)": float(np.median(timings)),
        "P95 (ms)": float(np.percentile(timings, 95)),
        "P99 (ms)": float(np.percentile(timings, 99)),
        "Throughput (FPS)": fps,
    }


def main():
    """Run comparative latency benchmark across PyTorch and ONNX engines."""
    parser = argparse.ArgumentParser(
        description="Benchmark Model Inference Latency and Throughput"
    )
    parser.add_argument(
        "--pt", type=str, default=str(DEFAULT_PT), help="Path to PyTorch weights"
    )
    parser.add_argument(
        "--onnx", type=str, default=str(DEFAULT_ONNX), help="Path to ONNX weights"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Square input image resolution"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iteration count"
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iteration count")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. PyTorch FP32
    if Path(args.pt).exists() and torch.cuda.is_available():
        res_pt32 = benchmark_pytorch(
            args.pt,
            imgsz=args.imgsz,
            half=False,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        results.append(res_pt32)

        # 2. PyTorch FP16
        res_pt16 = benchmark_pytorch(
            args.pt,
            imgsz=args.imgsz,
            half=True,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        results.append(res_pt16)

    # 3. ONNX Runtime
    if Path(args.onnx).exists():
        res_onnx = benchmark_onnx(
            args.onnx,
            imgsz=args.imgsz,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        results.append(res_onnx)

    print(
        f"\n[BENCHMARK] Latency and Throughput Summary (Input: 1x3x{args.imgsz}x{args.imgsz})"
    )
    header = (
        f"{'Inference Engine':<25} | {'Precision':<10} | {'Size':<10} | "
        f"{'Mean (ms)':<10} | {'P95 (ms)':<10} | {'FPS':<10}"
    )
    print(f"  {header}")

    for r in results:
        print(
            f"  {r['Engine']:<25} | {r['Precision']:<10} | {r['Size (MB)']:<10} | "
            f"{r['Mean (ms)']:<10.2f} | {r['P95 (ms)']:<10.2f} | {r['Throughput (FPS)']:<10.1f}"
        )

    # Save Markdown report in English
    report_path = OUTPUT_DIR / "latency_benchmark.md"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# UAV Model Deployment: Latency & Throughput Benchmark\n\n")
        file.write(f"- **Device:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.1)\n")
        file.write(f"- **Input Shape:** `1x3x{args.imgsz}x{args.imgsz}`\n")
        file.write(
            f"- **Benchmark Iterations:** {args.iterations} runs (Warmup: {args.warmup} runs)\n\n"
        )
        file.write(
            "| Engine / Format | Precision | Model Size | Mean Latency (ms) | P50 (ms) | P95 (ms) | Throughput (FPS) |\n"
        )
        file.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            file.write(
                f"| **{r['Engine']}** | {r['Precision']} | {r['Size (MB)']} | "
                f"**{r['Mean (ms)']:.2f} ms** | {r['P50 (ms)']:.2f} ms | {r['P95 (ms)']:.2f} ms | "
                f"**{r['Throughput (FPS)']:.1f} FPS** |\n"
            )

        file.write("\n### Deployment Guidelines for NVIDIA Jetson Edge AI Devices:\n")
        file.write(
            "When deploying models onto embedded UAV computers such as NVIDIA Jetson Orin Nano:\n"
        )
        file.write(
            "1. Use the pre-installed `trtexec` tool in JetPack to build a TensorRT engine from `best.onnx`:\n"
        )
        file.write("   ```bash\n")
        file.write(
            "   /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw\n"
        )
        file.write("   ```\n")
        file.write(
            "2. Enabling FP16 mode reduces memory footprint by approximately 50% and delivers 1.5x to 2.5x speedups over PyTorch.\n"
        )

    print(f"\n[BENCHMARK] Latency report generated at: {report_path}")


if __name__ == "__main__":
    main()
