#!/usr/bin/env python3
"""Run the TENSOR/PyTorch benchmark suite with repeated trials.

Outputs:
  benchmark_outputs/gpt_summary.json
  benchmark_outputs/gpt_summary.csv
  benchmark_outputs/matmul_summary.json
"""
import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


MODEL_SIZES = {
    "small": dict(max_seq_len=128, embed_dim=64, num_heads=4, num_layers=2, ff_dim=256, batch_size=2, seq_len=16, measured_steps=40),
    "medium": dict(max_seq_len=128, embed_dim=128, num_heads=8, num_layers=3, ff_dim=512, batch_size=4, seq_len=32, measured_steps=40),
    "large": dict(max_seq_len=256, embed_dim=256, num_heads=8, num_layers=4, ff_dim=1024, batch_size=4, seq_len=64, measured_steps=20),
}


def run(cmd, cwd, env=None):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, cwd=cwd, env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)


def stat_block(values):
    values = [float(v) for v in values]
    return {
        "trials": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tensor_env(cfg, args, seed, gemm_backend="cublas"):
    env = {
        "TENSOR_MAX_SEQ_LEN": cfg["max_seq_len"],
        "TENSOR_EMBED_DIM": cfg["embed_dim"],
        "TENSOR_NUM_HEADS": cfg["num_heads"],
        "TENSOR_NUM_LAYERS": cfg["num_layers"],
        "TENSOR_FF_DIM": cfg["ff_dim"],
        "TENSOR_BATCH_SIZE": cfg["batch_size"],
        "TENSOR_SEQ_LEN": cfg["seq_len"],
        "TENSOR_WARMUP_STEPS": args.warmup_steps,
        "TENSOR_MEASURED_STEPS": cfg["measured_steps"],
    }
    # Keep the GPU benchmark in fair throughput mode by default.
    env.pop("TENSOR_PROFILE_PHASES", None)
    env.pop("TENSOR_PROFILE_BACKWARD", None)
    env["TENSOR_GEMM_BACKEND"] = gemm_backend
    return env


def pytorch_cmd(args, cfg, device, output, seed):
    return [
        sys.executable,
        "scripts/benchmark_pytorch_gpt.py",
        "--data", args.data,
        "--output", str(output),
        "--device", device,
        "--seed", str(seed),
        "--max-seq-len", str(cfg["max_seq_len"]),
        "--embed-dim", str(cfg["embed_dim"]),
        "--num-heads", str(cfg["num_heads"]),
        "--num-layers", str(cfg["num_layers"]),
        "--ff-dim", str(cfg["ff_dim"]),
        "--batch-size", str(cfg["batch_size"]),
        "--seq-len", str(cfg["seq_len"]),
        "--warmup-steps", str(args.warmup_steps),
        "--measured-steps", str(cfg["measured_steps"]),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--torch-threads", str(args.torch_threads),
    ]


def summarize_gpt_case(label, framework, device, size_name, cfg, trial_results, gemm_backend="n/a"):
    metrics = [r["metrics"] for r in trial_results]
    return {
        "label": label,
        "framework": framework,
        "device": device,
        "model_size": size_name,
        "gemm_backend": gemm_backend,
        "model_config": trial_results[0]["model_config"],
        "training_config": trial_results[0]["training_config"],
        "hardware": trial_results[0].get("hardware", {}),
        "time_per_step_ms": stat_block([m["time_per_step_ms"] for m in metrics]),
        "tokens_per_sec": stat_block([m["tokens_per_sec"] for m in metrics]),
        "steps_per_sec": stat_block([m["steps_per_sec"] for m in metrics]),
        "final_loss": stat_block([m["final_loss"] for m in metrics if m.get("final_loss") is not None]) if any(m.get("final_loss") is not None for m in metrics) else None,
    }


def run_gpt_suite(args, outdir):
    build = Path(args.build_dir)
    cases = []
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]

    tensor_gemm_backends = [x.strip() for x in args.tensor_gemm_backends.split(",") if x.strip()]
    for size_name in sizes:
        cfg = MODEL_SIZES[size_name]
        case_specs = [("TENSOR", "cpu", "n/a"), ("PyTorch", "cpu", "n/a")]
        case_specs.extend(("TENSOR", "cuda", backend) for backend in tensor_gemm_backends)
        case_specs.append(("PyTorch", "cuda", "n/a"))

        for framework, device, gemm_backend in case_specs:
            if args.cpu_only and device == "cuda":
                continue
            if args.gpu_only and device == "cpu":
                continue

            trial_results = []
            backend_suffix = f"_{gemm_backend}" if framework == "TENSOR" and device == "cuda" else ""
            label = f"{framework}_{device}{backend_suffix}_{size_name}"
            for trial in range(args.trials):
                seed = args.seed + trial
                trial_out = outdir / "trials" / f"{label}_trial{trial}.json"
                trial_out.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if framework == "TENSOR" and device == "cpu":
                        exe = build / "gpt_benchmark"
                        if not exe.exists():
                            raise FileNotFoundError(exe)
                        run([str(exe), args.data], cwd=args.repo_root, env=tensor_env(cfg, args, seed, "cublas"))
                        shutil.copyfile(Path(args.repo_root) / "gpt_benchmark_results.json", trial_out)
                    elif framework == "TENSOR" and device == "cuda":
                        exe = build / "gpt_benchmark_gpu"
                        if not exe.exists():
                            raise FileNotFoundError(exe)
                        run([str(exe), args.data], cwd=args.repo_root, env=tensor_env(cfg, args, seed, gemm_backend))
                        shutil.copyfile(Path(args.repo_root) / "gpt_benchmark_gpu_results.json", trial_out)
                    else:
                        run(pytorch_cmd(args, cfg, device, trial_out, seed), cwd=args.repo_root)
                except Exception as exc:
                    if device == "cuda" and args.skip_missing_gpu:
                        print(f"Skipping {label}: {exc}")
                        trial_results = []
                        break
                    raise

                trial_results.append(read_json(trial_out))

            if trial_results:
                cases.append(summarize_gpt_case(label, framework, device, size_name, cfg, trial_results, gemm_backend))

    summary_path = outdir / "gpt_summary.json"
    summary_path.write_text(json.dumps({"generated_at_unix": time.time(), "cases": cases}, indent=2) + "\n", encoding="utf-8")

    csv_path = outdir / "gpt_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "framework", "device", "model_size", "gemm_backend", "median_ms_step", "stddev_ms_step", "median_tokens_sec", "stddev_tokens_sec", "trials"])
        writer.writeheader()
        for c in cases:
            writer.writerow({
                "label": c["label"],
                "framework": c["framework"],
                "device": c["device"],
                "model_size": c["model_size"],
                "gemm_backend": c.get("gemm_backend", "n/a"),
                "median_ms_step": c["time_per_step_ms"]["median"],
                "stddev_ms_step": c["time_per_step_ms"]["stddev"],
                "median_tokens_sec": c["tokens_per_sec"]["median"],
                "stddev_tokens_sec": c["tokens_per_sec"]["stddev"],
                "trials": c["time_per_step_ms"]["trials"],
            })

    return summary_path


def run_matmul_suite(args, outdir):
    if args.cpu_only:
        return None
    exe = Path(args.build_dir) / "matmul_benchmark"
    if not exe.exists():
        if args.skip_missing_gpu:
            print(f"Skipping matmul benchmark: {exe} not found")
            return None
        raise FileNotFoundError(exe)

    trial_sets = []
    for trial in range(args.trials):
        trial_out = outdir / "trials" / f"matmul_trial{trial}.json"
        run([str(exe), str(trial_out)], cwd=args.repo_root, env={"TENSOR_BENCH_TRIAL": trial})
        trial_sets.append(read_json(trial_out))

    grouped = {}
    for rows in trial_sets:
        per_trial = {}
        for r in rows:
            key = (r["dtype"], r["kernel_name"], r["M"], r["K"], r["N"])
            per_trial.setdefault(key, []).append(r)
        for key, duplicates in per_trial.items():
            # Some raw sweeps intentionally hit the same shape through both square
            # and rectangular lists. Collapse those duplicates so one benchmark
            # process contributes one value per key.
            representative = dict(duplicates[0])
            representative["time_ms"] = statistics.median([x["time_ms"] for x in duplicates])
            representative["tflops"] = statistics.median([x["tflops"] for x in duplicates])
            grouped.setdefault(key, []).append(representative)

    # Baseline medians by dtype/shape, computed from the cuBLAS rows in this same run set.
    baseline = {}
    for key, rows in grouped.items():
        dtype, kernel, M, K, N = key
        if kernel.startswith("cuBLAS"):
            baseline[(dtype, M, K, N)] = statistics.median([x["tflops"] for x in rows])

    summary = []
    for key, rows in sorted(grouped.items()):
        dtype, kernel, M, K, N = key
        times = [x["time_ms"] for x in rows]
        tflops = [x["tflops"] for x in rows]
        pct_values = []
        base = baseline.get((dtype, M, K, N))
        if base and base > 0:
            pct_values = [100.0 * x / base for x in tflops]
        summary.append({
            "dtype": dtype,
            "kernel_name": kernel,
            "M": M,
            "K": K,
            "N": N,
            "time_ms": stat_block(times),
            "tflops": stat_block(tflops),
            "pct_of_cublas": stat_block(pct_values) if pct_values else None,
            "gpu_name": rows[0].get("gpu_name"),
        })

    path = outdir / "matmul_summary.json"
    path.write_text(json.dumps({"generated_at_unix": time.time(), "results": summary}, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--output-dir", default="benchmark_outputs")
    ap.add_argument("--sizes", default="small,medium,large", help="Comma-separated: small,medium,large")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--tensor-gemm-backends", default="cublas,custom", help="Comma-separated TENSOR CUDA GEMM backends: cublas,custom")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--skip-missing-gpu", action="store_true", default=True)
    ap.add_argument("--skip-gpt", action="store_true")
    ap.add_argument("--skip-matmul", action="store_true")
    args = ap.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        if size not in MODEL_SIZES:
            raise SystemExit(f"unknown model size: {size}")

    args.repo_root = str(Path(args.repo_root).resolve())
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        args.build_dir = str(Path(args.repo_root) / build_dir)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_gpt:
        gpt_summary = run_gpt_suite(args, outdir)
        print(f"Wrote {gpt_summary}")
    if not args.skip_matmul:
        matmul_summary = run_matmul_suite(args, outdir)
        if matmul_summary:
            print(f"Wrote {matmul_summary}")


if __name__ == "__main__":
    main()