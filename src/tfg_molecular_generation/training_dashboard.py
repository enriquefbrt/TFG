import argparse
import ast
import csv
import math
import os
import re

DICT_LINE_PATTERN = re.compile(r"\{.*?\}")
KV_PATTERN = re.compile(
    r"""['"]?(loss|grad_norm|learning_rate|epoch|step)['"]?\s*[:=]\s*([-+]?[\d.]+(?:e[-+]?\d+)?|nan|inf|-inf)""",
    re.IGNORECASE,
)


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None
    lowered = text.lower()
    if lowered == "nan":
        return float("nan")
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    try:
        return float(text)
    except ValueError:
        return None


def parse_dict_from_line(line):
    match = DICT_LINE_PATTERN.search(line)
    if not match:
        return None
    payload = match.group(0)
    try:
        parsed = ast.literal_eval(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    keys = {str(k) for k in parsed.keys()}
    interesting = {"loss", "grad_norm", "learning_rate", "epoch", "step"}
    if keys.isdisjoint(interesting):
        return None
    return parsed


def parse_kv_from_line(line):
    matches = KV_PATTERN.findall(line)
    if not matches:
        return None
    out = {}
    for key, value in matches:
        out[key] = value
    return out


def parse_log(log_path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parsed = parse_dict_from_line(line)
            if parsed is None:
                parsed = parse_kv_from_line(line)
            if parsed is None:
                continue

            row = {
                "line_number": line_number,
                "epoch": to_float(parsed.get("epoch")),
                "loss": to_float(parsed.get("loss")),
                "grad_norm": to_float(parsed.get("grad_norm")),
                "learning_rate": to_float(parsed.get("learning_rate")),
                "step": to_float(parsed.get("step")),
            }

            if (
                row["epoch"] is None
                and row["loss"] is None
                and row["grad_norm"] is None
                and row["learning_rate"] is None
            ):
                continue

            rows.append(row)
    return rows


def sanitize_for_plot(values):
    clean = []
    for value in values:
        if value is None:
            clean.append(float("nan"))
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            clean.append(float("nan"))
            continue
        clean.append(value)
    return clean


def maybe_smooth(values, window):
    if window <= 1:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = [v for v in values[start : idx + 1] if not (isinstance(v, float) and math.isnan(v))]
        if not chunk:
            smoothed.append(float("nan"))
        else:
            smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["line_number", "epoch", "step", "loss", "grad_norm", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_dashboard(rows, output_png, smoothing_window=1, title=None):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to build the dashboard image. "
            "Install it with: pip install matplotlib"
        ) from exc

    if not rows:
        raise ValueError("No metric rows were parsed from the log.")

    epochs = [row["epoch"] if row["epoch"] is not None else idx for idx, row in enumerate(rows)]
    loss = sanitize_for_plot([row["loss"] for row in rows])
    grad_norm = sanitize_for_plot([row["grad_norm"] for row in rows])
    learning_rate = sanitize_for_plot([row["learning_rate"] for row in rows])

    if smoothing_window > 1:
        loss_plot = maybe_smooth(loss, smoothing_window)
        grad_plot = maybe_smooth(grad_norm, smoothing_window)
        lr_plot = maybe_smooth(learning_rate, smoothing_window)
    else:
        loss_plot = loss
        grad_plot = grad_norm
        lr_plot = learning_rate

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(title or "Training Dashboard", fontsize=14)

    axes[0].plot(epochs, loss_plot, color="#1f77b4", linewidth=1.4)
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, grad_plot, color="#d62728", linewidth=1.4)
    axes[1].set_ylabel("Grad Norm")
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, lr_plot, color="#2ca02c", linewidth=1.4)
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build a dashboard (loss/grad_norm/lr vs epoch) from a training .log file."
    )
    parser.add_argument("--log_file", type=str, required=True, help="Path to training log file.")
    parser.add_argument(
        "--output_png",
        type=str,
        default="outputs/training_dashboard.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/training_metrics_from_log.csv",
        help="Parsed metrics CSV path.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=1,
        help="Moving average window size for plotting (1 disables smoothing).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional dashboard title.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        raise FileNotFoundError(f"log_file not found: {args.log_file}")
    if args.smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1")

    rows = parse_log(args.log_file)
    if not rows:
        raise ValueError(
            "No metrics found in log. Expected lines with keys like loss/grad_norm/learning_rate/epoch."
        )

    write_csv(rows, args.output_csv)
    build_dashboard(
        rows=rows,
        output_png=args.output_png,
        smoothing_window=args.smoothing_window,
        title=args.title,
    )

    print(f"Parsed rows: {len(rows)}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved dashboard: {args.output_png}")


if __name__ == "__main__":
    main()
