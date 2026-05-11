import argparse
import ast
import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional


DICT_LINE_RE = re.compile(r"\{.*?\}")


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


def find_latest_log(logs_dir: str, pattern: str = "pretrain*.log") -> str:
    candidates = glob.glob(os.path.join(logs_dir, pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No log files found in {logs_dir} with pattern '{pattern}'."
        )
    return max(candidates, key=os.path.getmtime)


def parse_log(log_path: str):
    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = DICT_LINE_RE.search(line)
            if not match:
                continue

            try:
                payload = ast.literal_eval(match.group(0))
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            # Eval rows
            if "eval_loss" in payload:
                eval_rows.append(
                    {
                        "line_number": line_number,
                        "epoch": to_float(payload.get("epoch")),
                        "eval_loss": to_float(payload.get("eval_loss")),
                        "eval_runtime": to_float(payload.get("eval_runtime")),
                        "eval_samples_per_second": to_float(
                            payload.get("eval_samples_per_second")
                        ),
                        "eval_steps_per_second": to_float(
                            payload.get("eval_steps_per_second")
                        ),
                    }
                )
                continue

            # Train rows
            if any(
                key in payload for key in ("loss", "grad_norm", "learning_rate", "epoch", "step")
            ):
                train_rows.append(
                    {
                        "line_number": line_number,
                        "epoch": to_float(payload.get("epoch")),
                        "step": to_float(payload.get("step")),
                        "loss": to_float(payload.get("loss")),
                        "grad_norm": to_float(payload.get("grad_norm")),
                        "learning_rate": to_float(payload.get("learning_rate")),
                    }
                )

    return train_rows, eval_rows


def find_latest_trainer_state(checkpoints_dir: str) -> str:
    candidates = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*", "trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No trainer_state.json found under {checkpoints_dir}/checkpoint-*/"
        )

    def checkpoint_step(path: str) -> int:
        base = os.path.basename(os.path.dirname(path))  # checkpoint-<step>
        try:
            return int(base.split("-")[-1])
        except Exception:
            return -1

    return max(candidates, key=checkpoint_step)


def parse_trainer_state(trainer_state_path: str):
    with open(trainer_state_path, "r", encoding="utf-8", errors="replace") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []

    for idx, payload in enumerate(log_history, start=1):
        if not isinstance(payload, dict):
            continue

        # Eval rows
        if "eval_loss" in payload:
            eval_rows.append(
                {
                    "line_number": None,
                    "epoch": to_float(payload.get("epoch")),
                    "step": to_float(payload.get("step", payload.get("global_step"))),
                    "eval_loss": to_float(payload.get("eval_loss")),
                    "eval_runtime": to_float(payload.get("eval_runtime")),
                    "eval_samples_per_second": to_float(
                        payload.get("eval_samples_per_second")
                    ),
                    "eval_steps_per_second": to_float(
                        payload.get("eval_steps_per_second")
                    ),
                    "_order": idx,
                }
            )
            continue

        # Train rows
        if any(
            key in payload for key in ("loss", "grad_norm", "learning_rate", "epoch", "step")
        ):
            train_rows.append(
                {
                    "line_number": None,
                    "epoch": to_float(payload.get("epoch")),
                    "step": to_float(payload.get("step", payload.get("global_step"))),
                    "loss": to_float(payload.get("loss")),
                    "grad_norm": to_float(payload.get("grad_norm")),
                    "learning_rate": to_float(payload.get("learning_rate")),
                    "_order": idx,
                }
            )

    return train_rows, eval_rows


def _row_sort_key(row: Dict):
    epoch = row.get("epoch")
    step = row.get("step")
    line_number = row.get("line_number")
    order = row.get("_order")
    return (
        float("inf") if epoch is None else epoch,
        float("inf") if step is None else step,
        float("inf") if line_number is None else line_number,
        float("inf") if order is None else order,
    )


def _row_key_train(row: Dict):
    return (
        round(row.get("epoch"), 8) if row.get("epoch") is not None else None,
        round(row.get("step"), 8) if row.get("step") is not None else None,
        round(row.get("loss"), 8) if row.get("loss") is not None else None,
        round(row.get("grad_norm"), 8) if row.get("grad_norm") is not None else None,
        round(row.get("learning_rate"), 12)
        if row.get("learning_rate") is not None
        else None,
    )


def _row_key_eval(row: Dict):
    return (
        round(row.get("epoch"), 8) if row.get("epoch") is not None else None,
        round(row.get("step"), 8) if row.get("step") is not None else None,
        round(row.get("eval_loss"), 8) if row.get("eval_loss") is not None else None,
        round(row.get("eval_runtime"), 8)
        if row.get("eval_runtime") is not None
        else None,
        round(row.get("eval_samples_per_second"), 8)
        if row.get("eval_samples_per_second") is not None
        else None,
        round(row.get("eval_steps_per_second"), 8)
        if row.get("eval_steps_per_second") is not None
        else None,
    )


def merge_rows(primary: List[Dict], secondary: List[Dict], key_fn):
    merged = []
    seen = set()
    for row in primary + secondary:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    merged.sort(key=_row_sort_key)
    return merged


def write_csv(rows: List[Dict], output_csv: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def smooth(values: List[Optional[float]], window: int) -> List[float]:
    if window <= 1:
        return [float("nan") if v is None else v for v in values]

    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = [v for v in values[start : idx + 1] if v is not None]
        if not chunk:
            out.append(float("nan"))
        else:
            out.append(sum(chunk) / len(chunk))
    return out


def make_plots(
    train_rows: List[Dict],
    eval_rows: List[Dict],
    output_dir: str,
    smoothing_window: int,
    title_prefix: str,
) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate plots. Install it with: pip install matplotlib"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # Plot 1: train dashboard (loss, grad_norm, lr)
    if train_rows:
        x_train = [
            row["epoch"] if row.get("epoch") is not None else i for i, row in enumerate(train_rows)
        ]
        train_loss = smooth([row.get("loss") for row in train_rows], smoothing_window)
        train_grad = smooth([row.get("grad_norm") for row in train_rows], smoothing_window)
        train_lr = smooth([row.get("learning_rate") for row in train_rows], smoothing_window)

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        fig.suptitle(f"{title_prefix} - Train Dashboard", fontsize=14)

        axes[0].plot(x_train, train_loss, color="#1f77b4", linewidth=1.4)
        axes[0].set_ylabel("Train Loss")
        axes[0].grid(alpha=0.25)

        axes[1].plot(x_train, train_grad, color="#d62728", linewidth=1.4)
        axes[1].set_ylabel("Grad Norm")
        axes[1].grid(alpha=0.25)

        axes[2].plot(x_train, train_lr, color="#2ca02c", linewidth=1.4)
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(alpha=0.25)

        train_png = os.path.join(output_dir, "train_dashboard.png")
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.savefig(train_png, dpi=180)
        plt.close(fig)
        paths["train_dashboard_png"] = train_png

    # Plot 2: eval dashboard (train vs eval loss + eval throughput)
    if eval_rows:
        x_eval = [row["epoch"] for row in eval_rows]
        eval_loss = [row.get("eval_loss") for row in eval_rows]
        eval_sps = [row.get("eval_samples_per_second") for row in eval_rows]
        eval_stps = [row.get("eval_steps_per_second") for row in eval_rows]

        x_train_loss = []
        y_train_loss = []
        for row in train_rows:
            loss = row.get("loss")
            epoch = row.get("epoch")
            if loss is not None and epoch is not None:
                x_train_loss.append(epoch)
                y_train_loss.append(loss)
        y_train_loss = smooth(y_train_loss, smoothing_window)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{title_prefix} - Eval Dashboard", fontsize=14)

        if x_train_loss:
            axes[0].plot(
                x_train_loss,
                y_train_loss,
                label="train_loss (smoothed)",
                linewidth=1.2,
                color="#1f77b4",
            )
        axes[0].plot(x_eval, eval_loss, label="eval_loss", linewidth=2.0, color="#ff7f0e")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(
            x_eval,
            eval_sps,
            label="eval_samples_per_second",
            linewidth=1.5,
            color="#2ca02c",
        )
        axes[1].plot(
            x_eval,
            eval_stps,
            label="eval_steps_per_second",
            linewidth=1.5,
            color="#9467bd",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Throughput")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        eval_png = os.path.join(output_dir, "eval_dashboard.png")
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.savefig(eval_png, dpi=180)
        plt.close(fig)
        paths["eval_dashboard_png"] = eval_png

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/eval CSVs and plots from a pretrain log."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to a specific log file. If omitted, the latest pretrain*.log from --logs_dir is used.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored when --log_file is omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/log_plots",
        help="Directory where CSVs and PNGs will be saved.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=20,
        help="Moving average window for train curves.",
    )
    parser.add_argument(
        "--trainer_state_json",
        type=str,
        default=None,
        help=(
            "Optional path to trainer_state.json. If provided, metrics are merged with log "
            "and can recover earlier epochs not present in the current .log file."
        ),
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help=(
            "Optional checkpoints directory (e.g. models/t5_pretrain_scaffold_decorators). "
            "Latest checkpoint-*/trainer_state.json will be used."
        ),
    )
    parser.add_argument(
        "--skip_trainer_state",
        action="store_true",
        help="Ignore trainer_state.json even if provided/detected.",
    )
    args = parser.parse_args()

    if args.smoothing_window < 1:
        raise ValueError("--smoothing_window must be >= 1.")

    log_file = args.log_file or find_latest_log(args.logs_dir, "pretrain*.log")
    if not os.path.isfile(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    train_rows, eval_rows = parse_log(log_file)
    trainer_state_used = None

    if not args.skip_trainer_state:
        trainer_state_path = None
        if args.trainer_state_json:
            trainer_state_path = args.trainer_state_json
        elif args.checkpoints_dir:
            trainer_state_path = find_latest_trainer_state(args.checkpoints_dir)

        if trainer_state_path:
            if not os.path.isfile(trainer_state_path):
                raise FileNotFoundError(f"trainer_state.json not found: {trainer_state_path}")
            state_train_rows, state_eval_rows = parse_trainer_state(trainer_state_path)
            train_rows = merge_rows(train_rows, state_train_rows, _row_key_train)
            eval_rows = merge_rows(eval_rows, state_eval_rows, _row_key_eval)
            trainer_state_used = trainer_state_path

    if not train_rows and not eval_rows:
        raise ValueError(
            f"No train/eval metrics found in {log_file}. "
            "Expected dictionary-like lines with loss/eval_loss fields."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    train_csv = os.path.join(args.output_dir, "train_metrics.csv")
    eval_csv = os.path.join(args.output_dir, "eval_metrics.csv")

    write_csv(
        train_rows,
        train_csv,
        ["line_number", "epoch", "step", "loss", "grad_norm", "learning_rate"],
    )
    write_csv(
        eval_rows,
        eval_csv,
        [
            "line_number",
            "epoch",
            "eval_loss",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        ],
    )

    title_prefix = os.path.basename(log_file)
    plot_paths = make_plots(
        train_rows=train_rows,
        eval_rows=eval_rows,
        output_dir=args.output_dir,
        smoothing_window=args.smoothing_window,
        title_prefix=title_prefix,
    )

    print(f"Log used: {log_file}")
    if trainer_state_used:
        print(f"Trainer state used: {trainer_state_used}")
    print(f"Train rows parsed: {len(train_rows)}")
    print(f"Eval rows parsed: {len(eval_rows)}")
    print(f"Saved: {train_csv}")
    print(f"Saved: {eval_csv}")
    for key, value in plot_paths.items():
        print(f"Saved: {value} ({key})")


if __name__ == "__main__":
    main()
