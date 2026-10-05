"""Charts and diagnostics derived exclusively from executed experiment files."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from streamgate.data import load_corpus, windows
from streamgate.inference import StreamingEngine

COLORS = {"linear": "#0f766e", "softmax": "#4338ca", "unigram": "#94a3b8"}
LABELS = {"linear": "StreamGate", "softmax": "Softmax attention", "unigram": "Unigram"}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150,
                    "font.size": 10, "axes.spines.top": False,
                    "axes.spines.right": False, "axes.titleweight": "bold"})


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(Path("figures") / name, facecolor="white")
    plt.close(fig)


def main():
    metrics = json.loads(Path("metrics.json").read_text())
    benchmark = json.loads(Path("runs/benchmark.json").read_text())
    with Path("runs/training.csv").open() as f:
        learning = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in ("softmax", "linear"):
        rows = [r for r in learning if r["model"] == kind]
        ax.plot([int(r["step"]) for r in rows],
                [float(r["selection_val_nll"]) for r in rows], marker="o",
                color=COLORS[kind], label=LABELS[kind])
    ax.set(title="Validation learning curves", xlabel="Optimizer steps",
           ylabel="NLL · nats per byte (lower is better)")
    ax.legend(); ax.grid(alpha=.15)
    finish(fig, "learning-curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    order = ["unigram", "softmax", "linear"]
    for ax, key, title in zip(axes, ["perplexity", "next_byte_accuracy"],
                              ["Holdout byte perplexity ↓", "Next-byte accuracy ↑"]):
        values = [metrics["models"][k]["holdout"][key] for k in order]
        if key.endswith("accuracy"):
            values = [v*100 for v in values]
        bars = ax.bar([LABELS[k] for k in order], values, color=[COLORS[k] for k in order])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(title);ax.set_ylim(0, max(values)*1.17)
        ax.tick_params(axis="x", rotation=12)
        ax.set_ylabel("Perplexity" if key == "perplexity" else "Accuracy (%)")
    fig.suptitle("57,536 held-out byte targets · fixed 64-byte windows", fontsize=11)
    finish(fig, "model-comparison.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in ("softmax", "linear"):
        rows = [r for r in benchmark["rows"] if r["model"] == kind]
        ax.plot([r["prefix"] for r in rows], [r["prefix_state_bytes"]/1024 for r in rows],
                marker="o", color=COLORS[kind], label=LABELS[kind])
    ax.set(title="Measured attention-state tensor payload", xlabel="Prefix bytes",
           ylabel="KiB · batch 1, float32, two blocks")
    ax.annotate("StreamGate: 8.5 KiB at every length", (128, 8.5),
                xytext=(65, 50), arrowprops={"arrowstyle": "->", "color": COLORS["linear"]})
    ax.legend();ax.grid(alpha=.15)
    finish(fig, "state-memory.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for kind in ("softmax", "linear"):
        rows = [r for r in benchmark["rows"] if r["model"] == kind]
        for ax, key in zip(axes, ["stream_request_median_ms", "decode_ms_per_byte_median"]):
            ax.plot([r["prefix"] for r in rows], [r[key] for r in rows], marker="o",
                    color=COLORS[kind], label=LABELS[kind])
    axes[0].set(title="Whole request: prefill + 32 updates", ylabel="Median milliseconds")
    axes[1].set(title="Decode only, amortized", ylabel="Median milliseconds / byte")
    for ax in axes:
        ax.set_xlabel("Prefix bytes");ax.grid(alpha=.15);ax.legend()
    fig.suptitle("Observed CPU timings · 20 trials · no speedup extrapolation", fontsize=11)
    finish(fig, "streaming-latency.png")

    (train, _, test), manifest = load_corpus()
    def group(byte):
        if byte in (9, 10, 13, 32): return "Whitespace"
        if 97 <= byte <= 122: return "Lowercase"
        if 65 <= byte <= 90: return "Uppercase"
        if 48 <= byte <= 57: return "Digits"
        if byte >= 128: return "UTF-8 high bytes"
        return "Punctuation / other"
    groups = np.array([group(i) for i in range(256)])
    group_rows, window_rows = [], []
    for kind in ("softmax", "linear"):
        with np.load(f"runs/{kind}_holdout.npz", allow_pickle=False) as a:
            losses, predicted, targets, starts = [a[k] for k in ("losses", "predictions", "targets", "starts")]
        for name in dict.fromkeys(groups):
            mask = groups[targets] == name
            if not mask.any(): continue
            group_rows.append({"model": kind, "group": name, "targets": int(mask.sum()),
                               "mean_nll": float(losses[mask].astype(np.float64).mean()),
                               "accuracy": float((predicted[mask] == targets[mask]).mean())})
        for i, start in enumerate(starts):
            window_rows.append({"model": kind, "test_byte_offset": int(start),
                                "mean_nll": float(losses[i].astype(np.float64).mean()),
                                "accuracy": float((predicted[i] == targets[i]).mean()),
                                "wrong_bytes": int((predicted[i] != targets[i]).sum())})
    for path, rows in (("runs/error_by_byte_group.csv", group_rows), ("error_analysis.csv", window_rows)):
        with Path(path).open("w", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(dict.fromkeys(r["group"] for r in group_rows))
    for shift, kind in zip([-.18, .18], ["softmax", "linear"]):
        rows = {r["group"]:r for r in group_rows if r["model"] == kind}
        ax.bar(np.arange(len(names))+shift, [rows[n]["mean_nll"] for n in names], width=.36,
               color=COLORS[kind], label=LABELS[kind])
    ax.set_xticks(np.arange(len(names)), names, rotation=15)
    ax.set(title="Where prediction is difficult", ylabel="Holdout NLL · nats per byte")
    ax.legend();finish(fig, "error-by-byte-group.png")

    x, _, _ = windows(test, metrics["config"]["context"])
    engine = StreamingEngine.load(kind="linear")
    model = engine.model
    x = tf.constant(x[:128])
    hidden = model.embed(x)
    gate_rows = []
    fig, axes = plt.subplots(1, model.config.layers, figsize=(8, 4))
    for i, (ax, block) in enumerate(zip(axes, model.blocks)):
        gates = tf.sigmoid(block.attention.gate(block.norm1(hidden))).numpy()
        for h in range(model.config.heads):
            values = gates[:, :, h].ravel()
            gate_rows.append({"block": i+1, "head": h, "observations": len(values),
                              "mean": float(values.mean()), "std": float(values.std()),
                              "min": float(values.min()), "max": float(values.max()),
                              "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))})
        ax.boxplot([gates[:,:,h].ravel() for h in range(model.config.heads)], tick_labels=range(model.config.heads), showfliers=False)
        ax.set(title=f"Block {i+1} forgetting gates", xlabel="Head", ylabel="Gate value");ax.set_ylim(0, 1)
        hidden = block(hidden)
    finish(fig, "learned-gates.png")
    Path("runs/gate_statistics.json").write_text(json.dumps(gate_rows, indent=2)+"\n")
    # Report the direction of window-level differences without implying independence.
    with np.load("runs/linear_holdout.npz", allow_pickle=False) as a:
        linear_losses = a["losses"].astype(np.float64).mean(axis=1)
    with np.load("runs/softmax_holdout.npz", allow_pickle=False) as a:
        soft_losses = a["losses"].astype(np.float64).mean(axis=1)
    diagnostics = {"holdout_windows": len(linear_losses),
        "linear_better_nll_windows": int((linear_losses < soft_losses).sum()),
        "warning": "Windows from one book are correlated; no significance or generalization claim.",
        "byte_group_counts_in_train": {g:int((groups[train] == g).sum()) for g in dict.fromkeys(groups)},
        "data_body_bytes": manifest["body_bytes"]}
    Path("runs/diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
