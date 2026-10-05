"""Predeclared equal-token-budget experiment; validation-only selection."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo
import numpy as np
import tensorflow as tf
from streamgate.data import load_corpus, windows, digest
from streamgate.model import ByteDecoder, DecoderConfig

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def evaluate(model, x, y, batch=16):
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32),
                                  tf.TensorSpec([None, None], tf.int32)])
    def score(a, b):
        logits = model(a)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=b, logits=logits)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return losses, predictions
    losses, predictions = [], []
    for start in range(0, len(x), batch):
        loss, pred = score(x[start:start+batch], y[start:start+batch])
        losses.append(loss.numpy()); predictions.append(pred.numpy())
    loss = np.concatenate(losses); pred = np.concatenate(predictions)
    nll = float(loss.astype(np.float64).mean())
    return {"targets": int(y.size), "nll_nats_per_byte": nll,
            "bits_per_byte": nll / np.log(2), "perplexity": float(np.exp(nll)),
            "next_byte_accuracy": float((pred == y).mean())}, loss, pred


def main():
    cfg = json.loads(Path("config.json").read_text())
    for folder in ("runs", "checkpoints", "figures"):
        Path(folder).mkdir(exist_ok=True)
    (train, val, test), manifest = load_corpus()
    context = cfg["context"]
    vx, vy, vs = windows(val, context)
    tx, ty, ts = windows(test, context)
    # Fixed random window plan, shared across architectures; no seed search.
    rng = np.random.default_rng(cfg["seed"])
    plan = rng.integers(0, len(train) - context,
                        size=(cfg["steps"], cfg["batch"]), dtype=np.int32)
    selection = np.linspace(0, len(vx)-1, cfg["selection_windows"], dtype=np.int32)
    np.savez_compressed("runs/window_plan.npz", training_starts=plan,
                        selection_indices=selection, validation_starts=vs, test_starts=ts)
    outcomes, history, error_rows = {}, [], []
    for kind in ("softmax", "linear"):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(cfg["seed"])
        model_cfg = DecoderConfig(kind=kind, **{k: cfg[k] for k in ("width", "heads", "layers", "ff_width")})
        model = ByteDecoder(model_cfg)
        model(tf.zeros([1, context], tf.int32))
        optimizer = tf.keras.optimizers.Adam(cfg["learning_rate"])
        optimizer.build(model.trainable_variables)

        @tf.function(input_signature=[tf.TensorSpec([cfg["batch"], context], tf.int32),
                                      tf.TensorSpec([cfg["batch"], context], tf.int32)])
        def update(x, y):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y, logits=model(x, training=True)))
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, cfg["clip_norm"])
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        best, best_step = float("inf"), 0
        started = time.perf_counter()
        for step, starts in enumerate(plan, 1):
            idx = starts[:, None] + np.arange(context)[None, :]
            loss = float(update(train[idx], train[idx+1]))
            if step % cfg["checkpoint_every"] == 0:
                metric, _, _ = evaluate(model, vx[selection], vy[selection])
                selection_nll = metric["nll_nats_per_byte"]
                history.append({"model": kind, "step": step, "batch_train_nll": loss,
                                "selection_val_nll": selection_nll,
                                "elapsed_seconds": time.perf_counter()-started})
                if selection_nll < best:
                    best, best_step = selection_nll, step
                    model.save_npz(f"checkpoints/{kind}.npz")
                print(json.dumps(history[-1]), flush=True)
        train_seconds = time.perf_counter()-started
        model.load_npz(f"checkpoints/{kind}.npz")
        validation, _, _ = evaluate(model, vx, vy)
        holdout, losses, predictions = evaluate(model, tx, ty)
        np.savez_compressed(f"runs/{kind}_holdout.npz", losses=losses,
                            predictions=predictions, targets=ty, starts=ts)
        for j in np.argsort(losses.mean(axis=1))[-10:][::-1]:
            error_rows.append({"model": kind, "test_byte_offset": int(ts[j]),
                               "mean_nll": float(losses[j].mean()),
                               "accuracy": float((predictions[j] == ty[j]).mean()),
                               "context": bytes(tx[j].astype(np.uint8)).decode("utf-8", errors="replace"),
                               "target": bytes(ty[j].astype(np.uint8)).decode("utf-8", errors="replace")})
        outcomes[kind] = {"config": model_cfg.to_dict(), "parameters": model.count_params(),
                          "selected_step": best_step, "selection_val_nll": best,
                          "validation": validation, "holdout": holdout,
                          "training_seconds_including_selection_and_tracing": train_seconds}
    counts = np.bincount(train, minlength=256).astype(np.float64) + 1.0
    probabilities = counts / counts.sum()
    base_nll = float(-np.log(probabilities[ty]).mean())
    outcomes["unigram"] = {"holdout": {"targets": int(ty.size), "nll_nats_per_byte": base_nll,
        "bits_per_byte": base_nll/np.log(2), "perplexity": float(np.exp(base_nll)),
        "next_byte_accuracy": float((ty == probabilities.argmax()).mean())},
        "policy": "Add-one smoothed byte counts fitted on training only"}
    winner = min(("softmax", "linear"), key=lambda k: outcomes[k]["validation"]["nll_nats_per_byte"])
    results = {"question": "Can a learned-gate positive-kernel decoder replace growing KV storage with fixed-size recurrent state while retaining useful byte prediction quality?",
               "selected_model": winner, "models": outcomes, "config": cfg,
               "training_targets_per_model": int(plan.size * context),
               "selection_rule": "Checkpoint: minimum NLL on 128 fixed validation windows; architecture: minimum full-validation NLL. Holdout unused for either decision.",
               "split_bytes": {"train": len(train), "validation": len(val), "test": len(test)},
               "single_seed": True, "cross_validation": False}
    Path("metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    with Path("runs/training.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    Path("runs/error_analysis.json").write_text(json.dumps(error_rows, indent=2) + "\n")
    audit = {"completed_at": datetime.now(ZoneInfo("Asia/Riyadh")).isoformat(),
             "python": platform.python_version(), "tensorflow": tf.__version__,
             "numpy": np.__version__, "keras": tf.keras.__version__,
             "platform": platform.platform(), "processor": platform.processor(),
             "devices": [str(d) for d in tf.config.list_physical_devices()],
             "threads": {"intra": 2, "inter": 2}, "deterministic_ops": True,
             "automated_attribution": "Implemented and executed with OpenAI Codex assistance for Saud Alotaibi; human learning review is not filled.",
             "dataset_sha256": manifest["body_sha256"],
             "files_sha256": {str(p): digest(p.read_bytes()) for p in
                               [Path("config.json"), Path("train.py"), Path("streamgate/model.py"),
                                Path("runs/window_plan.npz"), *Path("checkpoints").glob("*.npz")]}}
    Path("audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
