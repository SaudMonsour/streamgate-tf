"""Measured CPU workloads and trained-checkpoint parity; no predicted speedups."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from streamgate.data import load_corpus
from streamgate.inference import StreamingEngine
from streamgate.model import state_bytes

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


def main():
    cfg = json.loads(Path("config.json").read_text())
    (_, _, test), _ = load_corpus()
    rows, parity, samples = [], [], []
    for kind in ("softmax", "linear"):
        engine = StreamingEngine.load(kind=kind)
        model = engine.model
        reference = tf.function(model.call,
            input_signature=[tf.TensorSpec([None, None], tf.int32)])
        for prefix in cfg["benchmark_prefixes"]:
            updates = cfg["benchmark_updates"]
            ids = test[:prefix+updates][None]
            tensor = tf.constant(ids)
            # Verify every new-token logit against the full-sequence reference.
            expected = reference(tensor).numpy()
            first, state = engine.prefill_graph(tensor[:, :prefix])
            outputs = [first.numpy()]
            for offset in range(prefix, prefix+updates):
                out, state = engine.step_graph(tensor[:, offset:offset+1], state, tf.constant(offset))
                outputs.append(out.numpy())
            actual = np.concatenate(outputs, axis=1)
            error = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            matches = bool(np.array_equal(actual.argmax(-1), expected.argmax(-1)))
            parity.append({"model": kind, "prefix": prefix, "updates": updates,
                           "max_absolute_logit_error": error, "all_greedy_predictions_match": matches})

            def streamed():
                start = time.perf_counter()
                logits, s = engine.prefill_graph(tensor[:, :prefix])
                logits.numpy()
                pref_seconds = time.perf_counter()-start
                for offset in range(prefix, prefix+updates):
                    logits, s = engine.step_graph(tensor[:, offset:offset+1], s, tf.constant(offset))
                    logits.numpy()
                total = time.perf_counter()-start
                return total * 1000, pref_seconds * 1000, (total-pref_seconds)*1000/updates, s

            def repeated_full():
                start = time.perf_counter()
                for end in range(prefix, prefix+updates+1):
                    reference(tensor[:, :end]).numpy()
                return (time.perf_counter()-start)*1000

            for _ in range(cfg["benchmark_warmup"]):
                streamed(); repeated_full()
            requests, prefills, decode, fulls = [], [], [], []
            # Alternate order to limit order/warm-cache bias.
            for trial in range(cfg["benchmark_trials"]):
                if trial % 2 == 0:
                    total, prefill, per_byte, s = streamed(); full = repeated_full()
                else:
                    full = repeated_full(); total, prefill, per_byte, s = streamed()
                requests.append(total);prefills.append(prefill);decode.append(per_byte);fulls.append(full)
            _, prefix_state = engine.prefill_graph(tensor[:, :prefix])
            actual_bytes = state_bytes(prefix_state)
            c = model.config
            formula = (c.layers * c.heads * ((c.width//c.heads)**2+c.width//c.heads) * 4
                       if kind == "linear" else 2 * c.layers * prefix * c.width * 4)
            assert actual_bytes == formula
            rows.append({"model": kind, "prefix": prefix, "updates": updates,
                "prefix_state_bytes": actual_bytes, "final_state_bytes": state_bytes(s),
                "stream_request_median_ms": float(np.median(requests)),
                "stream_request_p95_ms": float(np.percentile(requests, 95)),
                "prefill_median_ms": float(np.median(prefills)),
                "decode_ms_per_byte_median": float(np.median(decode)),
                "repeated_full_request_median_ms": float(np.median(fulls)),
                "raw_stream_request_ms": requests, "raw_decode_ms_per_byte": decode,
                "raw_prefill_ms": prefills, "raw_repeated_full_request_ms": fulls})
            print(json.dumps({k:v for k,v in rows[-1].items() if not k.startswith("raw_")}), flush=True)
        samples.append({"model": kind, "prompt": "Holmes said, ", "seed": 42,
                        "temperature": 0.8, "generated_bytes": 128,
                        "text": engine.session("Holmes said, ").generate(128).decode("utf-8", errors="replace")})
        assert engine.step_graph.experimental_get_tracing_count() == 1
        assert engine.prefill_graph.experimental_get_tracing_count() == 1
    result = {"rows": rows, "parity": parity, "trials": cfg["benchmark_trials"],
        "warmup": cfg["benchmark_warmup"], "batch": 1, "device": "CPU",
        "threads": {"intra": 2, "inter": 2},
        "timing_policy": "perf_counter; materialize each logit tensor; include prefill and Python state orchestration; exclude loading, tracing, tokenization, sampling. Repeated-full reference emits all prefix logits. Streaming emits full prefill then one logit per update.",
        "state_policy": "Float32 attention tensors only; excludes weights, position scalar, transient activations, TensorFlow allocator and graph overhead.",
        "context_warning": "Trained on 64-byte windows. Prefix 256 is a systems scaling workload, not evidence of quality beyond training context. No sliding-window eviction is used; both modes process the entire prefix."}
    Path("runs/benchmark.json").write_text(json.dumps(result, indent=2)+"\n")
    Path("runs/samples.json").write_text(json.dumps(samples, indent=2)+"\n")


if __name__ == "__main__":
    main()
