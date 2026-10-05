# Experiment execution record

The published run uses `config.json` without a seed or hyperparameter search. Both models receive the exact starts in `runs/window_plan.npz`. The corpus is downloaded from Gutenberg #1661 and fails closed if its source/body hashes differ from `data/manifest.json`. Internal repetitions in the book are preserved; no input/target window crosses a contiguous split boundary.

The recorded environment is Python 3.12, TensorFlow CPU 2.20.0, and Keras 3.15.1; see `audit.json` for the exact NumPy version, device, threads, timestamps, and hashes. CPU kernels and host load can change timing and small numerical differences.

After installing `requirements.txt`, these commands execute the components from the repository root:

```bash
python -m unittest discover -s tests -v
python train.py
python verify.py
python benchmark.py
python report.py
```

`train.py` overwrites checkpoints, metrics, audit, and training/evaluation artifacts. Run it in a separate checkout if you want to preserve the published run. `verify.py` uses existing checkpoints and a fresh download; it does not retrain. Benchmarking should run on an otherwise idle host, without verification or training running concurrently. `report.py` reads executed artifacts and derives plots and diagnostic tables.

Full validation chooses the architecture only after checkpoint selection on 128 fixed validation windows. Holdout loss is the float64 mean of the recorded float32 per-target cross-entropies. Accuracy compares the argmax byte with the true next byte. All 899 holdout windows are represented in `error_analysis.csv`; raw target-level arrays are in `runs/*_holdout.npz`. Trailing targets that cannot fill a 64-target window are excluded.

The source has 575,793 processed bytes. Byte offsets, including split boundaries, can occur inside multi-byte Unicode characters; the task models bytes and does not require every split/window to independently decode as UTF-8. Human-readable snippets replace invalid edge fragments. No table-row count, word perplexity, fold variance, permutation importance, or external benchmark score is claimed.

The initial exploratory CPU timing run overlapped with checkpoint verification. It was discarded; the published benchmark was rerun after that process finished. Neither model was retrained or selected based on latency. Timing trials are workload repetitions, not independent model training seeds.

The notebook's code cells were executed sequentially by `make_notebook.py` in the same Python environment. This host denies the network sockets required by a Jupyter kernel, so the in-process runner captures real print output and assertions without claiming kernel execution. The notebook contains this runner note in its metadata and can also be opened in a normal Jupyter environment.

## Sources

* [Project Gutenberg catalog #1661](https://www.gutenberg.org/ebooks/1661): title, author, source text, and public-domain notice in the USA.
* [Katharopoulos et al., ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html): positive-kernel attention and its recurrent form. This implementation adds a simple input-dependent sigmoid decay and is not a faithful reproduction of that paper's experiment.
* [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/): numerical operations, trainable layers, autodifferentiation, and optimizer primitives.

## Limitations

One small English book, one seed, a short context, and a limited training budget. Both checkpoints remain undertrained language models. The sinusoidal positions can be evaluated outside the training range, but that does not establish quality there. Different initialization paths and 520 additional gate parameters also affect this architectural comparison. A stronger conclusion needs multiple seeds, datasets, longer training, a gate ablation, and separate GPU kernel measurements.

Inference state is constant in prefix length for fixed width and heads, but scales quadratically with each head's key dimension. During training the scan stores sequence intermediates. Sequential prefill is slower than parallel softmax in the measured CPU workload. No fused kernel, parallel scan, production deployment, or claim of novel algorithmic invention is included.
