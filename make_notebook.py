"""Create and execute a compact walkthrough without embedded duplicate charts."""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import nbformat as nbf


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell("# StreamGate-TF: result and state walkthrough\n\nInspect the executed experiment and use the streaming API. This notebook does not train the models. The checkpoints are small byte-language models, not instruction-following assistants."),
        nbf.v4.new_code_cell("import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\nimport json\nfrom pathlib import Path\nmetrics = json.loads(Path('metrics.json').read_text())\nfor kind, result in metrics['models'].items():\n    score = result['holdout']\n    print(f\"{kind:8s} perplexity={score['perplexity']:.4f}, accuracy={score['next_byte_accuracy']:.2%}\")"),
        nbf.v4.new_markdown_cell("## Learning and holdout quality\n\n![Learning curves](figures/learning-curves.png)\n\n![Model comparison](figures/model-comparison.png)\n\nBoth architectures saw identical training-window starts and the same number of byte targets. Validation selected checkpoints and architecture; the holdout did not. This is one seed on one small English corpus."),
        nbf.v4.new_code_cell("from streamgate.inference import StreamingEngine\nengine = StreamingEngine.load(kind='linear')\nsession = engine.session('Holmes said, ')\ninitial_payload = session.attention_state_bytes\ncontinuation = session.generate(count=32, seed=42)\nprint(continuation.decode('utf-8', errors='replace'))\nprint('Before:', initial_payload, 'After:', session.attention_state_bytes, 'bytes')\nassert session.attention_state_bytes == initial_payload == 8704"),
        nbf.v4.new_markdown_cell("## The measured trade-off\n\n![State memory](figures/state-memory.png)\n\n![CPU latency](figures/streaming-latency.png)\n\nFixed attention state is not total-process RAM. StreamGate's sequential prefill made whole requests slower than the cached softmax comparator in this CPU workload. A 256-byte prefix tests state mechanics and scaling, not useful quality beyond the 64-byte training context."),
        nbf.v4.new_code_cell("import numpy as np\nimport tensorflow as tf\ntokens = tf.constant([[72,111,108,109,101,115]], tf.int32)\nfull = engine.model(tokens).numpy()\nfirst, state = engine.model.prefill(tokens[:,:3])\nsecond, state = engine.model.prefill(tokens[:,3:], state=state, offset=3)\nchunked = np.concatenate([first.numpy(), second.numpy()], axis=1)\nprint('Maximum chunk/full logit difference:', np.abs(full-chunked).max())\nnp.testing.assert_allclose(full, chunked, atol=1e-4, rtol=1e-4)"),
        nbf.v4.new_markdown_cell("## Errors and gates\n\n![Error categories](figures/error-by-byte-group.png)\n\n![Gate distributions](figures/learned-gates.png)\n\nReview the highest-loss snippets in `runs/error_analysis.json`. Gate differences describe learned retention patterns; they do not prove semantic specialization. Human review prompts in `LEARNING_NOTES.md` are intentionally unfilled."),
        nbf.v4.new_code_cell("benchmark = json.loads(Path('runs/benchmark.json').read_text())\nverification = json.loads(Path('runs/verification.json').read_text())\nassert verification['status'] == 'passed'\nassert all(item['all_greedy_predictions_match'] for item in benchmark['parity'])\nprint('Fresh-source checkpoint replay:', verification['status'])\nprint('Largest recorded streaming/full logit difference:', max(item['max_absolute_logit_error'] for item in benchmark['parity']))")
    ]
    # This runner requires no kernel sockets and executes the actual cell code.
    # All code cells use print/assert, so no rich-expression display is emulated.
    namespace = {"__name__": "__main__"}
    number = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        number += 1
        stream = StringIO()
        with redirect_stdout(stream):
            exec(compile(cell.source, f"analysis.ipynb:cell{number}", "exec"), namespace)
        cell.execution_count = number
        if stream.getvalue():
            cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=stream.getvalue())]
    nb.metadata.execution = {"runner": "in-process Python cell runner", "reason": "This environment denies Jupyter kernel network sockets. The actual code cells were executed sequentially in Python 3.12; no kernel execution is claimed."}
    nbf.validate(nb)
    nbf.write(nb, "analysis.ipynb")
    print("Executed notebook:", len(nb.cells), "cells")


if __name__ == "__main__":
    main()
