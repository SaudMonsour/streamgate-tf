# Using StreamGate-TF

Python 3.12 and the direct dependencies in `requirements.txt` are the recorded environment. From the repository root, install them in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Generate with the included checkpoint

```bash
python generate.py --model linear --prompt "Holmes said, " --bytes 128
```

`--model softmax` loads the comparator. `--temperature` defaults to 0.8; `--seed` defaults to 42. The checkpoint generates byte sequences, not guaranteed valid UTF-8. The CLI replaces invalid sequences only when displaying them; the API returns the original bytes. The model was trained on 64-byte windows, so continuing a long session is mechanically supported but does not guarantee useful long-context predictions.

## Explicit request-owned state

```python
from streamgate.inference import StreamingEngine

engine = StreamingEngine.load(kind="linear")
request_a = engine.session("Holmes said, ")
request_b = engine.session("Watson asked, ")
continuation = request_a.generate(count=64, seed=42)
print(continuation.decode("utf-8", errors="replace"))
print(request_a.attention_state_bytes)  # 8704 for the included linear checkpoint
```

Creating or advancing `request_b` does not change `request_a`. Model weights are shared; attention state and position are separate. `ByteSession` supports batch one. Lower-level model and engine graphs support equal-length batched tensors. There is no server, ragged request batching, GPU kernel, or production scheduler.

## Use the layer in another model

```python
import tensorflow as tf
from streamgate.model import GatedLinearAttention

attention = GatedLinearAttention(width=64, heads=4)
hidden = tf.random.normal([2, 12, 64])
output, state = attention(hidden, return_state=True)
next_hidden = tf.random.normal([2, 1, 64])
next_output, state = attention.step(next_hidden, state)
```

The output has the same batch, sequence, and width dimensions as the input. State is an `(S, z)` tuple shaped `[batch, heads, head_dim, head_dim]` and `[batch, heads, head_dim]`. To reset a request, call `initial_state(batch)` or prefill with `state=None`. Pass state explicitly between calls. `step` accepts exactly one position; chunked calls use `attention(chunk, state=state, return_state=True)`.

The layer is causal and uses the current key/value in each output. Padding masks and variable-length batches are not implemented; do not interpret padding tokens as absent. Inputs and state use float32 in this study. Avoid treating this minimal API as a drop-in replacement for pretrained Transformer attention: weights require training, and attention semantics differ.

## Decoder-level chunking

```python
import tensorflow as tf
from streamgate.inference import StreamingEngine

model = StreamingEngine.load(kind="linear").model
first = tf.constant([[72, 111, 108]], dtype=tf.int32)
second = tf.constant([[109, 101, 115]], dtype=tf.int32)
logits_a, state = model.prefill(first)
logits_b, state = model.prefill(second, state=state, offset=3)
```

The offset is the number of previously consumed bytes. The session API maintains it automatically. Reusing the wrong offset changes sinusoidal positions and breaks equivalence. Checkpoints must be loaded with their corresponding configuration from `metrics.json`; incompatible shapes are rejected.
