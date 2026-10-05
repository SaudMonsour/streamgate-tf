"""Explicit streaming session and checkpoint loading, usable without training."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .model import ByteDecoder, DecoderConfig, state_bytes


class StreamingEngine:
    def __init__(self, model):
        self.model = model
        c = model.config
        if c.kind == "linear":
            pair = (tf.TensorSpec([None, c.heads, c.width//c.heads, c.width//c.heads], tf.float32),
                    tf.TensorSpec([None, c.heads, c.width//c.heads], tf.float32))
        else:
            pair = (tf.TensorSpec([None, c.heads, None, c.width//c.heads], tf.float32),) * 2
        self.prefill_graph = tf.function(model.prefill,
            input_signature=[tf.TensorSpec([None, None], tf.int32)])
        self.step_graph = tf.function(model.step, input_signature=[
            tf.TensorSpec([None, 1], tf.int32), tuple(pair for _ in range(c.layers)),
            tf.TensorSpec([], tf.int32)])

    @classmethod
    def load(cls, root=Path("."), kind="linear"):
        root = Path(root)
        cfg = json.loads((root / "metrics.json").read_text())["models"][kind]["config"]
        model = ByteDecoder(DecoderConfig(**cfg))
        model.load_npz(root / "checkpoints" / f"{kind}.npz")
        return cls(model)

    def session(self, prompt):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty UTF-8 string")
        tokens = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).astype(np.int32)[None]
        return ByteSession(self, tokens)


class ByteSession:
    def __init__(self, engine, tokens):
        self.engine = engine
        logits, self.state = engine.prefill_graph(tf.constant(tokens))
        self.next_logits = logits[:, -1]
        self.position = tokens.shape[1]

    @property
    def attention_state_bytes(self):
        return state_bytes(self.state)

    def consume(self, token):
        if not isinstance(token, (int, np.integer)) or not 0 <= token <= 255:
            raise ValueError("one byte ID in [0,255] is required")
        logits, self.state = self.engine.step_graph(
            tf.constant([[token]], tf.int32), self.state, tf.constant(self.position))
        self.position += 1
        self.next_logits = logits[:, -1]
        return self.next_logits

    def generate(self, count=128, temperature=0.8, seed=42):
        if count < 0 or temperature <= 0:
            raise ValueError("count must be nonnegative and temperature positive")
        rng = np.random.default_rng(seed)
        output = []
        for _ in range(count):
            scores = self.next_logits.numpy()[0].astype(np.float64) / temperature
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            token = int(rng.choice(256, p=probabilities))
            output.append(token)
            self.consume(token)
        return bytes(output)
