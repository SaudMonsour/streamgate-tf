"""Attention equations and explicit, request-owned recurrent state.

The linear layer uses a positive ELU kernel with a learned per-head forgetting
gate. It is a pedagogical variant, not a reproduction of a named gated-linear
attention paper. No pretrained weights or TensorFlow attention wrappers.
"""
from dataclasses import asdict, dataclass
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class DecoderConfig:
    kind: str = "linear"
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff_width: int = 128
    vocab: int = 256

    def __post_init__(self):
        if self.kind not in ("linear", "softmax"):
            raise ValueError("kind must be linear or softmax")
        if min(self.width, self.heads, self.layers, self.ff_width, self.vocab) < 1:
            raise ValueError("dimensions must be positive")
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")

    def to_dict(self):
        return asdict(self)


class GatedLinearAttention(tf.keras.layers.Layer):
    def __init__(self, width, heads, **kwargs):
        super().__init__(**kwargs)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.dim = width, heads, width // heads
        self.qkv = tf.keras.layers.Dense(3 * width, use_bias=False)
        self.gate = tf.keras.layers.Dense(
            heads, kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(3.0))
        self.output_projection = tf.keras.layers.Dense(width, use_bias=False)

    def initial_state(self, batch):
        return (tf.zeros([batch, self.heads, self.dim, self.dim]),
                tf.zeros([batch, self.heads, self.dim]))

    def project(self, x):
        shape = tf.shape(x)
        q, k, v = tf.unstack(tf.reshape(
            self.qkv(x), [shape[0], shape[1], 3, self.heads, self.dim]), axis=2)
        # Scale to keep outer products modest; normalization cancels common scale.
        scale = tf.cast(self.dim, x.dtype) ** -0.5
        return ((tf.nn.elu(q) + 1.0) * scale,
                (tf.nn.elu(k) + 1.0) * scale, v,
                tf.sigmoid(self.gate(x)))

    def call(self, x, state=None, return_state=False):
        q, k, v, g = self.project(x)
        if state is None:
            state = self.initial_state(tf.shape(x)[0])

        def update(previous, current):
            s, z = previous
            kt, vt, gt = current
            return (gt[..., None, None] * s + tf.einsum("bhd,bhe->bhde", kt, vt),
                    gt[..., None] * z + kt)

        states, normalizers = tf.scan(
            update, (tf.transpose(k, [1, 0, 2, 3]),
                     tf.transpose(v, [1, 0, 2, 3]),
                     tf.transpose(g, [1, 0, 2])), initializer=state)
        qt = tf.transpose(q, [1, 0, 2, 3])
        numerator = tf.einsum("tbhd,tbhde->tbhe", qt, states)
        denominator = tf.reduce_sum(qt * normalizers, axis=-1, keepdims=True)
        y = numerator / tf.maximum(denominator, 1e-6)
        y = tf.reshape(tf.transpose(y, [1, 0, 2, 3]),
                       [tf.shape(x)[0], tf.shape(x)[1], self.width])
        y = self.output_projection(y)
        return (y, (states[-1], normalizers[-1])) if return_state else y

    def step(self, x, state):
        """One token, with state provided and returned rather than hidden mutation."""
        tf.debugging.assert_equal(tf.shape(x)[1], 1, message="step requires exactly one position")
        q, k, v, g = self.project(x)
        q, k, v, g = q[:, 0], k[:, 0], v[:, 0], g[:, 0]
        s, z = state
        s = g[..., None, None] * s + tf.einsum("bhd,bhe->bhde", k, v)
        z = g[..., None] * z + k
        y = tf.einsum("bhd,bhde->bhe", q, s)
        y /= tf.maximum(tf.reduce_sum(q * z, axis=-1, keepdims=True), 1e-6)
        return self.output_projection(tf.reshape(y, [tf.shape(x)[0], 1, self.width])), (s, z)


class SoftmaxAttention(tf.keras.layers.Layer):
    def __init__(self, width, heads, **kwargs):
        super().__init__(**kwargs)
        self.width, self.heads, self.dim = width, heads, width // heads
        self.qkv = tf.keras.layers.Dense(3 * width, use_bias=False)
        self.output_projection = tf.keras.layers.Dense(width, use_bias=False)

    def initial_state(self, batch):
        return (tf.zeros([batch, self.heads, 0, self.dim]),
                tf.zeros([batch, self.heads, 0, self.dim]))

    def project(self, x):
        q, k, v = tf.unstack(tf.reshape(self.qkv(x),
            [tf.shape(x)[0], tf.shape(x)[1], 3, self.heads, self.dim]), axis=2)
        return tuple(tf.transpose(t, [0, 2, 1, 3]) for t in (q, k, v))

    def attend(self, q, k, v, mask=None):
        scores = tf.matmul(q, k, transpose_b=True) * self.dim ** -0.5
        if mask is not None:
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        y = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        y = tf.reshape(tf.transpose(y, [0, 2, 1, 3]),
                       [tf.shape(q)[0], tf.shape(q)[2], self.width])
        return self.output_projection(y)

    def call(self, x, state=None, return_state=False):
        q, k, v = self.project(x)
        previous = 0 if state is None else tf.shape(state[0])[2]
        if state is not None:
            k, v = tf.concat([state[0], k], axis=2), tf.concat([state[1], v], axis=2)
        mask = (tf.range(tf.shape(k)[2])[None, :] <=
                previous + tf.range(tf.shape(q)[2])[:, None])
        y = self.attend(q, k, v, mask)
        return (y, (k, v)) if return_state else y

    def step(self, x, state):
        tf.debugging.assert_equal(tf.shape(x)[1], 1, message="step requires exactly one position")
        q, k, v = self.project(x)
        k, v = tf.concat([state[0], k], axis=2), tf.concat([state[1], v], axis=2)
        return self.attend(q, k, v), (k, v)


class Block(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        attention = GatedLinearAttention if config.kind == "linear" else SoftmaxAttention
        self.attention = attention(config.width, config.heads)
        self.ff1 = tf.keras.layers.Dense(config.ff_width, activation=tf.nn.gelu)
        self.ff2 = tf.keras.layers.Dense(config.width)

    def call(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff2(self.ff1(self.norm2(x)))

    def prefill(self, x, state):
        y, state = self.attention(self.norm1(x), state=state, return_state=True)
        x = x + y
        return x + self.ff2(self.ff1(self.norm2(x))), state

    def step(self, x, state):
        y, state = self.attention.step(self.norm1(x), state)
        x = x + y
        return x + self.ff2(self.ff1(self.norm2(x))), state


class ByteDecoder(tf.keras.Model):
    def __init__(self, config=DecoderConfig(), **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocab, config.width)
        self.blocks = [Block(config, name=f"block_{i}") for i in range(config.layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def embed(self, tokens, offset=0):
        x = self.embedding(tokens)
        pos = tf.cast(tf.range(tf.shape(tokens)[1]) + offset, tf.float32)
        rates = tf.exp(-tf.range(0, self.config.width, 2, dtype=tf.float32)
                       * (np.log(10000.0) / self.config.width))
        angles = pos[:, None] * rates[None, :]
        position = tf.reshape(tf.stack([tf.sin(angles), tf.cos(angles)], axis=-1),
                              [tf.shape(tokens)[1], self.config.width])
        return x + position[None, :, :] * 0.1

    def logits(self, x):
        return tf.matmul(self.norm(x), self.embedding.embeddings, transpose_b=True)

    def call(self, tokens, training=False):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.logits(x)

    def initial_state(self, batch=1):
        return tuple(block.attention.initial_state(batch) for block in self.blocks)

    def prefill(self, tokens, state=None, offset=0):
        if state is None:
            state = self.initial_state(tf.shape(tokens)[0])
        x = self.embed(tokens, offset)
        new_state = []
        for block, previous in zip(self.blocks, state, strict=True):
            x, previous = block.prefill(x, previous)
            new_state.append(previous)
        return self.logits(x), tuple(new_state)

    def step(self, tokens, state, offset):
        tf.debugging.assert_equal(tf.shape(tokens)[1], 1, message="step requires exactly one byte")
        x = self.embed(tokens, offset)
        new_state = []
        for block, previous in zip(self.blocks, state, strict=True):
            x, previous = block.step(x, previous)
            new_state.append(previous)
        return self.logits(x), tuple(new_state)

    def save_npz(self, path):
        np.savez_compressed(path, **{f"w{i}": w.numpy() for i, w in enumerate(self.weights)})

    def load_npz(self, path):
        self(tf.zeros([1, 2], tf.int32))
        with np.load(path, allow_pickle=False) as checkpoint:
            if len(checkpoint.files) != len(self.weights):
                raise ValueError("checkpoint/config mismatch")
            for i, weight in enumerate(self.weights):
                value = checkpoint[f"w{i}"]
                if value.shape != tuple(weight.shape):
                    raise ValueError(f"shape mismatch at weight {i}")
                weight.assign(value)


def state_bytes(state):
    """Tensor payload only; excludes weights, positions, activations and allocator."""
    return sum(int(tf.size(t)) * t.dtype.size for pair in state for t in pair)
