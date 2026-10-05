"""Independent equations, causality, chunking and request isolation."""
import unittest
import numpy as np
import tensorflow as tf
from streamgate.model import ByteDecoder, DecoderConfig, GatedLinearAttention, state_bytes
from streamgate.data import windows


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)

    def test_kernel_matches_explicit_weighted_attention(self):
        layer = GatedLinearAttention(8, 2)
        x = tf.random.normal([2, 7, 8])
        actual = layer(x).numpy()
        # Make gates input-dependent: the zero-initialized gate is not enough.
        layer.gate.kernel.assign(tf.random.normal(layer.gate.kernel.shape) * 0.2)
        actual = layer(x).numpy()
        q, k, v, g = [t.numpy() for t in layer.project(x)]
        expected = np.zeros([2, 7, 2, 4], np.float32)
        for b in range(2):
            for t in range(7):
                for h in range(2):
                    weights = np.array([
                        np.dot(q[b, t, h], k[b, j, h]) * np.prod(g[b, j+1:t+1, h])
                        for j in range(t + 1)])
                    expected[b, t, h] = weights @ v[b, :t+1, h] / max(weights.sum(), 1e-6)
        expected = layer.output_projection(expected.reshape(2, 7, 8)).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_full_vs_steps_and_chunks(self):
        for kind in ("linear", "softmax"):
            model = ByteDecoder(DecoderConfig(kind=kind, width=16, heads=2, ff_width=32))
            tokens = tf.constant([[72, 111, 108, 109, 101, 115, 46]], tf.int32)
            reference = model(tokens).numpy()
            state = model.initial_state(1)
            outputs = []
            for i in range(tokens.shape[1]):
                logits, state = model.step(tokens[:, i:i+1], state, i)
                outputs.append(logits.numpy())
            np.testing.assert_allclose(np.concatenate(outputs, axis=1), reference, atol=3e-5, rtol=3e-5)
            first, state = model.prefill(tokens[:, :3])
            second, state = model.prefill(tokens[:, 3:], state, offset=3)
            np.testing.assert_allclose(np.concatenate([first, second], axis=1), reference, atol=3e-5, rtol=3e-5)

    def test_causality(self):
        for kind in ("linear", "softmax"):
            model = ByteDecoder(DecoderConfig(kind=kind, width=16, heads=2, ff_width=32))
            a = model(tf.constant([[1, 2, 3, 4, 5]])).numpy()
            b = model(tf.constant([[1, 2, 3, 201, 200]])).numpy()
            np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6)

    def test_state_isolation_and_payload(self):
        for kind in ("linear", "softmax"):
            model = ByteDecoder(DecoderConfig(kind=kind, width=16, heads=2, ff_width=32))
            model(tf.constant([[1, 2]]))
            shared = model.initial_state(1)
            saved = [t.numpy().copy() for pair in shared for t in pair]
            a, state_a = model.prefill(tf.constant([[1, 2, 3]]), shared)
            model.prefill(tf.constant([[90, 91]]), shared)
            b, _ = model.prefill(tf.constant([[1, 2, 3]]), shared)
            np.testing.assert_allclose(a, b, atol=1e-6)
            for expected, tensor in zip(saved, [t for pair in shared for t in pair]):
                np.testing.assert_array_equal(expected, tensor.numpy())
            if kind == "linear":
                self.assertEqual(state_bytes(state_a), 2 * 2 * (8 * 8 + 8) * 4)
                _, longer = model.prefill(tf.constant([[1] * 30]))
                self.assertEqual(state_bytes(state_a), state_bytes(longer))
            else:
                self.assertEqual(state_bytes(state_a), 2 * 2 * 3 * 16 * 4)

    def test_windows_have_disjoint_targets(self):
        data = np.arange(130, dtype=np.int32)
        x, y, starts = windows(data, 64)
        self.assertEqual(x.shape, (2, 64))
        self.assertEqual(len(np.unique(y)), y.size)
        self.assertLess(y.max(), len(data))
        np.testing.assert_array_equal(x[:, 0], starts)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            DecoderConfig(width=15, heads=4)
        with self.assertRaises(ValueError):
            DecoderConfig(kind="unknown")

    def test_step_rejects_multi_token_input(self):
        for kind in ("linear", "softmax"):
            model = ByteDecoder(DecoderConfig(kind=kind, width=16, heads=2, ff_width=32))
            model(tf.constant([[1, 2]]))
            with self.assertRaises(tf.errors.InvalidArgumentError):
                model.step(tf.constant([[1, 2]]), model.initial_state(1), 0)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                model.blocks[0].attention.step(tf.zeros([1, 2, 16]),
                                               model.blocks[0].attention.initial_state(1))


if __name__ == "__main__":
    unittest.main()
