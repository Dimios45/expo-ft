"""Small layout tests; real checkpoint parity is a separate integration run."""

import unittest

import numpy as np
import torch

from expo_ft.conversion.yam_pi05 import Mapper


class Source:
    def __init__(self, x):
        self.x = x

    def get_tensor(self, key):
        return torch.from_numpy(self.x)


class LayoutTests(unittest.TestCase):
    def test_bijection_and_duplicate_consumption(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        m = Mapper(Source(x))
        y = m.get(
            "test", lambda a: a.transpose(2, 0, 1), lambda a: a.transpose(1, 2, 0)
        )
        np.testing.assert_array_equal(y, x.transpose(2, 0, 1))
        with self.assertRaisesRegex(ValueError, "twice"):
            m.get("test")

    def test_noninvertible_mapping_rejected(self):
        m = Mapper(Source(np.array([1.0, 2.0], dtype=np.float32)))
        with self.assertRaisesRegex(AssertionError, "Lossy"):
            m.get("test", lambda a: a * 0)

    def test_attention_output_orientation(self):
        rng = np.random.default_rng(2)
        weight = rng.normal(size=(5, 12)).astype(np.float32)
        kernel = weight.T.reshape(3, 4, 5)
        x = rng.normal(size=(2, 3, 4)).astype(np.float32)
        np.testing.assert_allclose(
            np.einsum("bnh,nhd->bd", x, kernel), x.reshape(2, 12) @ weight.T, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
