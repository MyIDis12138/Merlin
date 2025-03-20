import unittest

import torch

from minidl.model.model_registry import ModelBuilder


class TestMRIBaseline(unittest.TestCase):

    def test_base_model(self):
        config = {
            "model": {
                "name": "mri_baseline",
                "n_classes": 4,
                "d_model": 128,
                "out_dropout": 0.3,
            }
        }
        model = ModelBuilder.build_model(config).to("cuda")
        batch_size = 4
        x = torch.randn(batch_size, 3, 174, 512, 512).to("cuda")
        y = model(x)
        self.assertEqual(y.shape, (batch_size, 4), "Output shape should be (batch_size, n_classes)")


if __name__ == "__main__":
    unittest.main()
