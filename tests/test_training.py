import math
import unittest

import torch
from torch import nn

from date2date.experiment import MODEL_NAMES, build_models
from date2date.train import train_batch_samples


class TrainBatchSamplesTest(unittest.TestCase):
    def _training_components(self, model_type="vanilla_gru"):
        encoder, decoder = build_models(model_type, hidden_size=8)
        return (
            encoder,
            decoder,
            torch.optim.Adam(encoder.parameters(), lr=0.01),
            torch.optim.Adam(decoder.parameters(), lr=0.01),
            nn.CrossEntropyLoss(),
        )

    def _call(self, input_strs, target_strs, model_type="vanilla_gru"):
        components = self._training_components(model_type)
        return train_batch_samples(input_strs, target_strs, *components)

    def test_rejects_scalar_strings(self):
        with self.assertRaisesRegex(TypeError, "single string"):
            self._call("2002-01-23", ["23/01/2002"])
        with self.assertRaisesRegex(TypeError, "single string"):
            self._call(["2002-01-23"], "23/01/2002")

    def test_rejects_invalid_batches(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self._call([], [])
        with self.assertRaisesRegex(ValueError, "same number"):
            self._call(["2002-01-23"], ["23/01/2002", "08/12/1999"])
        with self.assertRaisesRegex(TypeError, "every item"):
            self._call(["2002-01-23", 123], ["23/01/2002", "08/12/1999"])

    def test_singleton_batch_is_supported(self):
        loss = self._call(["2002-01-23"], ["23/01/2002"])
        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(loss, 0)

    def test_all_models_update_parameters_for_a_real_batch(self):
        inputs = ["2002-01-23", "1999-12-08"]
        targets = ["23/01/2002", "08/12/1999"]
        for model_type in MODEL_NAMES:
            with self.subTest(model_type=model_type):
                components = self._training_components(model_type)
                encoder, decoder = components[:2]
                encoder_before = [parameter.detach().clone() for parameter in encoder.parameters()]
                decoder_before = [parameter.detach().clone() for parameter in decoder.parameters()]

                loss = train_batch_samples(inputs, targets, *components)

                self.assertTrue(math.isfinite(loss))
                self.assertTrue(
                    any(
                        not torch.equal(before, after)
                        for before, after in zip(encoder_before, encoder.parameters())
                    )
                )
                self.assertTrue(
                    any(
                        not torch.equal(before, after)
                        for before, after in zip(decoder_before, decoder.parameters())
                    )
                )


if __name__ == "__main__":
    unittest.main()
