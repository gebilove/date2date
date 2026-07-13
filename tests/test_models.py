import unittest

import torch
from torch import nn

from attention import AdditiveAttention
from date2date.data import DEVICE, vocab_size
from date2date.experiment import build_models


class Date2DateModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.inputs = torch.randint(0, vocab_size, (2, 10), device=DEVICE)
        self.targets = torch.randint(0, vocab_size, (2, 11), device=DEVICE)

    def test_vanilla_and_bahdanau_support_batched_training_shapes(self):
        for model_type in ("vanilla_gru", "bahdanau"):
            with self.subTest(model_type=model_type):
                encoder, decoder = build_models(model_type, hidden_size=8)
                encoder_output, encoder_hidden = encoder(self.inputs)
                decoder_output, decoder_hidden = decoder(encoder_output, self.targets)

                self.assertEqual(encoder_output.shape, (2, 10, 8))
                self.assertEqual(encoder_hidden.shape, (1, 2, 8))
                self.assertEqual(decoder_output.shape, (2, 11, vocab_size))
                self.assertEqual(decoder_hidden.shape, (1, 2, 8))

                loss = nn.CrossEntropyLoss()(
                    decoder_output.reshape(-1, vocab_size), self.targets.reshape(-1)
                )
                loss.backward()
                for model in (encoder, decoder):
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    self.assertTrue(all(gradient is not None for gradient in gradients))
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

                autoregressive_output, autoregressive_hidden = decoder(
                    encoder_output.detach(), target_tensor=None, max_len=4
                )
                self.assertEqual(autoregressive_output.shape, (2, 4, vocab_size))
                self.assertEqual(autoregressive_hidden.shape, (1, 2, 8))

    def test_bahdanau_decoder_reuses_shared_additive_attention(self):
        encoder, decoder = build_models("bahdanau", hidden_size=8)
        encoder_output, _ = encoder(self.inputs)
        decoder_output, _ = decoder(encoder_output, self.targets)

        self.assertIsInstance(decoder.attention, AdditiveAttention)
        self.assertEqual(decoder_output.shape, (2, 11, vocab_size))
        self.assertEqual(decoder.attention.attention_weights.shape, (2, 1, 10))
        torch.testing.assert_close(
            decoder.attention.attention_weights.sum(dim=-1), torch.ones(2, 1)
        )
        state_keys = decoder.state_dict().keys()
        self.assertIn("attention.wq.weight", state_keys)
        self.assertIn("attention.wk.weight", state_keys)
        self.assertIn("attention.wv.weight", state_keys)


if __name__ == "__main__":
    unittest.main()
