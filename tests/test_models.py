import unittest

import torch
from torch import nn

from attention import AdditiveAttention, DotProductAttention, MultiHeadAttention
from date2date.data import DEVICE, vocab_size
from date2date.experiment import MODEL_NAMES, build_models


class Date2DateModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.inputs = torch.randint(0, vocab_size, (2, 10), device=DEVICE)
        self.targets = torch.randint(0, vocab_size, (2, 11), device=DEVICE)

    def test_all_models_support_batched_training_shapes(self):
        for model_type in MODEL_NAMES:
            with self.subTest(model_type=model_type):
                encoder, decoder = build_models(
                    model_type,
                    hidden_size=8,
                    attention_heads=2,
                )
                encoder_output, encoder_hidden = encoder(self.inputs)
                decoder_output, decoder_hidden = decoder(encoder_output, self.targets)

                self.assertEqual(encoder_output.shape, (2, 10, 8))
                self.assertEqual(encoder_hidden.shape, (1, 2, 8))
                self.assertEqual(decoder_output.shape, (2, 11, vocab_size))
                self.assertEqual(decoder_hidden.shape, (1, 2, 8))
                self.assertTrue(torch.isfinite(decoder_output).all())

                loss = nn.CrossEntropyLoss()(
                    decoder_output.reshape(-1, vocab_size),
                    self.targets.reshape(-1),
                )
                loss.backward()
                for model in (encoder, decoder):
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    self.assertTrue(all(gradient is not None for gradient in gradients))

    def test_all_models_support_batched_autoregressive_shapes(self):
        for model_type in MODEL_NAMES:
            with self.subTest(model_type=model_type):
                encoder, decoder = build_models(
                    model_type,
                    hidden_size=8,
                    attention_heads=2,
                )
                encoder_output, _ = encoder(self.inputs)
                decoder_output, decoder_hidden = decoder(
                    encoder_output,
                    target_tensor=None,
                    max_len=7,
                )

                self.assertEqual(decoder_output.shape, (2, 7, vocab_size))
                self.assertEqual(decoder_hidden.shape, (1, 2, 8))
                self.assertTrue(torch.isfinite(decoder_output).all())

    def test_attention_models_use_expected_attention_layers(self):
        expected_layers = {
            "bahdanau": AdditiveAttention,
            "dot_product_attention": DotProductAttention,
            "multi_head_attention": MultiHeadAttention,
        }
        for model_type, attention_type in expected_layers.items():
            with self.subTest(model_type=model_type):
                _, decoder = build_models(
                    model_type,
                    hidden_size=8,
                    attention_heads=3,
                )
                self.assertIsInstance(decoder.attention, attention_type)

        _, decoder = build_models(
            "multi_head_attention",
            hidden_size=8,
            attention_heads=3,
        )
        self.assertEqual(decoder.attention.head_count, 3)


if __name__ == "__main__":
    unittest.main()
