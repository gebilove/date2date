import sys
import unittest
from unittest import mock

from attention import MultiHeadAttention
from date2date import experiment
from date2date.experiment_log import INDEX_FIELDS


class ExperimentBatchingTest(unittest.TestCase):
    def test_train_epochs_shuffles_reproducibly_and_keeps_short_batch(self):
        samples = [
            ("2000-01-01", "01/01/2000"),
            ("2001-02-02", "02/02/2001"),
            ("2002-03-03", "03/03/2002"),
            ("2003-04-04", "04/04/2003"),
            ("2004-05-05", "05/05/2004"),
        ]

        def collect_orders(seed):
            calls = []

            def fake_train(input_strs, target_strs, *args):
                calls.append(tuple(input_strs))
                return 1.0

            with mock.patch.object(experiment, "train_batch_samples", fake_train):
                losses = experiment.train_epochs(
                    samples.copy(),
                    mock.sentinel.encoder,
                    mock.sentinel.decoder,
                    mock.sentinel.encoder_optimizer,
                    mock.sentinel.decoder_optimizer,
                    mock.sentinel.criterion,
                    epochs=2,
                    batch_size=2,
                    seed=seed,
                )
            return calls, losses

        first_calls, first_losses = collect_orders(17)
        second_calls, second_losses = collect_orders(17)

        self.assertEqual(first_calls, second_calls)
        self.assertEqual(first_losses, [1.0, 1.0])
        self.assertEqual(second_losses, [1.0, 1.0])
        self.assertEqual([len(batch) for batch in first_calls], [2, 2, 1, 2, 2, 1])
        first_epoch = sum((list(batch) for batch in first_calls[:3]), [])
        second_epoch = sum((list(batch) for batch in first_calls[3:]), [])
        self.assertCountEqual(first_epoch, [sample[0] for sample in samples])
        self.assertCountEqual(second_epoch, [sample[0] for sample in samples])
        self.assertNotEqual(first_epoch, second_epoch)

    def test_model_names_include_all_sequence_models(self):
        self.assertEqual(
            experiment.MODEL_NAMES,
            (
                "vanilla_gru",
                "bahdanau",
                "dot_product_attention",
                "multi_head_attention",
            ),
        )

    def test_build_models_uses_requested_attention_head_count(self):
        _, decoder = experiment.build_models(
            "multi_head_attention",
            hidden_size=8,
            attention_heads=3,
        )
        self.assertIsInstance(decoder.attention, MultiHeadAttention)
        self.assertEqual(decoder.attention.head_count, 3)

    def test_build_models_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            experiment.build_models("unknown", hidden_size=8)

    def test_parse_args_supports_attention_heads(self):
        with mock.patch.object(sys, "argv", ["experiment"]):
            args = experiment.parse_args()
        self.assertEqual(args.attention_heads, 4)

        with mock.patch.object(
            sys,
            "argv",
            [
                "experiment",
                "--model",
                "multi_head_attention",
                "--attention-heads",
                "3",
            ],
        ):
            args = experiment.parse_args()
        self.assertEqual(args.model, "multi_head_attention")
        self.assertEqual(args.attention_heads, 3)

    def test_parse_args_rejects_non_positive_attention_heads(self):
        with mock.patch.object(
            sys,
            "argv",
            ["experiment", "--attention-heads", "0"],
        ):
            with self.assertRaises(SystemExit):
                experiment.parse_args()

    def test_experiment_index_records_batch_and_attention_configuration(self):
        self.assertIn("batch_size", INDEX_FIELDS)
        self.assertIn("attention_heads", INDEX_FIELDS)


if __name__ == "__main__":
    unittest.main()
