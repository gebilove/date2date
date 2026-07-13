import unittest
from unittest import mock

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

    def test_experiment_index_records_batch_size(self):
        self.assertIn("batch_size", INDEX_FIELDS)


if __name__ == "__main__":
    unittest.main()
