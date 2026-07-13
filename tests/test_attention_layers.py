import math
import unittest

import torch
from torch.nn import functional as F

from attention import (
    AdditiveAttention,
    DotProductAttention,
    MultiHeadAttention,
    ScalarAdditiveAttention,
)


class DotProductAttentionTest(unittest.TestCase):
    def test_matches_scaled_dot_product_reference_and_backpropagates(self):
        torch.manual_seed(1)
        queries = torch.randn(2, 3, 4, requires_grad=True)
        keys = torch.randn(2, 5, 4, requires_grad=True)
        values = torch.randn(2, 5, 6, requires_grad=True)
        attention = DotProductAttention()

        output = attention(queries, keys, values)
        expected_weights = F.softmax(
            torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(4), dim=-1
        )
        expected_output = torch.bmm(expected_weights, values)

        self.assertEqual(output.shape, (2, 3, 6))
        self.assertEqual(attention.attention_weights.shape, (2, 3, 5))
        torch.testing.assert_close(attention.attention_weights, expected_weights)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(
            attention.attention_weights.sum(dim=-1), torch.ones(2, 3)
        )
        self.assertTrue(torch.all(attention.attention_weights >= 0))

        output.square().mean().backward()
        for tensor in (queries, keys, values):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_batches_are_independent(self):
        torch.manual_seed(2)
        queries = torch.randn(2, 2, 3)
        keys = torch.randn(2, 4, 3)
        values = torch.randn(2, 4, 5)
        attention = DotProductAttention()

        batched = attention(queries, keys, values)
        first = attention(queries[:1], keys[:1], values[:1])
        second = attention(queries[1:], keys[1:], values[1:])

        torch.testing.assert_close(batched[:1], first)
        torch.testing.assert_close(batched[1:], second)


class AdditiveAttentionTest(unittest.TestCase):
    def test_matches_bahdanau_reference_for_multiple_queries(self):
        torch.manual_seed(3)
        attention = AdditiveAttention(query_size=3, key_size=4, hidden_size=5)
        queries = torch.randn(2, 2, 3, requires_grad=True)
        keys = torch.randn(2, 4, 4, requires_grad=True)
        values = torch.randn(2, 4, 6, requires_grad=True)

        output = attention(queries, keys, values)
        scores = attention.wv(
            torch.tanh(
                attention.wq(queries).unsqueeze(2)
                + attention.wk(keys).unsqueeze(1)
            )
        ).squeeze(-1)
        expected_weights = F.softmax(scores, dim=-1)
        expected_output = torch.bmm(expected_weights, values)

        self.assertEqual(output.shape, (2, 2, 6))
        self.assertEqual(attention.attention_weights.shape, (2, 2, 4))
        torch.testing.assert_close(attention.attention_weights, expected_weights)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(
            attention.attention_weights.sum(dim=-1), torch.ones(2, 2)
        )

        output.sum().backward()
        for tensor in (queries, keys, values):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for parameter in attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_single_query_keeps_query_dimension(self):
        attention = AdditiveAttention(query_size=3, key_size=3, hidden_size=4)
        output = attention(
            torch.randn(2, 1, 3),
            torch.randn(2, 5, 3),
            torch.randn(2, 5, 7),
        )

        self.assertEqual(output.shape, (2, 1, 7))
        self.assertEqual(attention.attention_weights.shape, (2, 1, 5))


class ScalarAdditiveAttentionTest(unittest.TestCase):
    def test_scalar_adapter_preserves_kernel_regression_shapes(self):
        attention = ScalarAdditiveAttention(hidden_size=4)
        queries = torch.randn(3)
        keys = torch.randn(3, 5)
        values = torch.randn(3, 5)

        output = attention(queries, keys, values)

        self.assertEqual(output.shape, (3,))
        self.assertEqual(attention.attention_weights.shape, (3, 5))
        torch.testing.assert_close(
            attention.attention_weights.sum(dim=-1), torch.ones(3)
        )


class MultiHeadAttentionTest(unittest.TestCase):
    def test_matches_manual_multi_head_reference_and_backpropagates(self):
        torch.manual_seed(4)
        attention = MultiHeadAttention(
            query_size=3,
            key_size=4,
            value_size=5,
            hidden_size=2,
            head_count=3,
        )
        queries = torch.randn(2, 4, 3, requires_grad=True)
        keys = torch.randn(2, 6, 4, requires_grad=True)
        values = torch.randn(2, 6, 5, requires_grad=True)

        output = attention(queries, keys, values)

        projected_queries = attention.wq(queries).reshape(2, 4, 3, 2).permute(0, 2, 1, 3)
        projected_keys = attention.wk(keys).reshape(2, 6, 3, 2).permute(0, 2, 1, 3)
        projected_values = attention.wv(values).reshape(2, 6, 3, 2).permute(0, 2, 1, 3)
        scores = torch.matmul(projected_queries, projected_keys.transpose(-2, -1)) / math.sqrt(2)
        expected_weights = F.softmax(scores, dim=-1)
        per_head = torch.matmul(expected_weights, projected_values)
        combined = per_head.permute(0, 2, 1, 3).reshape(2, 4, 6)
        expected_output = attention.wo(combined)

        self.assertEqual(output.shape, (2, 4, 2))
        self.assertEqual(attention.attention_weights.shape, (2, 3, 4, 6))
        torch.testing.assert_close(attention.attention_weights, expected_weights)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(
            attention.attention_weights.sum(dim=-1), torch.ones(2, 3, 4)
        )

        output.square().sum().backward()
        for tensor in (queries, keys, values):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for layer in (attention.wq, attention.wk, attention.wv, attention.wo):
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_split_and_combine_preserve_projected_order(self):
        attention = MultiHeadAttention(2, 2, 2, hidden_size=3, head_count=2)
        projected = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)

        split = attention.split_heads_for_attention(projected)
        combined = attention.combine_attention(split)

        self.assertEqual(split.shape, (4, 4, 3))
        torch.testing.assert_close(combined, projected)


if __name__ == "__main__":
    unittest.main()
