"""Reusable attention layers and helpers for the attention notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class NWKernelRegression(nn.Module):
    """Parametric Nadaraya-Watson kernel regression.

    Expected shapes:
        queries: (batch,)
        keys: (batch, num_kv)
        values: (batch, num_kv)
    """

    def __init__(self, init_weight: float | None = None) -> None:
        super().__init__()
        if init_weight is None:
            weight = torch.rand(1)
        else:
            weight = torch.tensor([init_weight], dtype=torch.float32)
        self.w = nn.Parameter(weight)
        self.attention_weights: torch.Tensor | None = None

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = queries.reshape(-1, 1)
        scores = -((queries - keys) * self.w) ** 2 / 2
        self.attention_weights = F.softmax(scores, dim=1)
        return (self.attention_weights * values).sum(dim=1)


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention for batched query and key sequences.

    Expected shapes:
        queries: (batch, num_queries, query_size)
        keys: (batch, num_keys, key_size)
        values: (batch, num_keys, value_size)
    """

    def __init__(self, query_size: int, key_size: int, hidden_size: int) -> None:
        super().__init__()
        self.wq = nn.Linear(query_size, hidden_size)
        self.wk = nn.Linear(key_size, hidden_size)
        self.wv = nn.Linear(hidden_size, 1)
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.wq(queries)
        keys = self.wk(keys)
        scores = self.wv(
            torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))
        ).squeeze(-1)
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.attention_weights, values)

class ScalarAdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = AdditiveAttention(
            query_size=1,
            key_size=1,
            hidden_size=hidden_size,
        )
        self.attention_weights: torch.Tensor | None = None

    def forward(self, queries, keys, values):
        queries = queries.reshape(-1, 1, 1)
        keys = keys.unsqueeze(-1)
        values = values.unsqueeze(-1)

        output = self.attention(queries, keys, values)
        self.attention_weights = self.attention.attention_weights.squeeze(1)
        return output.reshape(-1)


class DotProductAttention(nn.Module):
    """Scaled dot-product attention for batched query and key sequences."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        feature_size = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / feature_size**0.5
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.attention_weights, values)


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    ``hidden_size`` is the feature size of each head and of the final output.
    """

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        hidden_size: int,
        head_count: int,
    ) -> None:
        super().__init__()
        self.wq = nn.Linear(query_size, hidden_size * head_count)
        self.wk = nn.Linear(key_size, hidden_size * head_count)
        self.wv = nn.Linear(value_size, hidden_size * head_count)
        self.wo = nn.Linear(hidden_size * head_count, hidden_size)
        self.head_count = head_count
        self.hidden_size = hidden_size
        self.attention = DotProductAttention()
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        projected_queries = self.split_heads_for_attention(self.wq(queries))
        projected_keys = self.split_heads_for_attention(self.wk(keys))
        projected_values = self.split_heads_for_attention(self.wv(values))

        output = self.attention(
            projected_queries,
            projected_keys,
            projected_values,
        )
        batch_size = queries.shape[0]
        self.attention_weights = self.attention.attention_weights.reshape(
            batch_size,
            self.head_count,
            queries.shape[1],
            keys.shape[1],
        )
        return self.wo(self.combine_attention(output))

    def split_heads_for_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.reshape(
            tensor.shape[0], tensor.shape[1], self.head_count, self.hidden_size
        )
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(-1, tensor.shape[2], tensor.shape[3])

    def combine_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.reshape(
            tensor.shape[0] // self.head_count,
            self.head_count,
            tensor.shape[1],
            tensor.shape[2],
        )
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


@dataclass
class KernelRegressionData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_truth: torch.Tensor

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    @property
    def n_test(self) -> int:
        return len(self.x_test)


def target_function(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.sin(x) + x**0.8


def make_kernel_regression_data(
    n_train: int = 50,
    x_max: float = 5.0,
    test_step: float = 0.1,
    noise_std: float = 0.5,
    seed: int | None = 1,
) -> KernelRegressionData:
    if seed is not None:
        torch.manual_seed(seed)

    x_train, _ = torch.sort(torch.rand(n_train) * x_max)
    y_train = target_function(x_train) + torch.normal(0.0, noise_std, (n_train,))
    x_test = torch.arange(0, x_max, test_step)
    y_truth = target_function(x_test)
    return KernelRegressionData(x_train=x_train, y_train=y_train, x_test=x_test, y_truth=y_truth)


def make_leave_one_out_pairs(x_train: torch.Tensor, y_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_train = len(x_train)
    mask = (1 - torch.eye(n_train, device=x_train.device)).bool()
    keys = x_train.repeat(n_train, 1)[mask].reshape(n_train, -1)
    values = y_train.repeat(n_train, 1)[mask].reshape(n_train, -1)
    return keys, values


def make_test_pairs(x_train: torch.Tensor, y_train: torch.Tensor, n_test: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = x_train.repeat(n_test, 1)
    values = y_train.repeat(n_test, 1)
    return keys, values
