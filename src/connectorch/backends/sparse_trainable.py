"""Trainable CSR propagation without a dense adjacency gradient.

The forward is ordinary CSR sparse matrix multiplication. The custom backward
computes gradients for existing edge values with sampled dense-dense matrix
multiplication (SDDMM), so it never materialises the conceptual ``[N, N]``
gradient. Unlike gather-scatter, it also avoids ``[E, B]`` message tensors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..exceptions import BackendError
from .base import _check_shapes, register
from .sparse_mm import SparseMMPropagator

__all__ = ["SparseTrainablePropagator"]


class _SparseTrainableMM(torch.autograd.Function):
    """CSR SpMM forward with sparse sampled gradients for the stored values."""

    @staticmethod
    def forward(
        ctx: Any,
        crow_indices: Tensor,
        col_indices: Tensor,
        values: Tensor,
        state: Tensor,
        num_nodes: int,
    ) -> Tensor:
        adjacency = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=(num_nodes, num_nodes),
            check_invariants=False,
        )
        ctx.save_for_backward(crow_indices, col_indices, values, state)
        ctx.num_nodes = num_nodes
        return torch.sparse.mm(adjacency, state)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[None, None, Tensor | None, Tensor | None, None]:
        crow_indices, col_indices, values, state = ctx.saved_tensors
        adjacency = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=(ctx.num_nodes, ctx.num_nodes),
            check_invariants=False,
        )
        grad_values = grad_state = None
        if ctx.needs_input_grad[2]:
            try:
                # beta=0 means adjacency's values only provide the sparsity pattern.
                grad_values = torch.sparse.sampled_addmm(
                    adjacency, grad_output, state.transpose(0, 1), beta=0
                ).values()
            except RuntimeError as error:
                raise BackendError(
                    'backend="sparse_trainable" needs torch.sparse.sampled_addmm '
                    f"support for CSR tensors on {values.device.type}: {error}"
                ) from error
        if ctx.needs_input_grad[3]:
            grad_state = torch.sparse.mm(adjacency.transpose(0, 1), grad_output)
        return None, None, grad_values, grad_state, None


@register("sparse_trainable")
class SparseTrainablePropagator(SparseMMPropagator):
    """CSR propagation with gradients restricted to the existing edge pattern.

    The backend requires unique ``(source, target)`` pairs, as every CSR
    coordinate has one value. It supports CPU and CUDA in PyTorch versions where
    ``torch.sparse.sampled_addmm`` supports CSR on that device.
    """

    supports_sparse_backward = True
    saves_edge_batch_activations = False

    def forward(self, h: Tensor, edge_weight: Tensor) -> Tensor:
        """Return ``[N, B]`` messages for a ``[N, B]`` state."""
        _check_shapes(self, h, edge_weight)
        return _SparseTrainableMM.apply(
            self.crow_indices,
            self.col_indices,
            edge_weight,
            h,
            self.num_nodes,
        )

    def activation_bytes(self, batch: int, steps: int, itemsize: int) -> int:
        """Conservatively count one saved ``[N, B]`` state per recurrent step."""
        return self.num_nodes * batch * steps * itemsize

    def activation_summary(self, batch: int, steps: int) -> str:
        return f"{self.num_nodes:,} nodes x batch {batch} x {steps} steps x 1 saved state"
