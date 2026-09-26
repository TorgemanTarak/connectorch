"""Backend parity and the invariants that keep large graphs out of dense memory."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from connectorch import Connectome, ConnectorchMemoryError
from connectorch.backends import build_propagator
from connectorch.nn import ConnectomeRNN

BACKENDS = ["dense", "sparse_mm", "sparse_trainable", "scatter"]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def random_connectome(num_nodes: int, density: float, seed: int) -> Connectome:
    rng = np.random.default_rng(seed)
    num_edges = max(2, int(num_nodes * num_nodes * density))
    return Connectome.from_edges(
        source=rng.integers(0, num_nodes, num_edges),
        target=rng.integers(0, num_nodes, num_edges),
        weight=rng.normal(size=num_edges),
        nodes={"node_id": np.arange(num_nodes)},
    )


def propagate(backend: str, brain: Connectome, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    edge_index = torch.as_tensor(brain.edge_index, dtype=torch.int64, device=h.device)
    propagator = build_propagator(backend, edge_index, brain.num_nodes, trainable=False)
    return propagator.to(h.device)(h, w)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orientation_is_target_from_source(backend: str) -> None:
    """A --2--> B --3--> C, one step from h=[1,0,0], must put 2 on B and nothing on C."""
    brain = Connectome.from_edges(source=["A", "B"], target=["B", "C"], weight=[2.0, 3.0])
    h = torch.tensor([[1.0], [0.0], [0.0]])
    w = torch.as_tensor(brain.edge_attribute("weight"), dtype=torch.float32)
    assert propagate(backend, brain, h, w).flatten().tolist() == [0.0, 2.0, 0.0]


@pytest.mark.parametrize("num_nodes", [5, 50, 300])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backends_agree_on_forward(num_nodes: int, dtype: torch.dtype) -> None:
    brain = random_connectome(num_nodes, 0.05, seed=num_nodes)
    h = torch.randn(num_nodes, 4, dtype=dtype)
    w = torch.as_tensor(brain.edge_attribute("weight"), dtype=dtype)
    tolerance = 1e-6 if dtype is torch.float32 else 1e-12

    reference = propagate("dense", brain, h, w)
    for backend in ("sparse_mm", "sparse_trainable", "scatter"):
        got = propagate(backend, brain, h, w)
        assert torch.allclose(got, reference, atol=tolerance), backend


@pytest.mark.parametrize("num_nodes", [5, 50, 300])
def test_backends_agree_on_gradients(num_nodes: int) -> None:
    brain = random_connectome(num_nodes, 0.05, seed=num_nodes + 1)
    h = torch.randn(num_nodes, 4, dtype=torch.float64)
    base = torch.as_tensor(brain.edge_attribute("weight"), dtype=torch.float64)

    grads = {}
    for backend in BACKENDS:
        w = base.clone().requires_grad_(True)
        propagate(backend, brain, h, w).square().sum().backward()
        grads[backend] = w.grad

    for backend in ("sparse_mm", "sparse_trainable", "scatter"):
        assert torch.allclose(grads[backend], grads["dense"], atol=1e-10), backend


def test_sparse_trainable_state_gradient_agrees_with_dense() -> None:
    brain = random_connectome(30, 0.1, seed=91)
    weight = torch.as_tensor(brain.edge_attribute("weight"), dtype=torch.float64)
    initial_state = torch.randn(30, 5, dtype=torch.float64)
    gradients = {}
    for backend in ("dense", "sparse_trainable"):
        state = initial_state.clone().requires_grad_(True)
        propagate(backend, brain, state, weight).square().sum().backward()
        gradients[backend] = state.grad
    assert torch.allclose(gradients["sparse_trainable"], gradients["dense"], atol=1e-10)


@pytest.mark.parametrize(
    ("device", "amp_dtype"),
    [
        ("cpu", torch.bfloat16),
        pytest.param("cuda", torch.float16, marks=[CUDA, pytest.mark.cuda]),
        pytest.param("cuda", torch.bfloat16, marks=[CUDA, pytest.mark.cuda]),
    ],
)
@pytest.mark.parametrize("backward_under_autocast", [False, True])
def test_sparse_trainable_autocast_matches_scatter(
    device: str, amp_dtype: torch.dtype, backward_under_autocast: bool
) -> None:
    """Autocast must preserve the sparse kernels' dtype in both directions."""
    brain = random_connectome(30, 0.1, seed=93)
    weight = torch.as_tensor(
        brain.edge_attribute("weight"), dtype=torch.float32, device=device
    ).requires_grad_(True)
    state = torch.linspace(-1, 1, 150, device=device).reshape(30, 5).requires_grad_(True)
    reference = propagate("scatter", brain, state, weight)
    expected = torch.autograd.grad(reference.square().mean(), (weight, state))

    with torch.autocast(device, dtype=amp_dtype):
        output = propagate("sparse_trainable", brain, state, weight)
        loss = output.square().mean()
    with torch.autocast(device, dtype=amp_dtype, enabled=backward_under_autocast):
        actual = torch.autograd.grad(loss, (weight, state))

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(actual, expected)


def test_shuffled_input_order_does_not_change_the_result() -> None:
    """Canonical edge ordering must make weights and edges impossible to misalign."""
    rng = np.random.default_rng(7)
    source = rng.integers(0, 30, 200)
    target = rng.integers(0, 30, 200)
    weight = rng.normal(size=200)

    order = rng.permutation(200)
    a = Connectome.from_edges(source=source, target=target, weight=weight)
    b = Connectome.from_edges(source=source[order], target=target[order], weight=weight[order])
    assert a.fingerprint() == b.fingerprint()

    h = torch.randn(a.num_nodes, 3, dtype=torch.float64)
    for backend in BACKENDS:
        out_a = propagate(backend, a, h, torch.as_tensor(a.edge_attribute("weight")))
        out_b = propagate(backend, b, h, torch.as_tensor(b.edge_attribute("weight")))
        assert torch.allclose(out_a, out_b), backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_state_propagates_to_zero(backend: str) -> None:
    brain = random_connectome(20, 0.1, seed=3)
    h = torch.zeros(20, 2)
    w = torch.as_tensor(brain.edge_attribute("weight"), dtype=torch.float32)
    assert torch.count_nonzero(propagate(backend, brain, h, w)) == 0


# ----------------------------------------------------------------------
# the memory invariant
# ----------------------------------------------------------------------


def _tiny_edge_index(num_edges: int = 10) -> torch.Tensor:
    """Distinct edges in canonical order, for tests about graph *size* not content."""
    source = torch.arange(num_edges, dtype=torch.int64)
    return torch.stack([source, source + 1])


def test_dense_backend_refuses_a_large_graph() -> None:
    edge_index = _tiny_edge_index()
    with pytest.raises(ConnectorchMemoryError) as excinfo:
        build_propagator("dense", edge_index, 166_691, trainable=False)
    message = str(excinfo.value)
    assert "166,691" in message
    assert "103.5 GiB" in message, "the estimate must be computed, not vague"
    assert "scatter" in message


def test_sparse_mm_refuses_trainable_weights_on_a_large_graph() -> None:
    """sparse.mm backward materialises a dense [N, N] gradient. Refuse before the OOM."""
    edge_index = _tiny_edge_index()
    with pytest.raises(ConnectorchMemoryError, match="dense adjacency gradient"):
        build_propagator("sparse_mm", edge_index, 166_691, trainable=True)


def test_auto_picks_scatter_for_training_and_sparse_mm_for_inference() -> None:
    brain = random_connectome(40, 0.05, seed=11)
    assert ConnectomeRNN(brain, weights="trainable", initializer="weight").backend == "scatter"
    assert ConnectomeRNN(brain, weights="weight").backend == "sparse_mm"


def test_sparse_trainable_accepts_a_large_trainable_graph() -> None:
    edge_index = _tiny_edge_index()
    propagator = build_propagator("sparse_trainable", edge_index, 166_691, trainable=True)
    assert propagator.backend_name == "sparse_trainable"


def test_unknown_backend_names_the_alternatives() -> None:
    brain = random_connectome(10, 0.2, seed=5)
    with pytest.raises(Exception, match="unknown backend"):
        ConnectomeRNN(brain, weights="weight", backend="magic")


# ----------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_agree_on_cuda(backend: str) -> None:
    brain = random_connectome(200, 0.05, seed=21)
    h = torch.randn(200, 8, device="cuda")
    w = torch.as_tensor(brain.edge_attribute("weight"), dtype=torch.float32, device="cuda")
    reference = propagate("dense", brain, h, w)
    assert torch.allclose(propagate(backend, brain, h, w), reference, atol=1e-4)


@CUDA
@pytest.mark.cuda
def test_sparse_trainable_gradients_work_on_cuda() -> None:
    brain = random_connectome(2_000, 0.001, seed=92)
    state = torch.randn(2_000, 8, device="cuda", requires_grad=True)
    weight = torch.as_tensor(
        brain.edge_attribute("weight"), dtype=torch.float32, device="cuda"
    ).requires_grad_(True)
    propagate("sparse_trainable", brain, state, weight).square().mean().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()


@pytest.mark.parametrize("backend", ["sparse_mm", "sparse_trainable"])
def test_csr_backends_refuse_parallel_edges(backend: str) -> None:
    """CSR has one value per coordinate; its backward would misalign edge_weight."""
    from connectorch.exceptions import BackendError

    brain = Connectome.from_edges(
        source=[0, 0, 1],
        target=[1, 1, 2],
        weight=[1.0, 2.0, 3.0],
        aggregate_parallel_edges=False,
    )
    with pytest.raises(BackendError, match="parallel edges"):
        ConnectomeRNN(brain, weights="weight", backend=backend)


def test_scatter_handles_parallel_edges() -> None:
    """The training backend sums parallel edges correctly, forward and backward."""
    brain = Connectome.from_edges(
        source=[0, 0, 1],
        target=[1, 1, 2],
        weight=[1.0, 2.0, 3.0],
        aggregate_parallel_edges=False,
    )
    model = ConnectomeRNN(
        brain,
        weights="trainable",
        initializer="weight",
        backend="scatter",
        activation="identity",
        leak=1.0,
    )
    y = model(torch.zeros(1, 3), steps=1, state=torch.tensor([[1.0, 0.0, 0.0]]))
    assert y[0, 0].tolist() == [0.0, 3.0, 0.0]
    y.sum().backward()
    assert model.edge_weight.grad.shape == (3,)
