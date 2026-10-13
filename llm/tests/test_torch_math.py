# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from bsz_lm.model import GPT, initialize_model, parameter_groups
from bsz_lm.optimizers import Track3MatrixOptimizer


def test_w_and_h_initializations_and_parameter_partition():
    model = GPT(vocabulary_size=32, layers=1, width=8, head_dimension=4, mlp_multiplier=2)
    initialize_model(model, "W")
    assert torch.count_nonzero(model.blocks[0].attn.proj.weight) == 0
    assert torch.count_nonzero(model.proj.weight) == 0
    assert sum(len(values) for values in parameter_groups(model).values()) == len(
        list(model.parameters())
    )
    initialize_model(model, "H")
    assert torch.count_nonzero(model.blocks[0].attn.proj.weight) > 0
    assert torch.count_nonzero(model.proj.weight) > 0


def test_adamh_wrapper_preserves_norm_and_adamw_decays():
    hyper_parameter = torch.nn.Parameter(torch.randn(3, 2))
    initial_norm = hyper_parameter.norm()
    hyper = Track3MatrixOptimizer(
        [hyper_parameter],
        algorithm="adam",
        family="H",
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        momentum=None,
        shampoo_beta=None,
        weight_decay=0.0,
    )
    hyper_parameter.grad = torch.randn_like(hyper_parameter)
    hyper.step()
    assert torch.allclose(hyper_parameter.norm(), initial_norm, atol=1e-5)

    weight = torch.nn.Parameter(torch.ones(3, 2))
    standard = Track3MatrixOptimizer(
        [weight],
        algorithm="adam",
        family="W",
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        momentum=None,
        shampoo_beta=None,
        weight_decay=0.2,
    )
    weight.grad = torch.zeros_like(weight)
    standard.step()
    assert torch.allclose(weight, torch.full_like(weight, 0.98))


FIELDS = {
    "muon": dict(beta1=None, beta2=None, momentum=0.95, shampoo_beta=None),
    "klsoap": dict(beta1=0.9, beta2=0.95, momentum=None, shampoo_beta=0.9),
    "adam": dict(beta1=0.9, beta2=0.99, momentum=None, shampoo_beta=None),
    "lion": dict(beta1=0.9, beta2=0.99, momentum=None, shampoo_beta=None),
}


@pytest.mark.parametrize("family", ["W", "H"])
@pytest.mark.parametrize("algorithm", ["adam", "klsoap", "lion", "muon"])
def test_every_matrix_optimizer_steps_under_both_wrappers(algorithm, family):
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    start = parameter.detach().clone()
    optimizer = Track3MatrixOptimizer(
        [parameter], algorithm=algorithm, family=family, lr=0.01, weight_decay=0.0,
        **FIELDS[algorithm],
    )
    gradient = torch.randn_like(parameter)
    parameter.grad = gradient.clone()
    optimizer.step()
    if algorithm == "klsoap":
        # the first KL-SOAP step only builds the preconditioner from the gradient
        assert torch.equal(parameter.detach(), start)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    moved = parameter.detach()
    assert torch.isfinite(moved).all() and not torch.equal(moved, start)
    if family == "H":
        assert torch.allclose(moved.norm(), start.norm(), rtol=1e-5)
        return
    step = (start - moved) / 0.01
    if algorithm in ("lion", "adam"):
        # first step from zero moments: Lion is sign(g); bias-corrected Adam is g / |g|
        assert torch.allclose(step, gradient.sign(), atol=1e-4)
    elif algorithm == "muon":
        # orthogonalized (approximately, in bf16) and scaled by sqrt(max(1, rows / columns))
        singular = torch.linalg.svdvals(step / (6 / 4) ** 0.5)
        assert singular.min() > 0.5 and singular.max() < 1.5
