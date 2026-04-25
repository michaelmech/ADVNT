"""Neural network components for domain-adversarial learning."""

from __future__ import annotations


try:
    import torch
    from torch import nn
    from torch.autograd import Function
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None
    nn = object
    Function = object


if torch is not None:

    class _GradientReverse(Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module if torch is not None else object):
    """Gradient reversal layer used for domain-adversarial training."""

    def __init__(self, lambda_=1.0):
        if torch is None:  # pragma: no cover - runtime guard
            raise ImportError("PyTorch is required to use GradientReversalLayer.")
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return _GradientReverse.apply(x, self.lambda_)
