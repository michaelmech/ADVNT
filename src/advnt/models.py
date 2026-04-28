"""Neural network components for domain-adversarial learning."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted

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


class _BaseAdversarialMLP(BaseEstimator):
    """Shared PyTorch training utilities for neural ADVNT estimators."""

    def __init__(
        self,
        hidden_dims=(64, 32),
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=256,
        max_epochs=30,
        weight_decay=1e-5,
        lambda_grl=1.0,
        random_state=42,
        device="auto",
        verbose=False,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.lambda_grl = lambda_grl
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _check_torch(self):
        if torch is None:  # pragma: no cover - runtime guard
            raise ImportError(
                f"PyTorch is required for {self.__class__.__name__}. "
                "Install torch or use a non-neural estimator."
            )

    def _resolve_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _build_backbone(self, n_features):
        layers = []
        in_dim = n_features

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            in_dim = int(hidden_dim)

        if not layers:
            layers.append(nn.Identity())

        return nn.Sequential(*layers), in_dim

    def _set_feature_importances(self):
        first_linear = None
        for layer in self.backbone_:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break

        if first_linear is None:
            self.feature_importances_ = None
            return

        w = first_linear.weight.detach().cpu().numpy()
        self.feature_importances_ = np.mean(np.abs(w), axis=0)


class AdversarialValidationMLPClassifier(_BaseAdversarialMLP, ClassifierMixin):
    """Sklearn-style PyTorch MLP classifier with an adversarial-validation head.

    This estimator is intended to be used as the model inside
    :class:`advnt.validation.AdversarialValidator`, where ``y`` is the binary
    domain label (0=train, 1=test).
    """

    def _build_network(self, n_features):
        backbone, out_dim = self._build_backbone(n_features)
        av_head = nn.Sequential(
            GradientReversalLayer(lambda_=self.lambda_grl),
            nn.Linear(out_dim, 1),
        )
        return backbone, av_head

    def fit(self, X, y):
        self._check_torch()
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        classes = np.unique(y)
        if classes.size != 2 or not np.array_equal(classes, np.array([0.0, 1.0])):
            raise ValueError("AdversarialValidationMLPClassifier expects binary y encoded as {0, 1}.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])

        self.backbone_, self.av_head_ = self._build_network(self.n_features_in_)
        self.device_ = self._resolve_device()

        self.backbone_.to(self.device_)
        self.av_head_.to(self.device_)

        params = list(self.backbone_.parameters()) + list(self.av_head_.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

        self.backbone_.train()
        self.av_head_.train()

        for epoch in range(self.max_epochs):
            indices = torch.randperm(X_tensor.shape[0])
            epoch_loss = 0.0

            for start in range(0, X_tensor.shape[0], self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = X_tensor[batch_idx].to(self.device_)
                yb = y_tensor[batch_idx].to(self.device_)

                optimizer.zero_grad()
                feats = self.backbone_(xb)
                logits = self.av_head_(feats).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())

            if self.verbose:
                denom = max(1, int(np.ceil(X_tensor.shape[0] / self.batch_size)))
                print(f"epoch={epoch + 1} loss={epoch_loss / denom:.4f}")

        self._set_feature_importances()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["backbone_", "av_head_"])
        X = check_array(X, ensure_2d=True, dtype=np.float32)

        self.backbone_.eval()
        self.av_head_.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            feats = self.backbone_(xb)
            logits = self.av_head_(feats).squeeze(1)
        return logits.detach().cpu().numpy()

    def predict_proba(self, X):
        logits = self.decision_function(X)
        probs_1 = 1.0 / (1.0 + np.exp(-logits))
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AdversarialValidationMLPRegressor(_BaseAdversarialMLP, RegressorMixin):
    """Sklearn-style PyTorch MLP regressor with optional AV-domain head.

    Parameters
    ----------
    domain_loss_weight:
        Weight applied to the adversarial domain loss when ``domain_y`` is
        provided in :meth:`fit`.

    Notes
    -----
    ``fit(X, y, domain_y=None)`` uses pure regression loss when ``domain_y`` is
    omitted. When ``domain_y`` is provided (binary {0, 1}), the model jointly
    optimizes regression loss and adversarial domain loss.
    """

    def __init__(
        self,
        hidden_dims=(64, 32),
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=256,
        max_epochs=30,
        weight_decay=1e-5,
        lambda_grl=1.0,
        domain_loss_weight=0.2,
        random_state=42,
        device="auto",
        verbose=False,
    ):
        super().__init__(
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
            lambda_grl=lambda_grl,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        self.domain_loss_weight = domain_loss_weight

    def _build_network(self, n_features):
        backbone, out_dim = self._build_backbone(n_features)
        target_head = nn.Linear(out_dim, 1)
        av_head = nn.Sequential(
            GradientReversalLayer(lambda_=self.lambda_grl),
            nn.Linear(out_dim, 1),
        )
        return backbone, target_head, av_head

    def fit(self, X, y, domain_y=None):
        self._check_torch()
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        if domain_y is not None:
            domain_y = np.asarray(domain_y, dtype=np.float32).reshape(-1)
            if domain_y.shape[0] != X.shape[0]:
                raise ValueError("domain_y must have the same number of rows as X.")
            classes = np.unique(domain_y)
            if classes.size != 2 or not np.array_equal(classes, np.array([0.0, 1.0])):
                raise ValueError("domain_y must be binary and encoded as {0, 1}.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.backbone_, self.target_head_, self.av_head_ = self._build_network(self.n_features_in_)
        self.device_ = self._resolve_device()

        self.backbone_.to(self.device_)
        self.target_head_.to(self.device_)
        self.av_head_.to(self.device_)

        params = (
            list(self.backbone_.parameters())
            + list(self.target_head_.parameters())
            + list(self.av_head_.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        domain_tensor = None
        if domain_y is not None:
            domain_tensor = torch.as_tensor(domain_y, dtype=torch.float32)

        self.backbone_.train()
        self.target_head_.train()
        self.av_head_.train()

        for epoch in range(self.max_epochs):
            indices = torch.randperm(X_tensor.shape[0])
            epoch_loss = 0.0

            for start in range(0, X_tensor.shape[0], self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = X_tensor[batch_idx].to(self.device_)
                yb = y_tensor[batch_idx].to(self.device_)

                optimizer.zero_grad()

                feats = self.backbone_(xb)
                pred = self.target_head_(feats).squeeze(1)
                loss = mse(pred, yb)

                if domain_tensor is not None:
                    db = domain_tensor[batch_idx].to(self.device_)
                    dlogits = self.av_head_(feats).squeeze(1)
                    loss = loss + float(self.domain_loss_weight) * bce(dlogits, db)

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())

            if self.verbose:
                denom = max(1, int(np.ceil(X_tensor.shape[0] / self.batch_size)))
                print(f"epoch={epoch + 1} loss={epoch_loss / denom:.4f}")

        self._set_feature_importances()
        return self

    def predict(self, X):
        check_is_fitted(self, ["backbone_", "target_head_"])
        X = check_array(X, ensure_2d=True, dtype=np.float32)

        self.backbone_.eval()
        self.target_head_.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            feats = self.backbone_(xb)
            pred = self.target_head_(feats).squeeze(1)
        return pred.detach().cpu().numpy()

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        return r2_score(y, self.predict(X))
