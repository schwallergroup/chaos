# from __future__ import annotations
from inspect import signature
from typing import Optional, Union

import botorch
import pytorch_lightning as pl
import torch
from botorch import fit_gpytorch_model
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import (
    MIN_INFERRED_NOISE_LEVEL,
    FixedNoiseGP,
    HeteroskedasticSingleTaskGP,
    SingleTaskGP,
)
from botorch.models.transforms.input import InputTransform, Normalize, Warp
from botorch.models.transforms.outcome import Log, OutcomeTransform, Standardize
from botorch.models.utils import validate_input_scaling

# from torch import Tensor
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan, Interval, LessThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import LaplaceLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.priors import LogNormalPrior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor

from additive_bo.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel,
)


class GP(SingleTaskGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_val: float = 0.015,
        fix_noise: bool = False,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ):

        warp_tf = None
        if input_warping:
            warp_tf = Warp(
                indices=list(range(train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
            )
        super().__init__(
            train_x,
            train_y,
            GaussianLikelihood(
                noise_constraint=Interval(MIN_INFERRED_NOISE_LEVEL, noise_val * 2)
                # transform=None,
                # initial_value=1.0)
            ).double(),
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else warp_tf,
        )

        if fix_noise:
            self.likelihood.noise_covar.raw_noise.requires_grad = False
            self.likelihood.noise = noise_val

        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.noise_val = noise_val
        self.fix_noise = fix_noise
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self, train_x, train_y):
        return GP(
            train_x,
            train_y,
            self.noise_val,
            self.fix_noise,
            self.kernel,
            self.standardize,
            self.normalize,
            self.zero_mean,
            self.input_warping,
        )


class FixedGP(FixedNoiseGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_val: float = 783.6,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ):
        super().__init__(
            train_x,
            train_y,
            torch.full_like(train_y, noise_val),
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.noise_val = noise_val
        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping
        # self.covar_module2 = LinearKernel()
        # self.covar_module = self.covar_module1 + self.covar_module2
        # self.to(train_x)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self, train_x, train_y):
        return FixedGP(
            train_x,
            train_y,
            self.noise_val,
            self.kernel,
            self.standardize,
            self.normalize,
            self.zero_mean,
            self.input_warping,
        )


class HeteroskedasticGP(HeteroskedasticSingleTaskGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_val: float = 783.6,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ):
        super().__init__(
            train_x,
            train_y,
            # noise_val + 0.01 * torch.rand_like(train_y),
            # torch.normal(noise_val, , size=train_y.size()),
            torch.full_like(train_y, noise_val),
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.noise_val = noise_val
        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self, train_x, train_y):
        return HeteroskedasticGP(
            train_x,
            train_y,
            self.noise_val,
            self.kernel,
            self.standardize,
            self.normalize,
            self.zero_mean,
            self.input_warping,
        )


class SaasGP(SaasFullyBayesianSingleTaskGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_val: float = 0.06,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ):
        super().__init__(
            train_x,
            train_y,
            torch.full_like(train_y, noise_val),
            outcome_transform=Standardize(train_y.shape[-1] if standardize else None),
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.noise_val = noise_val
        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self, train_x, train_y):
        return SaasGP(
            train_x,
            train_y,
            self.noise_val,
            self.kernel,
            self.standardize,
            self.normalize,
            self.zero_mean,
            self.input_warping,
        )


class CustomHeteroskedasticGP(SingleTaskGP):
    r"""A single-task exact GP model using a heteroskeastic noise model.

    This model internally wraps another GP (a SingleTaskGP) to model the
    observation noise. This allows the likelihood to make out-of-sample
    predictions for the observation noise levels.
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_val: float,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ) -> None:
        r"""A single-task exact GP model using a heteroskedastic noise model.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
                Note that the noise model internally log-transforms the
                variances, which will happen after this transform is applied.
            input_transform: An input transfrom that is applied in the model's
                forward pass.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> se = torch.norm(train_X, dim=1, keepdim=True)
            >>> train_Yvar = 0.1 + se * torch.rand_like(train_Y)
            >>> model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
        """
        train_yvar = torch.full_like(train_y, noise_val)

        outcome_transform = Standardize(train_y.shape[-1]) if standardize else None
        input_transform = Normalize(train_x.shape[-1]) if normalize else None
        if outcome_transform is not None:
            train_y, train_yvar = outcome_transform(train_y, train_yvar)

        self._validate_tensor_args(X=train_x, Y=train_y, Yvar=train_yvar)
        validate_input_scaling(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        self._set_dimensions(train_X=train_x, train_Y=train_y)

        noise_model = SingleTaskGP(
            train_X=train_x,
            train_Y=torch.log(train_yvar),
            input_transform=input_transform,  # NOTE: potential bug here
        )
        mll = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        fit_gpytorch_model(mll)

        heteroskedastic_noise = HeteroskedasticNoise(
            noise_model=noise_model,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL, transform=torch.exp, inv_transform=None
            ),
        )

        likelihood = _GaussianLikelihoodBase(heteroskedastic_noise)
        super().__init__(
            train_x,
            train_y,
            likelihood=likelihood.double(),
            # outcome_transform=Log(),
            input_transform=input_transform,
        )

        self.noise_val = noise_val
        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping

        self.register_added_loss_term("noise_added_loss")
        self.update_added_loss_term(
            "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
        )
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        # self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def reinit(self, train_x, train_y):
        return CustomHeteroskedasticGP(
            train_x,
            train_y,
            self.noise_val,
            self.kernel,
            self.standardize,
            self.normalize,
            self.zero_mean,
            self.input_warping,
        )


# class RandomForest:
#     def __init__(self):


# class CustomHeteroskedasticSingleTaskGP(SingleTaskGP):
#     r"""A single-task exact GP model using a heteroskeastic noise model.
#
#     This model internally wraps another GP (a SingleTaskGP) to model the
#     observation noise. This allows the likelihood to make out-of-sample
#     predictions for the observation noise levels.
#     """
#
#     def __init__(
#             self,
#             train_x: Tensor,
#             train_y: Tensor,
#             train_yvar: Tensor,
#             outcome_transform: Optional[OutcomeTransform] = None,
#             input_transform: Optional[InputTransform] = None,
#     ) -> None:
#         r"""A single-task exact GP model using a heteroskedastic noise model.
#
#         Args:
#             train_x: A `batch_shape x n x d` tensor of training features.
#             train_y: A `batch_shape x n x m` tensor of training observations.
#             train_yvar: A `batch_shape x n x m` tensor of observed measurement
#                 noise.
#             outcome_transform: An outcome transform that is applied to the
#                 training data during instantiation and to the posterior during
#                 inference (that is, the `Posterior` obtained by calling
#                 `.posterior` on the model will be on the original scale).
#                 Note that the noise model internally log-transforms the
#                 variances, which will happen after this transform is applied.
#             input_transform: An input transfrom that is applied in the model's
#                 forward pass.
#
#         Example:
#             >>> train_x = torch.rand(20, 2)
#             >>> train_y = torch.sin(train_x).sum(dim=1, keepdim=True)
#             >>> se = torch.norm(train_x, dim=1, keepdim=True)
#             >>> train_yvar = 0.1 + se * torch.rand_like(train_y)
#             >>> model = HeteroskedasticSingleTaskGP(train_x, train_y, train_yvar)
#         """
#         if outcome_transform is not None:
#             train_y, train_yvar = outcome_transform(train_y, train_yvar)
#         self._validate_tensor_args(X=train_x, Y=train_y, Yvar=train_yvar)
#         validate_input_scaling(train_x=train_x, train_y=train_y, train_yvar=train_yvar)
#         self._set_dimensions(train_x=train_x, train_y=train_y)
#         noise_likelihood = GaussianLikelihood(
#             noise_prior=SmoothedBoxPrior(-3, 5, 0.5, transform=torch.log),
#             batch_shape=self._aug_batch_shape,
#             noise_constraint=GreaterThan(
#                 MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=1.0
#             ),
#         )
#         noise_model = SingleTaskGP(
#             train_x=train_x,
#             train_y=train_yvar,
#             likelihood=noise_likelihood,
#             outcome_transform=Log(),
#             input_transform=input_transform,
#         )
#
#         heteroskedastic_noise = HeteroskedasticNoise(
#             noise_model=noise_model,
#             noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL, transform=torch.exp, inv_transform=torch.log),
#         )
#
#         likelihood = _GaussianLikelihoodBase(heteroskedastic_noise)
#         super().__init__(
#             train_x=train_x,
#             train_y=train_y,
#             likelihood=likelihood,
#             input_transform=input_transform,
#         )
#         self.register_added_loss_term("noise_added_loss")
#         self.update_added_loss_term(
#             "noise_added_loss", NoiseModelAddedLossTerm(noise_model)
#         )
#         if outcome_transform is not None:
#             self.outcome_transform = outcome_transform
#         self.to(train_x)
