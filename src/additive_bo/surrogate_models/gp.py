# from __future__ import annotations
from inspect import signature
from typing import Any, List, Optional, Union

import botorch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
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
from botorch.sampling import IIDNormalSampler

# from torch import Tensor
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan, Interval, LessThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import LaplaceLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor

import wandb
from additive_bo.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel import (
    TanimotoKernel,
)


class StudentTLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, num_samples, scale=1.0, **kwargs):
        super(StudentTLikelihood, self).__init__(**kwargs)
        self.register_parameter(
            name="scale", parameter=torch.nn.Parameter(scale * torch.ones(1))
        )
        self.num_samples = num_samples

    def forward(self, function_samples, target):
        noise = target - function_samples
        noise_scale = self.scale.squeeze(0)
        log_probs = (
            -0.5 * (self.num_samples + 1) * torch.log(1 + (noise / noise_scale) ** 2)
        )
        return log_probs.sum()


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
            # StudentTLikelihood(500
            GaussianLikelihood(
                noise_prior=GammaPrior(2.0, 0.15)
                # noise_constraint=Interval(MIN_INFERRED_NOISE_LEVEL, noise_val * 2)
                # transform=None,
                # initial_value=1.0
                # noise_covar=torch.tensor([0.15], requires_grad=False)
            ).double(),
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else warp_tf,
        )
        # self.likelihood.noise_covar.register_constraint("raw_noise",
        #                                                   GreaterThan(1e-5))
        if fix_noise:
            self.likelihood.noise_covar.raw_noise.requires_grad = False
            #     self.likelihood.noise_constraint=Interval(MIN_INFERRED_NOISE_LEVEL, noise_val * 2)
            self.likelihood.noise = 0.0001  # noise_val

        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        # nugget_prior = SmoothedBoxPrior(-7, 0, sigma=0.1)
        # nugget_constraint = 'greater_than(-7)'
        # nugget = torch.tensor(1e-5)

        # Add the nugget term to the kernel
        # kernel = kernel + WhiteNoiseKernel(prior=nugget_prior, constraint=nugget_constraint)
        # kernel = gpytorch.kernels.AdditiveKernel(kernel, gpytorch.kernels.ConstantKernel(nugget))

        # concentration = 3
        # rate = 6

        # # Create the Gamma prior object
        # lengthscale_prior = GammaPrior(concentration=concentration, rate=rate)

        # Create the RBF kernel with the Gamma prior on lengthscale
        # kernel = RBFKernel(lengthscale_prior=lengthscale_prior)
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.noise_val = noise_val
        self.fix_noise = fix_noise
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping

    # def forward(self, x):
    #     mean_x = self.mean_module(x)
    #     covar_x = self.covar_module(x)
    #     return MultivariateNormal(mean_x, covar_x)

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
        noise_val: Union[float, Tensor] = 783.6,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ):
        train_yvar = (
            torch.full_like(train_y, noise_val**2)
            if isinstance(noise_val, float)
            else noise_val
        )

        super().__init__(
            train_x,
            train_y,
            train_yvar,  # torch.full_like(train_y, noise_val**2),
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
        noise_val: Union[float, Tensor] = 783.6,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        input_warping: bool = False,
    ):
        train_yvar = (
            torch.full_like(train_y, noise_val)
            if isinstance(noise_val, float)
            else noise_val
        )

        super().__init__(
            train_x,
            train_y,
            # noise_val + 0.01 * torch.rand_like(train_y),
            # torch.normal(noise_val, , size=train_y.size()),
            train_yvar,
            # torch.full_like(train_y, noise_val**2),
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
        noise_val: Union[float, Tensor],
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
        train_yvar = (
            torch.full_like(train_y, noise_val)
            if isinstance(noise_val, float)
            else noise_val
        )

        outcome_transform = Standardize(train_y.shape[-1]) if standardize else None
        input_transform = Normalize(train_x.shape[-1]) if normalize else None
        if outcome_transform is not None:
            train_y, train_yvar = outcome_transform(train_y, train_yvar)

        self._validate_tensor_args(X=train_x, Y=train_y, Yvar=train_yvar)
        # validate_input_scaling(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        self._set_dimensions(train_X=train_x, train_Y=train_y)

        noise_model = SingleTaskGP(
            train_X=train_x,
            train_Y=torch.log(train_yvar),
            # likelihood=GaussianLikelihood(),
            # covar_module=ScaleKernel(base_kernel=kernel),
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
            outcome_transform=outcome_transform,
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

    # def forward(self, x):
    #     mean_x = self.mean_module(x)
    #     covar_x = self.covar_module(x)
    #     return MultivariateNormal(mean_x, covar_x)

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


# class MostLikelyHeteroskedasticGP(HeteroskedasticSingleTaskGP):
#     def __init__(self,
#                 train_x: Tensor,
#                 train_y: Tensor,
#                 noise_val: float = 1e-4,
#                 kernel: Kernel = None,
#                 standardize: bool = True,
#                 normalize: bool = False,
#                 zero_mean: bool = False,
#                 var_estimate: str = 'mcr'):

#         homo_model = SingleTaskGP(train_X=train_x, train_Y=train_y,
#                                   covar_module=ScaleKernel(base_kernel=kernel), \
#                                   input_transform=Normalize(train_x.shape[-1]) if normalize else None,
#                                   outcome_transform=Standardize(train_y.shape[-1]) if standardize else None)
#         homo_model.likelihood.noise_covar.register_constraint("raw_noise",
#                                                           GreaterThan(1e-5))

#         homo_mll = ExactMarginalLogLikelihood(homo_model.likelihood, homo_model)
#         fit_gpytorch_model(homo_mll)

#         homo_mll.eval()
#         # test on the training points
#         # call it X_test just for ease of use
#         test_x = train_x.clone()

#         # homo_mll.eval()
#         with torch.no_grad():
#             # homo_posterior = homo_mll.model.posterior(test_x)
#             # homo_predictive_posterior = homo_mll.model.posterior(test_x,
#             #                                                     observation_noise=True)
#             if var_estimate == 'mcr':
#                 # watch broadcasting here
#                 observed_var = torch.tensor(
#                                     np.power(homo_mll.model.posterior(test_x).mean.numpy().reshape(-1,) - train_y.numpy(), 2),
#                                     dtype=torch.float
#                 )
#             else:
#                 sampler = IIDNormalSampler(sample_shape=1000)
#                 predictive_posterior = homo_mll.model.posterior(test_x, observation_noise=True)
#                 samples = sampler(predictive_posterior)
#                 observed_var = 0.5 * ((samples - train_y.reshape(-1,1))**2).mean(dim=0)

#         # print('OBSERVED VARIANCE', observed_var)
#         super().__init__(train_X=train_x,
#                         train_Y=train_y,
#                         train_Yvar=observed_var,
#                         outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
#                         input_transform=Normalize(train_x.shape[-1]) if normalize else None)

#         self.mean_module = ZeroMean() if zero_mean else ConstantMean()
#         self.covar_module = ScaleKernel(base_kernel=kernel)
#         self.noise_val = noise_val
#         self.kernel = kernel
#         self.standardize = standardize
#         self.normalize = normalize
#         self.zero_mean = zero_mean
#         self.var_estimate = var_estimate


#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return MultivariateNormal(mean_x, covar_x)

#     def reinit(self, train_x, train_y):
#         return MostLikelyHeteroskedasticGP(
#             train_x,
#             train_y,
#             self.noise_val,
#             self.kernel,
#             self.standardize,
#             self.normalize,
#             self.zero_mean,
#             self.var_estimate,
#         )

from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log
from botorch.posteriors.gpytorch import GPyTorchPosterior


class MostLikelyHeteroskedasticGP(CustomHeteroskedasticGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_val: float = 1e-4,
        kernel: Kernel = None,
        standardize: bool = True,
        normalize: bool = False,
        zero_mean: bool = False,
        var_estimate: str = "paper",
    ):

        # wandb.init(project='additives-plate-1')

        # Chain the transforms together
        # transform = ChainedOutcomeTransform([log_transform, standardize_transform])

        # print(log_transform(train_y), train_y)
        tf1 = Log()
        # tf2 = Standardize(1)
        # tf = ChainedOutcomeTransform(tf1=tf1, tf2=tf2)
        homo_model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=ScaleKernel(base_kernel=kernel),
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
        )
        #   tf1)
        # Standardize(train_y.shape[-1]) if standardize else None)
        homo_model.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(1e-5)
        )

        # print(torch.max(train_x), torch.min(train_x))

        homo_mll = ExactMarginalLogLikelihood(homo_model.likelihood, homo_model)
        fit_gpytorch_model(homo_mll)

        homo_mll.eval()
        # test on the training points
        # call it X_test just for ease of use
        test_x = train_x.clone()
        test_y = train_y.clone()

        # homo_mll.eval()
        with torch.no_grad():
            # homo_posterior = homo_mll.model.posterior(test_x)
            # homo_predictive_posterior = homo_mll.model.posterior(test_x,
            #                                                     observation_noise=True)
            if var_estimate == "mcr":
                # watch broadcasting here
                observed_var = torch.tensor(
                    np.power(
                        homo_mll.model.posterior(test_x)
                        .mean.numpy()
                        .reshape(
                            -1,
                        )
                        - train_y.numpy(),
                        2,
                    ),
                    dtype=torch.float,
                )
            else:
                sampler = IIDNormalSampler(sample_shape=1000)
                predictive_posterior = homo_mll.model.posterior(
                    test_x, observation_noise=True
                )
                samples = sampler(predictive_posterior)
                observed_var = 0.5 * ((samples - train_y.reshape(-1, 1)) ** 2).mean(
                    dim=0
                )
                # print('PREDICTED MEAN', predictive_posterior.mean)

                mean_train = predictive_posterior.mean
                _, axes = plt.subplots(1, 1, figsize=(6, 4))

                axes.scatter(
                    train_y.squeeze(),
                    mean_train.squeeze(),
                    color="green",  # self.trainer.datamodule.
                )
                axes.ticklabel_format(useOffset=False)

                plt.xlabel("Actual")
                plt.ylabel("Predicted")

                wandb.log({"pred-vs-actual-green": [wandb.Image(axes)]})
                plt.close("all")
                plt.clf()
                plt.cla()

        # print('OBSERVED VARIANCE', observed_var)
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            noise_val=observed_var,
            kernel=kernel,
            standardize=standardize,
            normalize=normalize,
            zero_mean=zero_mean,
            input_warping=False,
        )

        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=kernel)
        self.noise_val = noise_val
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.var_estimate = var_estimate

    # def forward(self, x):
    #     mean_x = self.mean_module(x)
    #     covar_x = self.covar_module(x)
    #     return MultivariateNormal(mean_x, covar_x)

    def reinit(self, train_x, train_y):
        return MostLikelyHeteroskedasticGP(
            train_x,
            train_y,
            self.noise_val,
            self.kernel,
            self.standardize,
            self.normalize,
            self.zero_mean,
            self.var_estimate,
        )


# class MostLikelyHeteroskedasticGP(CustomHeteroskedasticGP):
#     def __init__(self,
#                 train_x: Tensor,
#                 train_y: Tensor,
#                 noise_val: float = 1e-4,
#                 kernel: Kernel = None,
#                 standardize: bool = True,
#                 normalize: bool = False,
#                 zero_mean: bool = False,
#                 var_estimate: str = 'paper'):

#         wandb.init(project='additives-plate-1')

#         homo_model = SingleTaskGP(train_X=train_x, train_Y=train_y,
#                                   covar_module=ScaleKernel(base_kernel=kernel), \
#                                   input_transform=Normalize(train_x.shape[-1]) if normalize else None,
#                                   outcome_transform=Standardize(train_y.shape[-1]) if standardize else None)
#         homo_model.likelihood.noise_covar.register_constraint("raw_noise",
#                                                           GreaterThan(1e-5))

#         homo_mll = ExactMarginalLogLikelihood(homo_model.likelihood, homo_model)
#         fit_gpytorch_model(homo_mll)

#         homo_mll.eval()
#         # test on the training points
#         # call it X_test just for ease of use
#         test_x = train_x.clone()
#         test_y = train_y.clone()

#         # homo_mll.eval()
#         with torch.no_grad():
#             # homo_posterior = homo_mll.model.posterior(test_x)
#             # homo_predictive_posterior = homo_mll.model.posterior(test_x,
#             #                                                     observation_noise=True)
#             if var_estimate == 'mcr':
#                 # watch broadcasting here
#                 observed_var = torch.tensor(
#                                     np.power(homo_mll.model.posterior(test_x).mean.numpy().reshape(-1,) - train_y.numpy(), 2),
#                                     dtype=torch.float
#                 )
#             else:
#                 sampler = IIDNormalSampler(sample_shape=1000)
#                 predictive_posterior = homo_mll.model.posterior(test_x, observation_noise=True)
#                 samples = sampler(predictive_posterior)
#                 observed_var = 0.5 * ((samples - train_y.reshape(-1,1))**2).mean(dim=0)
#                 # print('PREDICTED MEAN', predictive_posterior.mean)

#                 mean_train = predictive_posterior.mean
#                 _, axes = plt.subplots(1, 1, figsize=(6, 4))

#                 axes.scatter(
#                     train_y.squeeze(),
#                     mean_train.squeeze(),
#                     color="green",  # self.trainer.datamodule.
#                 )
#                 axes.ticklabel_format(useOffset=False)

#                 plt.xlabel("Actual")
#                 plt.ylabel("Predicted")

#                 wandb.log({"pred-vs-actual-green": [wandb.Image(axes)]})
#                 plt.close("all")
#                 plt.clf()
#                 plt.cla()

#         # print('OBSERVED VARIANCE', observed_var)
#         super().__init__(train_x=train_x,
#                         train_y=train_y,
#                         noise_val=observed_var,
#                         kernel=kernel,
#                         standardize=standardize,
#                         normalize=normalize,
#                         zero_mean=zero_mean,
#                         input_warping=False,
#                         )

#         self.mean_module = ZeroMean() if zero_mean else ConstantMean()
#         self.covar_module = ScaleKernel(base_kernel=kernel)
#         self.noise_val = noise_val
#         self.kernel = kernel
#         self.standardize = standardize
#         self.normalize = normalize
#         self.zero_mean = zero_mean
#         self.var_estimate = var_estimate


#     # def forward(self, x):
#     #     mean_x = self.mean_module(x)
#     #     covar_x = self.covar_module(x)
#     #     return MultivariateNormal(mean_x, covar_x)

#     def reinit(self, train_x, train_y):
#         return MostLikelyHeteroskedasticGP(
#             train_x,
#             train_y,
#             self.noise_val,
#             self.kernel,
#             self.standardize,
#             self.normalize,
#             self.zero_mean,
#             self.var_estimate,
#         )

#     # def posterior(self,
#     #               X: Tensor,
#     #               output_indices: Optional[List[int]] = None,
#     #               observation_noise: Union[bool, Tensor] = False,
#     #               **kwargs: Any) -> GPyTorchPosterior:
#     #     if not True: #self.is_fitted:
#     #         return self.base_model.posterior(X,
#     #                                          output_indices,
#     #                                          observation_noise,
#     #                                          **kwargs)
#     #     else:
#     #         if not observation_noise:
#     #             #target_model
#     #             return self.posterior(X,
#     #                                                output_indices,
#     #                                                **kwargs)
#     #         else:
#     #             target_mvn = self.posterior(X,
#     #                                                      output_indices,
#     #                                                      **kwargs).mvn
#     #             noise_mvn = self.noise_model.posterior(X,
#     #                                                    output_indices,
#     #                                                    **kwargs).mvn
#     #             target_mean = target_mvn.mean
#     #             target_covar = target_mvn.covariance_matrix
#     #             noise_covar = torch.diag_embed(noise_mvn.mean.reshape(
#     #                 target_covar.shape[:-1]).max(
#     #                 torch.tensor(MIN_INFERRED_NOISE_LEVEL)))
#     #             if self._num_outputs > 1:
#     #                 mvn = MultitaskMultivariateNormal(
#     #                     target_mean, target_covar + noise_covar)
#     #             else:
#     #                 mvn = MultivariateNormal(
#     #                     target_mean, target_covar + noise_covar)

#     #             return GPyTorchPosterior(mvn=mvn)


# class RandomForest:
#     def __init__(self):
