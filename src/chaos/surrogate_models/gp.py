# from __future__ import annotations
import gpytorch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL, SingleTaskGP
from botorch.models.transforms.input import Normalize, Warp
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.priors import GammaPrior, LogNormalPrior
from torch import Tensor


from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from abc import ABC, abstractmethod

from typing import Optional
import gpytorch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL, SingleTaskGP
from botorch.models.transforms.input import InputTransform, Normalize, Warp
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.means.mean import Mean
from gpytorch.module import Module
from gpytorch.priors import GammaPrior, LogNormalPrior
from torch import Tensor
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, List
import torch
from botorch.settings import debug
import numpy as np
import chaos
from gpytorch.kernels import AdditiveKernel


class MyAdditiveKernel(AdditiveKernel):
    def __init__(self, kernel_config: List):
        kernels = []
        for k_config in kernel_config:
            class_path = k_config["class_path"]
            init_args = k_config.get("init_args", {})
            KernelClass = eval(class_path)
            kernel = KernelClass(**init_args)
            kernels.append(kernel)
        super().__init__(*kernels)


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class SimpleGP(SurrogateModel, SingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 2.0,
        initial_lengthscale_val: float = 0.5,
    ) -> None:
        super().__init__(
            train_x,
            train_y,
            likelihood,
            covar_module,
            mean_module,
            Standardize(train_y.shape[-1]) if standardize else None,
            Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        # self.likelihood.initialize(noise=initial_noise_val)

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        # Check existing parameters in the model
        existing_parameters = {name for name, _ in self.named_parameters()}

        # Only initialize parameters that exist in the model and have a non-None value
        hypers_to_use = {
            k: torch.tensor(v)
            for k, v in hypers.items()
            if k in existing_parameters and v is not None
        }

        # Apply the initialization
        self.initialize(**hypers_to_use)

        self.standardize = standardize
        self.normalize = normalize
        self.initial_noise_val = initial_noise_val
        self.noise_constraint = noise_constraint

    def fit(self, train_X, train_Y):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        try:
            with debug(True):
                fit_gpytorch_mll(mll)

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(self, x, observation_noise=False, return_var=True):
        self.eval()  # set the model to evaluation mode
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)
        return (posterior.mean, posterior.variance) if return_var else posterior.mean


from botorch.models import FixedNoiseGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


class MyFixedNoiseGP(SurrogateModel, FixedNoiseGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor],
        train_y: Union[np.ndarray, torch.Tensor],
        train_yvar: Union[np.ndarray, torch.Tensor],
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
    ) -> None:
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_yvar, learn_additional_noise=False
        )
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.standardize = standardize
        self.normalize = normalize

    def fit(self, x_train, y_train):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        try:
            with debug(True):
                fit_gpytorch_mll(mll)

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(self, x, observation_noise=False, return_var=True):
        self.eval()  # set the model to evaluation mode
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)
        return (posterior.mean, posterior.variance) if return_var else posterior.mean


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
        self._set_dimensions(train_X=train_x, train_Y=train_y)
        warp_tf = self._get_warp_tf(train_x, input_warping)
        kernel = self._get_kernel(train_x, kernel)
        likelihood = self._get_likelihood()
        super().__init__(
            train_x,
            train_y,
            likelihood=likelihood.double(),
            covar_module=ScaleKernel(base_kernel=kernel),
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else warp_tf,
        )

        self.mean_module = ZeroMean() if zero_mean else ConstantMean()
        self.noise_val = noise_val
        self.fix_noise = fix_noise
        self.kernel = kernel
        self.standardize = standardize
        self.normalize = normalize
        self.zero_mean = zero_mean
        self.input_warping = input_warping

        if self.fix_noise:
            self._fix_noise()

    def _get_warp_tf(self, train_x: Tensor, input_warping: bool):
        if input_warping:
            return Warp(
                indices=list(range(train_x.shape[-1])),
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
            )

    def _get_kernel(self, train_x: Tensor, kernel: Kernel):
        return kernel or MaternKernel(
            nu=2.5,
            ard_num_dims=train_x.shape[-1],
            batch_shape=self._aug_batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

    def _get_likelihood(self):
        noise_prior = GammaPrior(1.1, 1)  # 0.05
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        return GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

    def _fix_noise(self):
        self.covar_module.raw_outputscale.data.fill_(1.0)
        self.likelihood.noise_covar.raw_noise.data.fill_(0.1)
        output_scale_prior = GammaPrior(2.0, 1)
        likelihood_noise_prior = GammaPrior(2.0, 1)

        self.covar_module.register_prior(
            "outputscale_prior", output_scale_prior, "raw_outputscale"
        )
        self.likelihood.noise_covar.register_prior(
            "noise_prior", likelihood_noise_prior, "raw_noise"
        )

        self.covar_module.raw_outputscale.constraint = gpytorch.constraints.Interval(
            -10.0, 10.0
        )
        self.likelihood.noise_covar.raw_noise.constraint = (
            gpytorch.constraints.Interval(-5.0, 5.0)
        )

    def initialize_mll(self, state_dict=None):
        """
        Initialise model and loss function.

        Args:
            state_dict: current state dict used to speed up fitting
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        # load state dict if it is passed
        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)
        return mll

    def fit(self, train_x, train_y, state_dict=None):
        """
        Initialize and fit the GP model with new training data.

        Args:
            train_x: New training inputs.
            train_y: New training outputs.
            state_dict: current state dict used to speed up fitting
        """
        self.set_train_data(inputs=train_x, targets=train_y.view(-1), strict=False)
        mll = self.initialize_mll(state_dict)
        self.fit_with_retries(mll)

    def fit_with_retries(self, mll):
        max_retries = 20
        for i in range(max_retries):
            try:
                self.fit_mll(mll)
                break
            except RuntimeError as e:
                if i == max_retries - 1:  # If we're on our last try
                    raise e
                print(f"Encountered error in optimization, retrying... (#{i + 1})")

    def fit_mll(self, mll):
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            fit_gpytorch_mll(mll, max_retries=50)

    def fit(self, x_train, y_train):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        try:
            with debug(True):
                fit_gpytorch_mll(mll)

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(self, x, observation_noise=False, return_var=True):
        self.eval()  # set the model to evaluation mode
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)
        return (posterior.mean, posterior.variance) if return_var else posterior.mean
