"""
Module containing GPyTorch metrics defined here:

https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/metrics/metrics.py

Not yet included in the latest release.
TODO: Once new release of GPyTorch becomes available, remove this module.
"""

from math import pi

import torch
from gpytorch.distributions import (
    MultitaskMultivariateNormal,
    MultivariateNormal,
)

pi = torch.tensor(pi)


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    return -pred_dist.log_prob(test_y.squeeze()) / test_y.size(0)


def mean_standardized_log_loss(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    f_mean = pred_dist.mean.squeeze()
    f_var = pred_dist.variance.squeeze()
    return (
        0.5
        * (
            torch.log(2 * pi * f_var)
            + torch.square(test_y.squeeze() - f_mean) / (2 * f_var)
        ).mean()
    )


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    return -pred_dist.log_prob(test_y) / test_y.shape[combine_dim]


def quantile_coverage_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    quantile: float = 95.0,
):
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean.squeeze() - deviation * pred_dist.stddev.squeeze()
    upper = pred_dist.mean.squeeze() + deviation * pred_dist.stddev.squeeze()
    n_samples_within_bounds = (
        (test_y.squeeze() > lower) * (test_y.squeeze() < upper)
    ).sum()
    fraction = n_samples_within_bounds / test_y.size(0)
    return torch.abs(fraction - quantile / 100)
