def filter_data(data, base_filters, additional_filters=None):
    """
    Filter the provided DataFrame based on the given filters.

    Parameters:
    - data: DataFrame containing the data
    - filters: dictionary containing filters to apply to the data

    Returns:
    - Filtered DataFrame
    """
    filtered_data = data.copy()

    if additional_filters:
        base_filters.update(additional_filters)

    for key, value in base_filters.items():
        if isinstance(value, list):
            filtered_data = filtered_data[filtered_data[key].isin(value)]
        else:
            filtered_data = filtered_data[filtered_data[key] == value]
    return filtered_data


def get_best_config(grouped_data, metric, config_columns):
    """
    Extract the best configuration from grouped data based on a given metric.

    Parameters:
    - grouped_data: DataFrame containing the grouped data
    - metric: metric to analyze
    - config_columns: list of columns to consider for the best configuration extraction

    Returns:
    - Dictionary containing the best configuration for the given metric
    """
    best_config_row = grouped_data.loc[grouped_data[metric].idxmax()]
    best_config = {"params": {}}
    for col in config_columns:
        best_config["params"][col] = best_config_row[col]
    best_config["value"] = best_config_row[metric]
    return best_config


def extract_best_config_for_analysis(
    data, analysis_column, metrics_to_analyze, config_columns, additional_filters=None
):
    """
    Extract the best configuration for a given analysis column based on the provided metrics.

    Parameters:
    - data: DataFrame containing the data
    - analysis_column: string indicating the column to analyze (e.g., 'representation', 'kernel')
    - metrics_to_analyze: list of metrics to analyze
    - config_columns: list of columns to consider for the best configuration extraction
    - additional_filters: dictionary containing additional filters to apply to the data

    Returns:
    - Dictionary containing the best configurations for each metric
    """

    # Apply filters to the data
    if additional_filters:
        filtered_data = filter_data(data, additional_filters)
    else:
        filtered_data = data.copy()

    # Group by the analysis_column and config_columns
    grouped_data = filtered_data.groupby([analysis_column] + config_columns)

    best_configurations = {}
    for metric in metrics_to_analyze:
        aggregated_data = grouped_data[metric].mean().reset_index()
        best_config = get_best_config(aggregated_data, metric, config_columns)
        best_configurations[metric] = best_config

    return best_configurations


import wandb
import pandas as pd


def process_run(args):
    run_id, project_name = args
    api = wandb.Api()
    run = api.run(f"{project_name}/{run_id}")
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    name = run.name
    history = run.scan_history()

    df = pd.DataFrame(history)

    # Check if 'epoch' column exists
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].fillna(method="ffill")
        df = (
            df.groupby("epoch")
            .agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)
            .reset_index()
        )
    else:
        # Handle the failed runs
        print(f"Run {run.name} does not have an 'epoch' column.")
        return None, summary, config, name, run_id

    df["run_id"] = run.id
    df["name"] = name
    return df, summary, config, name, run_id


def format_results_data(data):
    data = data.rename(
        columns={
            "data.featurizer.init_args.representation": "representation",
            "surrogate_model.init_args.covar_module.init_args.base_kernel.class_path": "kernel",
            "data_selection.method": "init",
            "surrogate_model.init_args.normalize": "normalize",
            "data.featurizer.init_args.bond_radius": "bond_radius",
            "data.featurizer.init_args.nBits": "n_bits",
            "model.acquisition_class": "acquisition",
        }
    )
    data["kernel"].replace(
        {
            "gpytorch.kernels.linear_kernel.LinearKernel": "Linear",
            "gpytorch.kernels.matern_kernel.MaternKernel": "Matern",
            "chaos.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel": "Tanimoto",
            "gpytorch.kernels.rq_kernel.RQKernel": "RQ",
        },
        inplace=True,
    )
    return data


def convert_to_percentage(row, metric, n_data=720):
    if "top_" in metric:
        total = int(metric.split("_")[1])
    elif "quantile" in metric:
        quantile = int(metric.split("_")[1])
        total = int(n_data * (100 - quantile) / 100)
    else:
        return row[metric]  # For metrics that don't need conversion

    return row[metric] / total


import torch

from chaos.gprotorch.metrics import negative_log_predictive_density
from sklearn.metrics import r2_score, mean_absolute_error


def compute_top5_metrics(model, data, subset):
    # Assuming that model.heldout_x and model.heldout_y are your test features and labels
    heldout_x = data.heldout_x
    heldout_y = data.heldout_y

    # Sort data to get top and bottom 10%
    sorted_indices = torch.argsort(heldout_y.squeeze())
    top_5_percent = sorted_indices[-int(len(sorted_indices) * 0.05) :]
    bottom_5_percent = sorted_indices[: int(len(sorted_indices) * 0.05)]

    if subset == "top_5":
        x = heldout_x[top_5_percent]
        y = heldout_y[top_5_percent]
    elif subset == "bottom_5":
        x = heldout_x[bottom_5_percent]
        y = heldout_y[bottom_5_percent]
    elif subset == "all":
        x = heldout_x
        y = heldout_y
    else:
        raise ValueError("Invalid subset specified")

    # Get the posterior predictive distribution for the subset
    pred_dist = model.posterior(x)

    # Compute NLPD
    nlpd = negative_log_predictive_density(pred_dist, y).item()

    # Get predictive mean to compute other metrics
    pred_mean = pred_dist.mean.detach().cpu().numpy()

    # Compute R2 and MAE
    r2 = r2_score(y.cpu().numpy(), pred_mean)
    mae = mean_absolute_error(y.cpu().numpy(), pred_mean)

    return nlpd, r2, mae
