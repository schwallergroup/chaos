from collections.abc import MutableMapping
import importlib


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_to_nested_dict(flat_dict):
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        if value == "True" or value == "False":
            d[keys[-1]] = value == "True"
    return nested_dict


import copy


def instantiate_class(input_dict, *args, **kwargs):
    class_path = input_dict["class_path"]
    init_args = copy.deepcopy(input_dict.get("init_args", {}))
    init_args.update(kwargs)  # merge extra arguments into init_args

    # Iterate over init_args, checking if any values are themselves classes to be instantiated
    for arg_name, arg_value in init_args.items():
        if isinstance(arg_value, dict) and "class_path" in arg_value:
            init_args[arg_name] = instantiate_class(arg_value)

    module_name, class_name = class_path.rsplit(".", 1)
    MyClass = getattr(importlib.import_module(module_name), class_name)
    instance = MyClass(*args, **init_args)  # passing extra arguments to the class

    return instance


## Statistical analysis

from scipy.stats import mannwhitneyu, ttest_ind


def compute_statistical_influence(data, hyperparameter, metric):
    """
    Compute statistical significance for differences in distributions
    of a performance metric across different levels of a hyperparameter.

    Returns a dictionary with pairwise comparisons and p-values.
    """
    levels = data[hyperparameter].unique()
    results = {}

    for i, level_i in enumerate(levels):
        for j, level_j in enumerate(levels):
            if i < j:
                group_i = data[data[hyperparameter] == level_i][metric].dropna().values
                group_j = data[data[hyperparameter] == level_j][metric].dropna().values

                # Mann-Whitney U Test
                _, p_value_mwu = mannwhitneyu(group_i, group_j, alternative="two-sided")

                # Welch's T-test
                _, p_value_ttest = ttest_ind(group_i, group_j, equal_var=False)

                results[f"{level_i} vs {level_j}"] = {
                    "p_value_mwu": p_value_mwu,
                    "p_value_ttest": p_value_ttest,
                }

    return results


from itertools import product


def analyze_parameter_influence(
    df, parameters, metrics_to_analyze, compute_statistical_influence
):
    """
    Analyze the influence of parameters on given metrics.

    Parameters:
    - df: DataFrame containing the data.
    - parameters: List of parameter names to analyze.
    - metrics_to_analyze: List of metrics to analyze.
    - compute_statistical_influence: Function to compute the statistical influence.

    Returns:
    - Dictionary containing the results for each metric.
    """
    # Placeholder for results
    all_results = {}

    # Extract unique values for parameters
    unique_values = {param: df[param].unique().tolist() for param in parameters}

    # Loop over the metrics
    for metric in metrics_to_analyze:
        results = []

        # Looping over all combinations of parameters
        for param_values in product(
            *[unique_values[param] for param in parameters[:-1]]
        ):
            subset_conditions = {
                param: value for param, value in zip(parameters, param_values)
            }
            subset_df = df
            for param, value in subset_conditions.items():
                subset_df = subset_df[subset_df[param] == value]

            # Computing statistics for the influence of the last parameter
            influence = compute_statistical_influence(subset_df, parameters[-1], metric)
            result = subset_conditions
            result[f"{parameters[-1]}_influence"] = influence
            results.append(result)

        all_results[metric] = results

    return all_results


def extract_significant_results(all_results, hyperparameter, threshold=0.05):
    significant_results = {}

    for metric, metric_data in all_results.items():
        significant_metric_data = []

        for data in metric_data:
            significant_influence = {}
            for comparison, stats in data[f"{hyperparameter}_influence"].items():
                if any(p < threshold for p in stats.values()):
                    significant_influence[comparison] = stats

            if significant_influence:
                significant_entry = data.copy()
                significant_entry[f"{hyperparameter}_influence"] = significant_influence
                significant_metric_data.append(significant_entry)

        if significant_metric_data:
            significant_results[metric] = significant_metric_data

    return significant_results


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_metric_performance(
    df, hyperparam_to_analyze, hyperparameter_display_name, metrics_to_analyze
):
    # Set up the figure and axis
    fig, axes = plt.subplots(nrows=len(metrics_to_analyze), ncols=1, figsize=(10, 15))

    # Iterate through metrics and plot
    for idx, metric in enumerate(metrics_to_analyze):
        ax = axes[idx]
        sns.barplot(
            x=hyperparam_to_analyze,
            y=metric,
            data=df,
            ax=ax,
            errorbar="sd",
            estimator=np.mean,
            palette="viridis",
        )
        ax.set_title(f"Mean {metric} by {hyperparameter_display_name}")
        ax.set_xlabel("Kernel")
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_boxplot_metrics_comparison(
    df, param1, param2, metrics_to_analyze, barplot=False, palette=None
):
    # Setting the style and size
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=len(metrics_to_analyze), ncols=1, figsize=(10, 15))

    for idx, metric in enumerate(metrics_to_analyze):
        ax = axes[idx]
        # Creating the plot
        if barplot:
            sns.barplot(x=param1, y=metric, hue=param2, data=df, ax=ax, palette=palette)
        else:
            sns.boxplot(x=param1, y=metric, hue=param2, data=df, ax=ax, palette=palette)

        # Customize the plot
        ax.set_title(f"Performance Distribution for {metric}")
        ax.set_xlabel(param1)
        ax.set_ylabel(metric)
        ax.legend(title=param2)

    plt.tight_layout()
    plt.show()


import wandb
from tqdm import tqdm
import pandas as pd


def extract_data_from_wandb(project_name="liac/additives-rebuttal", filters={}):
    # sweeps = ['a37f8lm6', 'wulc1gmr', 'j1t7obm9', 'rx4tuw5p', 'bxxjq4ul', '6h91vmqu', 'ickm2ent']#, 'bxxjq4ul', '6h91vmqu']
    # Fetch runs data
    api = wandb.Api()
    runs = api.runs(project_name, filters=filters)  # {"sweep": {'$in': sweeps}})

    summary_list = []
    history_list = []
    config_list = []
    name_list = []

    for run in tqdm(runs):
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_list.append(run.name)

    # Convert to DataFrame
    data = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
        }
    )

    # Flatten the dictionaries in the 'summary' and 'config' columns
    summary_df = pd.json_normalize(data["summary"])
    config_df = pd.json_normalize(data["config"])

    # Concatenate all the dataframes together
    data = pd.concat(
        [data.drop(columns=["summary", "config"]), summary_df, config_df], axis=1
    )
    return data
