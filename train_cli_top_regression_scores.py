import argparse
import logging
from botorch.models.model import Model as BaseModel
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from chaos.bo.module import BoModule
from chaos.data.module import BOAdditivesDataModule, BaseDataModule
from chaos.surrogate_models.gp import SimpleGP, GP
from chaos.initialization.initializers import BOInitializer

from chaos.utils import flatten
from chaos.utils import convert_to_nested_dict

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
from chaos.gprotorch.metrics import negative_log_predictive_density
from sklearn.metrics import r2_score, mean_absolute_error

import copy


def get_mol_or_rxn_smile(representation):
    if representation in ["rxnfp", "drfp"]:
        return "reaction_smiles"
    return "Additive_Smiles"


def get_distance_metric(kernel):
    # if representation in ["drfp", "fingerprints", "fragprints"]:
    if (
        kernel
        == "chaos.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel"
    ):
        return "jaccard"
    return "euclidean"


from argparse import Namespace


def convert_to_dict(ns):
    result = {"class_path": ns.class_path, "init_args": {}}
    for key, value in vars(ns).items():
        if key == "class_path":
            continue

        keys = key.split(".")
        d = result["init_args"]
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


class MyLightningCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--n_iters", type=int, default=100)
        parser.add_argument("--n_trials", type=int, default=20)

        parser.add_class_arguments(BOInitializer, "data_selection")
        # parser.add_subclass_arguments(BaseModel, "surrogate_model")
        # parser.add_subclass_arguments(BaseModel, "surrogate_model")
        parser.add_subclass_arguments(SimpleGP, "surrogate_model")
        # parser.add_subclass_arguments(GP, "surrogate_model")

        # parser.link_arguments(
        #     "data.representation",
        #     "data.featurize_column",
        #     apply_on="parse",
        #     compute_fn=get_mol_or_rxn_smile,
        # )
        parser.link_arguments(
            "data_selection", "data.initializer", apply_on="instantiate"
        )

        parser.link_arguments(
            "data_selection.n_clusters", "data.init_sample_size", apply_on="instantiate"
        )

        parser.link_arguments(
            "data.train_x", "surrogate_model.init_args.train_x", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.train_y", "surrogate_model.init_args.train_y", apply_on="instantiate"
        )
        parser.link_arguments("data", "model.data", apply_on="instantiate")
        parser.link_arguments("n_iters", "trainer.max_epochs")

        return super().add_arguments_to_parser(parser)

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        model_config = argparse.Namespace(**self.config["surrogate_model"])
        model_config_dict = convert_to_nested_dict(vars(model_config))
        # model_config_dict = convert_to_dict(model_config)  # vars(model_config)
        self.config.model.model_config = model_config_dict
        self.config_init = self.parser.instantiate_classes(self.config)
        # self.config_init.model.model_config = model_config_dict_copy
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    def before_instantiate_classes(self) -> None:
        return super().before_instantiate_classes()


class WandbSaveConfigCallback(SaveConfigCallback):
    def __init__(self, parser, config, overwrite=True):
        super().__init__(parser, config, overwrite=overwrite)
        self.parser = parser
        self.config = config
        wandb_config = flatten(dict(self.config))
        wandb.config.update(wandb_config)


import torch


def compute_metrics(model, subset):
    # Assuming that model.heldout_x and model.heldout_y are your test features and labels
    heldout_x = model.data.heldout_x
    heldout_y = model.data.heldout_y

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
    pred_dist = model.model.posterior(x)

    # Compute NLPD
    nlpd = negative_log_predictive_density(pred_dist, y).item()

    # Get predictive mean to compute other metrics
    pred_mean = pred_dist.mean.detach().cpu().numpy()

    # Compute R2 and MAE
    r2 = r2_score(y.cpu().numpy(), pred_mean)
    mae = mean_absolute_error(y.cpu().numpy(), pred_mean)

    return nlpd, r2, mae


def cli_main():
    cli = MyLightningCli(
        model_class=BoModule,
        datamodule_class=BaseDataModule,  # BOAdditivesDataModule,  #
        run=False,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "logger": WandbLogger(project="additives-rebuttal"),
            "log_every_n_steps": 1,
            "min_epochs": 1,
            "max_steps": -1,
            "accelerator": "cpu",
            "devices": 1,
            "num_sanity_val_steps": 0,
        },
    )

    cli.trainer.fit(cli.model)

    # Compute metrics for top 10%, bottom 10%, and all
    nlpd_top_5, r2_top_5, mae_top_5 = compute_metrics(cli.model, "top_5")
    nlpd_bottom_5, r2_bottom_5, mae_bottom_5 = compute_metrics(cli.model, "bottom_5")
    nlpd_all, r2_all, mae_all = compute_metrics(cli.model, "all")

    run_id = cli.trainer.logger.experiment.id
    wandb.init(id=run_id, resume="allow", project="additives-rebuttal")

    # Log metrics to wandb summary
    # wandb.run.summary["NLPD_top_5"] = nlpd_top_5
    # wandb.run.summary["R2_top_5"] = r2_top_5
    # wandb.run.summary["MAE_top_5"] = mae_top_5
    # wandb.run.summary["NLPD_bottom_5"] = nlpd_bottom_5
    # wandb.run.summary["R2_bottom_5"] = r2_bottom_5
    # wandb.run.summary["MAE_bottom_5"] = mae_bottom_5
    # wandb.run.summary["NLPD_all"] = nlpd_all
    # wandb.run.summary["R2_all"] = r2_all
    # wandb.run.summary["MAE_all"] = mae_all

    wandb.log(
        {
            "NLPD_top_5": nlpd_top_5,
            "R2_top_5": r2_top_5,
            "MAE_top_5": mae_top_5,
            "NLPD_bottom_5": nlpd_bottom_5,
            "R2_bottom_5": r2_bottom_5,
            "MAE_bottom_5": mae_bottom_5,
            "NLPD_all": nlpd_all,
            "R2_all": r2_all,
            "MAE_all": mae_all,
        }
    )

    # Save metrics to wandb
    wandb.save("metrics.csv")

    # wandb.save('config.yaml')
    wandb.finish()


if __name__ == "__main__":
    cli_main()
