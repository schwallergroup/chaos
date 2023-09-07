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
from chaos.data.module import BaseDataModule
from chaos.surrogate_models.gp import SimpleGP
from chaos.initialization.initializers import BOInitializer

from chaos.utils import flatten
from chaos.utils import convert_to_nested_dict

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

import copy
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
        parser.add_subclass_arguments(SimpleGP, "surrogate_model")

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
    wandb.finish()


if __name__ == "__main__":
    cli_main()
