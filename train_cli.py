import argparse
import logging
import os
import random
import string
import sys

import gpytorch
from botorch.models.model import Model as BaseModel
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from additive_bo.bo.module import BoModule
from additive_bo.data.module import BOAdditivesDataModule
from additive_bo.data_init_selection.clustering import BOInitDataSelection

# from additive_bo.surrogate_models.gp import GP  # noqa F401
from additive_bo.utils import flatten

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


def get_mol_or_rxn_smile(representation):
    if representation in ["rxnfp", "drfp"]:
        return "reaction_smiles"
    return "Additive_Smiles"


def get_distance_metric(kernel):
    # if representation in ["drfp", "fingerprints", "fragprints"]:
    if (
        kernel
        == "additive_bo.gprotorch.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel"
    ):
        return "jaccard"
    return "euclidean"


class MyLightningCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--n_iters", type=int, default=100)
        parser.add_argument("--n_trials", type=int, default=20)

        parser.add_class_arguments(BOInitDataSelection, "data_selection")
        parser.add_subclass_arguments(BaseModel, "surrogate_model")
        parser.add_subclass_arguments(gpytorch.kernels.Kernel, "kernel")

        parser.link_arguments(
            "data.representation",
            "data.featurize_column",
            apply_on="parse",
            compute_fn=get_mol_or_rxn_smile,
        )
        parser.link_arguments(
            "data_selection", "data.init_selection_method", apply_on="instantiate"
        )
        parser.link_arguments(
            "data_selection.n_clusters", "data.init_sample_size", apply_on="instantiate"
        )
        parser.link_arguments(
            "kernel", "surrogate_model.init_args.kernel", apply_on="instantiate"
        )

        parser.link_arguments(
            "data.train_x", "surrogate_model.init_args.train_x", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.train_y", "surrogate_model.init_args.train_y", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.noise", "surrogate_model.init_args.noise_val", apply_on="instantiate"
        )
        parser.link_arguments("data", "model.data", apply_on="instantiate")

        parser.link_arguments("surrogate_model", "model.model", apply_on="instantiate")
        parser.link_arguments("n_iters", "trainer.max_epochs")

        return super().add_arguments_to_parser(parser)

    def before_instantiate_classes(self) -> None:
        return super().before_instantiate_classes()


class WandbSaveConfigCallback(SaveConfigCallback):
    def __init__(self, parser, config, overwrite=True):
        super().__init__(parser, config, overwrite=overwrite)
        print(self.config)
        self.parser = parser
        self.config = config
        wandb_config = flatten(dict(self.config))
        wandb.config.update(wandb_config)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


letters = string.ascii_lowercase


def generate_group_name():
    return "".join(random.choice(letters) for i in range(10))


def start_new_run(seed):
    print(sys.argv)
    cli = MyLightningCli(
        model_class=BoModule,
        datamodule_class=BOAdditivesDataModule,
        run=False,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "logger": WandbLogger(  # save_dir=f'./wandb-save-dir/{group}',
                project="additives-debugging"
            ),  # , reinit=True),
            "log_every_n_steps": 1,
            "min_epochs": 1,
            "max_steps": -1,
            "accelerator": "cpu",
            "devices": 1,
            # "reload_dataloaders_every_n_epochs": 1,
            "num_sanity_val_steps": 0,
            # "callbacks": [Timer()],
        },
        args=sys.argv.append(f"--seed={seed}"),
        # save_config_overwrite=True,
        # seed_everything_default=seed
    )
    return cli


def cli_main():
    group = generate_group_name()
    cli = MyLightningCli(
        model_class=BoModule,
        datamodule_class=BOAdditivesDataModule,
        run=False,
        save_config_callback=WandbSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "logger": WandbLogger(  # save_dir=f'./wandb-save-dir/{group}',
                project="additives-report"
            ),
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
