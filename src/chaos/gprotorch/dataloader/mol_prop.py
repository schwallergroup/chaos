"""
Instantiation of the abstract data loader class for
molecular property prediction datasets.
"""
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from chaos.gprotorch.data_featuriser import (
    bag_of_characters,
    cddd,
    chemberta_features,
    fingerprints,
    fragments,
    graphs,
    mqn_features,
    xtb,
)
from chaos.gprotorch.dataloader import DataLoader


class DataLoaderMP(DataLoader):
    def __init__(self):
        super(DataLoaderMP, self).__init__()
        self.task = "molecular_property_prediction"
        self._features = None
        self._labels = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    def validate(self, drop=True):
        """Checks if the features are valid SMILES strings and (potentially)
        drops the entries that are not.

        :param drop: whether to drop invalid entries
        :type drop: bool
        """

        invalid_idx = []

        # iterate through the features
        for i in range(len(self.features)):
            # try to convert each SMILES to an rdkit molecule
            mol = MolFromSmiles(self.features[i])

            # if it does not work, save the index and print its position to the console
            if mol is None:
                invalid_idx.append(i)
                print(f"Invalid SMILES at position {i+1}: {self.features[i]}")

        if drop:
            self.features = np.delete(self.features, invalid_idx).tolist()
            self.labels = np.delete(self.labels, invalid_idx)

    def featurize(
        self,
        representation,
        bond_radius=3,
        nBits=2048,
        graphein_config=None,
        max_ngram=5,
    ):
        """Transforms SMILES into the specified molecular representation.

        :param representation: the desired molecular representation, one of [fingerprints, fragments, fragprints]
        :type representation: str
        :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
        :type bond_radius: int
        :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
        :type nBits: int
        """

        valid_representations = [
            "fingerprints",
            "fragments",
            "fragprints",
            "bag_of_smiles",
            "bag_of_selfies",
            "chemberta",
            "graphs",
            "chemprints",
            "mqn",
            "cddd",
            "xtb",
            "cddd+xtb",
            "mqn+xtb",
            "cddd+xtb+mqn",
            "fingerprints+xtb",
            "fragprints+xtb",
        ]

        if representation == "fingerprints":
            self.features = fingerprints(
                self.features, bond_radius=bond_radius, nBits=nBits
            )

        elif representation == "fragments":
            self.features = fragments(self.features)

        elif representation == "fragprints":
            self.features = np.concatenate(
                (
                    fingerprints(self.features, bond_radius=bond_radius, nBits=nBits),
                    fragments(self.features),
                ),
                axis=1,
            )
        elif representation == "chemprints":
            self.features = np.concatenate(
                (
                    chemberta_features(self.features),
                    fingerprints(self.features, bond_radius=bond_radius, nBits=nBits),
                ),
                axis=1,
            )
        elif representation == "mqn":
            self.features = mqn_features(self.features)

        elif representation == "cddd":
            self.features = cddd(self.features)

        elif representation == "xtb":
            self.features = xtb(self.features)

        elif representation == "cddd+xtb":
            self.features = np.concatenate(
                (
                    cddd(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif representation == "cddd+xtb+mqn":
            self.features = np.concatenate(
                (
                    cddd(self.features),
                    xtb(self.features),
                    mqn_features(self.features),
                ),
                axis=1,
            )

        elif representation == "mqn+xtb":
            self.features = np.concatenate(
                (
                    mqn_features(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif representation == "fingerprints+xtb":
            self.features = np.concatenate(
                (
                    fingerprints(self.features, bond_radius=bond_radius, nBits=nBits),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif representation == "fragprints+xtb":
            self.features = np.concatenate(
                (
                    fingerprints(self.features, bond_radius=bond_radius, nBits=nBits),
                    fragments(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif representation == "bag_of_selfies":
            self.features = bag_of_characters(self.features, selfies=True)

        elif representation == "bag_of_smiles":
            self.features = bag_of_characters(self.features)

        elif representation == "random":
            self.features = random_features(self.features)

        elif representation == "chemberta":
            self.features = chemberta_features(self.features)

        elif representation == "chemberta+xtb":
            self.features = np.concatenate(
                (
                    chemberta_features(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif representation == "chemberta+xtb+mqn":
            self.features = np.concatenate(
                (
                    chemberta_features(self.features),
                    xtb(self.features),
                    mqn_features(self.features),
                ),
                axis=1,
            )

        elif representation == "graphs":
            self.features = graphs(self.features, graphein_config)

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )

    def load_benchmark(self, benchmark, path):
        """Loads features and labels from one of the included benchmark datasets
        and feeds them into the DataLoader.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[Photoswitch, ESOL, FreeSolv, Lipophilicity]``.
        :type benchmark: str
        :param path: the path to the dataset in csv format
        :type path: str
        """

        benchmarks = {
            "Photoswitch": {
                "features": "SMILES",
                "labels": "E isomer pi-pi* wavelength in nm",
            },
            "ESOL": {
                "features": "smiles",
                "labels": "measured log solubility in mols per litre",
            },
            "FreeSolv": {"features": "smiles", "labels": "expt"},
            "Lipophilicity": {"features": "smiles", "labels": "exp"},
        }

        if benchmark in benchmarks:
            df = pd.read_csv(path)
            # drop nans from the datasets
            nans = df[benchmarks[benchmark]["labels"]].isnull().to_list()
            nan_indices = [nan for nan, x in enumerate(nans) if x]
            self.features = (
                df[benchmarks[benchmark]["features"]].drop(nan_indices).to_list()
            )
            self.labels = (
                df[benchmarks[benchmark]["labels"]].dropna().to_numpy().reshape(-1, 1)
            )

        else:
            raise ValueError(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )


if __name__ == "__main__":
    loader = DataLoaderMP()
    loader.load_benchmark("ESOL", "../../data/property_prediction/ESOL.csv")
    print(loader.featurize("chemprints"))
