import os
from pathlib import Path

import graphein.molecule as gm
import numpy as np
import pandas as pd
import selfies as sf
import torch
from drfp import DrfpEncoder
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, rdMolDescriptors
from rxnfp.tokenization import SmilesTokenizer
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer,
)
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelWithLMHead, AutoTokenizer, BertModel
from rdkit import Chem


# Reactions
def one_hot(df):
    """
    Builds reaction representation as a bit vector which indicates whether
    a certain condition, reagent, reactant etc. is present in the reaction.

    :param df: pandas DataFrame with columns representing different
    parameters of the reaction (e.g. reactants, reagents, conditions).
    :type df: pandas DataFrame
    :return: array of shape [len(reaction_smiles), sum(unique values for different columns in df)]
     with one-hot encoding of reactions
    """
    df_ohe = pd.get_dummies(df)
    return df_ohe.to_numpy(dtype=np.float64)


def rxnfp(reaction_smiles):
    """
    https://rxn4chemistry.github.io/rxnfp/

    Builds reaction representation as a continuous RXNFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), 256] with rxnfp featurised reactions

    """
    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)


def rxnfp2(reaction_smiles):
    print(os.getcwd())
    model_path = "../rxn_yields/trained_models/uspto/uspto_milligram_smooth_random_test_epochs_2_pretrained/checkpoint-30204-epoch-2/"
    tokenizer_vocab_path = "../rxn_yields/trained_models/uspto/uspto_milligram_smooth_random_test_epochs_2_pretrained/checkpoint-30204-epoch-2/vocab.txt"

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(tokenizer_vocab_path)

    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)


def drfp(reaction_smiles, bond_radius=3, nBits=2048):
    """
    https://github.com/reymond-group/drfp

    Builds reaction representation as a binary DRFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), nBits] with drfp featurised reactions

    """
    fps = DrfpEncoder.encode(reaction_smiles, n_folded_length=nBits, radius=bond_radius)
    return np.array(fps, dtype=np.float64)


def drxnfp(reaction_smiles, bond_radius=3, nBits=2048):
    drfps = drfp(reaction_smiles, nBits, radius=bond_radius)
    rxnfps = rxnfp(reaction_smiles)
    return np.concatenate([drfps, rxnfps], axis=1)


def drfpfingerprints(reaction_smiles, smiles, bond_radius=3, nBits=2048):
    drfps = drfp(reaction_smiles, nBits, radius=bond_radius)
    fingerprints = fingerprints(smiles, bond_radius=bond_radius, nBits=nBits)
    return np.concatenate([drfps, fingerprints], axis=1)


# Molecules
def fingerprints(smiles, bond_radius=3, nBits=2048):
    rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]
    return np.asarray(fps)


# auxiliary function to calculate the fragment representation of a molecule
def fragments(smiles):
    # descList[115:] contains fragment-based features only
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    # Update: in the new RDKit version the indices are [124:]
    fragments = {d[0]: d[1] for d in Descriptors.descList[124:]}
    frags = np.zeros((len(smiles), len(fragments)))
    for i in range(len(smiles)):
        mol = MolFromSmiles(smiles[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags[i, :] = features

    return frags


# auxiliary function to calculate bag of character representation of a molecular string
def bag_of_characters(smiles, max_ngram=5, selfies=False):
    if selfies:  # convert SMILES to SELFIES
        strings = [sf.encoder(smiles[i]) for i in range(len(smiles))]
    else:  # otherwise stick with SMILES
        strings = smiles

    # extract bag of character (boc) representation from strings
    cv = CountVectorizer(ngram_range=(1, max_ngram), analyzer="char", lowercase=False)
    return cv.fit_transform(strings).toarray()


def mqn_features(smiles):
    """
    Builds molecular representation as a vector of Molecular Quantum Numbers.
    :param reaction_smiles: list of molecular smiles
    :type reaction_smiles: list
    :return: array of mqn featurised molecules
    """
    molecules = [MolFromSmiles(smile) for smile in smiles]
    mqn_descriptors = [rdMolDescriptors.MQNs_(molecule) for molecule in molecules]
    return np.asarray(mqn_descriptors)


def chemberta_features(smiles):
    # any model weights from the link above will work here
    model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenized_smiles = [tokenizer(smile, return_tensors="pt") for smile in smiles]
    outputs = [
        model(
            input_ids=tokenized_smile["input_ids"],
            attention_mask=tokenized_smile["attention_mask"],
            output_hidden_states=True,
        )
        for tokenized_smile in tokenized_smiles
    ]
    embeddings = torch.cat(
        [output["hidden_states"][0].sum(axis=1) for output in outputs], axis=0
    )
    return embeddings.detach().numpy()


def graphs(smiles, graphein_config=None):
    return [gm.construct_graph(smiles=i, config=graphein_config) for i in smiles]


def cddd(smiles):
    current_path = os.getcwd()
    os.chdir(Path(os.path.abspath(__file__)).parent)
    cddd = pd.read_csv("precalculated_featurisation/cddd_additives_descriptors.csv")
    canonical_smiles_list = [
        Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True, canonical=True)
        for smile in smiles
    ]
    cddd["canonical"] = cddd["smiles"].apply(
        lambda x: Chem.MolToSmiles(
            Chem.MolFromSmiles(x), isomericSmiles=True, canonical=True
        )
    )
    merged_df = pd.DataFrame({"canonical": canonical_smiles_list}).merge(
        cddd, on="canonical", how="left"
    )
    descriptors = merged_df.drop(columns=["canonical", "smiles", "new_smiles"]).values
    os.chdir(current_path)
    return descriptors


# def xtb(smiles):
#     current_path = os.getcwd()
#     os.chdir(Path(os.path.abspath(__file__)).parent)
#     xtb = pd.read_csv("precalculated_featurisation/xtb_qm_descriptors_2.csv")
#     xtb_array = np.zeros((xtb.shape[0], len(xtb.columns[:-2])))
#     for i, smile in enumerate(smiles):
#         row = xtb[xtb["additives"] == smile][xtb.columns[:-2]].values
#         xtb_array[i] = row
#     os.chdir(current_path)
#     return xtb_array


def xtb(smiles):
    current_path = os.getcwd()
    os.chdir(Path(os.path.abspath(__file__)).parent)
    xtb = pd.read_csv("precalculated_featurisation/xtb_qm_descriptors_2.csv")
    canonical_smiles_list = [
        Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True, canonical=True)
        for smile in smiles
    ]
    xtb["canonical"] = xtb["Additive_Smiles"].apply(
        lambda x: Chem.MolToSmiles(
            Chem.MolFromSmiles(x), isomericSmiles=True, canonical=True
        )
    )
    merged_df = pd.DataFrame({"canonical": canonical_smiles_list}).merge(
        xtb, on="canonical", how="left"
    )
    descriptors = merged_df.drop(
        columns=["canonical", "Additive_Smiles", "UV210_Prod AreaAbs"]
    ).values
    os.chdir(current_path)
    return descriptors


def random_features():
    # todo random continous random bit vector
    pass
