import graphein.molecule as gm
import numpy as np
import pandas as pd
import selfies as sf
import torch
from drfp import DrfpEncoder
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, rdMolDescriptors
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer,
)
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelWithLMHead, AutoTokenizer


# Reactions
def rxnfp(reaction_smiles):
    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=float)


def one_hot(df):
    df_ohe = pd.get_dummies(df)
    return df_ohe.to_numpy(dtype=float)


def drfp(reaction_smiles, nBits=2048):
    fps = DrfpEncoder.encode(reaction_smiles, n_folded_length=nBits)
    return np.asarray(fps, dtype=float)


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
        strings = []
        for i in range(len(smiles)):
            strings.append(sf.encoder(smiles[i]))
    else:  # otherwise stick with SMILES
        strings = smiles

    # extract bag of character (boc) representation from strings
    cv = CountVectorizer(ngram_range=(1, max_ngram), analyzer="char", lowercase=False)
    bocs = cv.fit_transform(strings).toarray()
    return bocs


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


def mqn_features(smiles):
    molecules = [MolFromSmiles(smile) for smile in smiles]
    mqn_descriptors = [rdMolDescriptors.MQNs_(molecule) for molecule in molecules]
    return np.asarray(mqn_descriptors)


def graphs(smiles, graphein_config=None):
    return [gm.construct_graph(smiles=i, config=graphein_config) for i in smiles]


def cddd(smiles):
    cddd = pd.read_csv("data/cddd_additives_descriptors.csv")
    cddd_array = np.zeros((cddd.shape[0], 512))
    for i, smile in enumerate(smiles):
        row = cddd[cddd["smiles"] == smile][cddd.columns[3:]].values
        cddd_array[i] = row
    return cddd_array


def xtb(smiles):
    xtb = pd.read_csv("data/xtb_qm_descriptors_2.csv")
    xtb_array = np.zeros((xtb.shape[0], len(xtb.columns[:-2])))
    for i, smile in enumerate(smiles):
        row = xtb[xtb["Additive_Smiles"] == smile][xtb.columns[:-2]].values
        xtb_array[i] = row
    return xtb_array


def random_features():
    # todo random continous random bit vector
    pass
