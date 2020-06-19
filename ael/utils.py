import json
import os

import mlflow
import numpy as np
import torch
import torchani

from ael import constants, models


def savemodel(model, path):
    torch.save(
        {
            "args": {
                "n_species": model.n_species,
                "aev_length": model.aev_length,
                # Drop first layer size which is n_species
                "layers_sizes": model.layers_sizes[1:],
                "dropp": model.dropp,
            },
            "state_dict": model.state_dict(),
        },
        path,
    )


def loadmodel(path, eval=True):
    d = torch.load(path)

    model = models.AffinityModel(**d["args"])

    model.load_state_dict(d["state_dict"])

    if eval:
        # Put model in evaluation mode
        model.eval()
    else:
        # Put model in training mode
        model.train()

    return model


def saveAEVC(AEVC, n_species, path):
    Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = AEVC.constants()

    torch.save(
        {
            "args": {
                "Rcr": Rcr,
                "EtaR": EtaR,
                "ShfR": ShfR,
                "Rca": Rca,
                "ShfZ": ShfZ,
                "EtaA": EtaA,
                "Zeta": Zeta,
                "ShfA": ShfA,
                "num_species": n_species,
            },
            "state_dict": AEVC.state_dict(),
        },
        path,
    )

    mlflow.log_artifact(path)


def loadAEVC(path):
    d = torch.load(path)

    AEVC = torchani.AEVComputer(**d["args"])

    AEVC.load_state_dict(d["state_dict"])

    return AEVC


def save_amap(amap, path=""):

    # Original amap contains np.int64
    converted = {int(k): int(v) for k, v in amap.items()}

    fname = os.path.join(path, "amap.json")
    with open(fname, "w") as fout:
        json.dump(converted, fout)

    mlflow.log_artifact(fname)


def load_amap(fname):

    with open(fname, "r") as fin:
        amap = json.load(fin)

    # Store loaded amap as MLFlow artifact
    mlflow.log_artifact(fname)

    # Convert strings to integers
    amap = {int(k): int(v) for k, v in amap.items()}

    return amap


def vina_to_pK(score):
    return -np.log10(np.exp(score / constants.R / constants.T))
