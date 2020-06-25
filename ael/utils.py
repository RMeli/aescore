import json
from typing import Dict

import mlflow
import numpy as np
import torch
import torchani
from torch import nn

from ael import constants, models


def savemodel(model: nn.ModuleDict, path) -> None:
    """
    Save AffinityModel.

    Parameters
    ----------
    model: torch.nn.ModuleDict
        Model
    path:
        Save path
    """
    torch.save(
        {
            "args": {
                "n_species": model.n_species,
                "aev_length": model.aev_length,
                # Drop first layer size which is n_species
                "layers_sizes": model.layers_sizes[1:],  # type: ignore
                "dropp": model.dropp,
            },
            "state_dict": model.state_dict(),
        },
        path,
    )

    mlflow.log_artifact(path)


def loadmodel(path, eval: bool = True) -> nn.ModuleDict:
    """
    Load AffinityModel.

    Parameters
    ----------
    path:
        Save path
    eval: bool
        Flag to put model in evaluation mode

    Returns
    -------
    nn.ModuleDict
        Model

    Notes
    -----
    Evaluation mode is needed to switch off the dropout layers when using the model
    for inference.
    """
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


def saveAEVC(AEVC: torchani.AEVComputer, n_species: int, path) -> None:
    """
    Save AEVComputer.

    Parameters
    ----------
    AEVC: torchani.AEVComputer
        AEVComputer
    n_species: int
        Number of species
    path:
        Save path
    """
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


def loadAEVC(path) -> torchani.AEVComputer:
    """
    Load AEVComputer.

    Parameters
    ----------

    Returns
    -------
    torchani.AEVComputer
        AEVComputer
    """
    d = torch.load(path)

    AEVC = torchani.AEVComputer(**d["args"])

    AEVC.load_state_dict(d["state_dict"])

    return AEVC


def save_amap(amap: Dict[int, int], path):
    """
    Save atomic number to index map.

    Parameters
    ----------
    amap:
        Atomic numbers to index map
    path:
        Save path
    """

    # Original amap contains np.int64
    converted = {int(k): int(v) for k, v in amap.items()}

    with open(path, "w") as fout:
        json.dump(converted, fout)

    mlflow.log_artifact(path)


def load_amap(fname) -> Dict[int, int]:
    """
    Load atomic number to index map.

    Parameters
    ----------
    fname
        File name

    Returns
    -------
    Dict[int, int]
        Atomic number to index map
    """

    with open(fname, "r") as fin:
        amap = json.load(fin)

    # Store loaded amap as MLFlow artifact
    mlflow.log_artifact(fname)

    # Convert strings to integers
    amap = {int(k): int(v) for k, v in amap.items()}

    return amap


def vina_to_pK(score):
    return -np.log10(np.exp(score / constants.R / constants.T))
