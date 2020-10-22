import os

import numpy as np
import pytest
import torch
import torchani
from torch import nn

from ael import constants, loaders, models, utils

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Radial coefficients
RcR = 5.2
EtaR = torch.tensor([16.0], device=device)
RsR = torch.tensor([0.9], device=device)

# Angular coefficients (Ga)
RcA = 3.5
Zeta = torch.tensor([32], device=device)
TsA = torch.tensor([0.19634954], device=device)  # Angular shift in GA
EtaA = torch.tensor([8.0], device=device)
RsA = torch.tensor([0.9], device=device)  # Radial shift in GA


def test_saveAEVC_loadAEVC(tmpdir):
    n_species = 10

    # Define AEVComputer
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 10
    # AEV: 1 * 10 + 1 * 10 * (10 + 1) // 2 = 10 (R) + 55 (A) = 65
    assert AEVC.aev_length == 65

    path = os.path.join(tmpdir, "aevc-tmp.pth")

    utils.saveAEVC(AEVC, n_species, path)

    AEVC_loaded = utils.loadAEVC(path)

    assert AEVC.aev_length == AEVC_loaded.aev_length == 65

    # Ints
    assert AEVC.num_species == AEVC_loaded.num_species == n_species

    # Floats
    assert np.allclose(AEVC_loaded.Rcr, RcR)
    assert np.allclose(AEVC_loaded.Rca, RcA)

    assert torch.allclose(AEVC_loaded.EtaR, EtaR)
    assert torch.allclose(AEVC_loaded.EtaA, EtaA)

    assert torch.allclose(AEVC_loaded.ShfR, RsR)
    assert torch.allclose(AEVC_loaded.ShfA, RsA)

    assert torch.allclose(AEVC_loaded.Zeta, Zeta)
    assert torch.allclose(AEVC_loaded.ShfZ, TsA)

    assert AEVC.radial_sublength == AEVC_loaded.radial_sublength
    assert AEVC.radial_length == AEVC_loaded.radial_length
    assert AEVC.angular_sublength == AEVC.angular_sublength
    assert AEVC.angular_length == AEVC.angular_length


@pytest.mark.parametrize("dropp", [None, 0.0, 0.25])
@pytest.mark.parametrize("eval", [True, False])
def test_savemodel_loadmodel(tmpdir, eval, dropp):
    n_species = 10

    # Define AEVComputer
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 10
    # AEV: 1 * 10 + 1 * 10 * (10 + 1) // 2 = 10 (R) + 55 (A) = 65
    assert AEVC.aev_length == 65

    model = models.AffinityModel(n_species, AEVC.aev_length, dropp=dropp)

    path = os.path.join(tmpdir, "model-tmp.pth")

    utils.savemodel(model, path)

    model_loaded = utils.loadmodel(path, eval=eval)

    assert model.aev_length == model_loaded.aev_length == 65
    assert model.n_species == model_loaded.n_species == n_species
    assert model.dropp == model.dropp
    assert model.layers_sizes == model_loaded.layers_sizes

    # Check weights
    for ANN, ANNl in zip(model.modules(), model_loaded.modules()):
        for layer, layerl in zip(ANN.modules(), ANNl.modules()):
            if type(layer) == nn.Linear:
                assert torch.allclose(layer.weight, layerl.weight)
                assert torch.allclose(layer.bias, layerl.bias)


def test_saveamap_loadamap(tmpdir):

    amap = {1: 0, 2: 1, 3: 2}

    path = os.path.join(tmpdir, "amap-tmp.json")

    utils.save_amap(amap, path)
    amapl = utils.load_amap(path)

    for k in amap.keys():
        assert amap[k] == amapl[k]


def test_labels_scaler_single(testdata, testdir):

    data = loaders.PDBData(testdata, 3.5, testdir)

    labels = data.labels.copy()

    scaler = utils.labels_scaler(data)

    assert np.allclose(data.labels, [1.0, -1.0])
    assert np.allclose(labels, scaler.inverse_transform(data.labels))


def test_labels_scaler(testdata, testdir):

    data1 = loaders.PDBData(testdata, 3.5, testdir)
    data2 = loaders.PDBData(testdata, 3.5, testdir)

    labels1 = data1.labels.copy()
    labels2 = data2.labels.copy()

    scaler = utils.labels_scaler(data1, data2)

    assert np.allclose(data1.labels, [1.0, -1.0])
    assert np.allclose(data2.labels, [1.0, -1.0])

    assert np.allclose(labels1, scaler.inverse_transform(data1.labels))
    assert np.allclose(labels2, scaler.inverse_transform(data2.labels))


@pytest.mark.parametrize(
    "score, result", [(0.0, 0.0), (constants.R * constants.T, -np.log10(np.exp(1)))]
)
def test_vina_to_pK(score, result):

    r = utils.vina_to_pK(score)

    assert r == pytest.approx(result)
