import pytest

import numpy as np

import torch
import torchani
from torch import nn

from ael import loaders, models

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


def test_atomicnn_default():
    n_inputs = 256
    default_sizes = [n_inputs, 160, 128, 96, 1]

    model = models.AtomicNN(n_inputs)

    assert len(model.layers) == 2 * (len(default_sizes) - 2) + 1

    for i, (in_size, out_size) in enumerate(zip(default_sizes[:-1], default_sizes[1:])):
        assert isinstance(model.layers[2 * i], nn.Linear)
        assert model.layers[2 * i].weight.shape == (out_size, in_size)


def test_atomicnn():
    n_inputs = 256
    sizes = [128, 64, 32, 16, 8, 4, 1]

    expected_sizes = [n_inputs] + sizes

    model = models.AtomicNN(n_inputs, sizes)

    assert len(model.layers) == 2 * (len(expected_sizes) - 2) + 1

    for i, (in_size, out_size) in enumerate(
        zip(expected_sizes[:-1], expected_sizes[1:])
    ):
        assert isinstance(model.layers[2 * i], nn.Linear)
        assert model.layers[2 * i].weight.shape == (out_size, in_size)


def test_atomicnn_dropout():
    n_inputs = 256
    sizes = [128, 64, 32, 16, 8, 4, 1]

    expected_sizes = [n_inputs] + sizes

    model = models.AtomicNN(n_inputs, sizes, 0.5)

    assert len(model.layers) == 3 * (len(expected_sizes) - 2) + 1

    for i, (in_size, out_size) in enumerate(
        zip(expected_sizes[:-1], expected_sizes[1:])
    ):
        assert isinstance(model.layers[3 * i], nn.Linear)
        assert model.layers[3 * i].weight.shape == (out_size, in_size)


def test_affinitymodel_parameters():
    n_inputs = 256
    dropp = 0.5
    n_species = 10
    layers_sizes = [128, 64, 1]

    model = models.AffinityModel(n_species, n_inputs, layers_sizes, dropp)

    assert model.n_species == n_species
    assert model.aev_length == n_inputs
    assert model.layers_sizes == [n_inputs] + layers_sizes
    assert model.dropp == pytest.approx(dropp)


def test_forward(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    _, labels, (species, coordinates) = next(iloader)

    # Move everything to device
    labels = labels.to(device)
    species = species.to(device)
    coordinates = coordinates.to(device)

    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 5
    # AEV: 1 * 5 + 1 * 5 * (5 + 1) // 2 = 5 (R) + 15 (A) = 20
    assert AEVC.aev_length == 20

    aev = AEVC.forward((species, coordinates))

    assert aev.species.shape == species.shape
    assert aev.aevs.shape == (batch_size, 42, 20)

    model = models.AffinityModel(n_species, AEVC.aev_length)

    # Move model to device
    model.to(device)

    output = model(aev.species, aev.aevs)

    assert output.shape == (batch_size,)

def test_forward_atomic(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    _, labels, (species, coordinates) = next(iloader)

    # Move everything to device
    labels = labels.to(device)
    species = species.to(device)
    coordinates = coordinates.to(device)

    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 5
    # AEV: 1 * 5 + 1 * 5 * (5 + 1) // 2 = 5 (R) + 15 (A) = 20
    assert AEVC.aev_length == 20

    aev = AEVC.forward((species, coordinates))

    assert aev.species.shape == species.shape
    assert aev.aevs.shape == (batch_size, 42, 20)

    model = models.AffinityModel(n_species, AEVC.aev_length)

    # Move model to device
    model.to(device)

    output = model(aev.species, aev.aevs)
    assert output.shape == (batch_size,)

    atomic_constributions = model._forward_atomic(aev.species, aev.aevs)

    assert atomic_constributions.shape == species.shape

    o = torch.sum(atomic_constributions, dim=1)

    assert np.allclose(output.cpu().detach().numpy(), o.cpu().detach().numpy())
