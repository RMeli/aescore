import mlflow
import numpy as np
import torch
import torchani
from torch import nn, optim

from ael import loaders, models, train

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


def test_train_small(testdata, testdir):

    with mlflow.start_run():

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

        # Define AEVComputer
        AEVC = torchani.AEVComputer(
            RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species
        )

        # Radial functions: 1
        # Angular functions: 1
        # Number of species: 5
        # AEV: 1 * 5 + 1 * 5 * (5 + 1) // 2 = 5 (R) + 15 (A) = 20
        assert AEVC.aev_length == 20

        model = models.AffinityModel(n_species, AEVC.aev_length)
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        mse = nn.MSELoss()

        # Check number of ANNs
        assert len(model) == n_species

        train_losses, valid_losses = train.train(
            model,
            optimizer,
            mse,
            AEVC,
            loader,
            loader,
            epochs=15,  # torchani.AEVComputer
        )

        # Validation loss is shifted when trainloader and testloader are the same
        assert np.allclose(train_losses[1:], valid_losses[:-1])


def test_train_small_cmap(testdata, testdir):

    # Map all elements to dummy atom
    cmap = {"C": ["N", "O"]}  # Map N and O to C, leave P and S

    with mlflow.start_run():

        # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
        data = loaders.PDBData(testdata, 0.1, testdir, cmap)

        batch_size = 2

        # Transform atomic numbers to species
        amap = loaders.anummap(data.species)
        data.atomicnums_to_idxs(amap)

        n_species = len(amap)

        # cmap maps everything to single dummy element
        assert n_species == 3

        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
        )

        # Define AEVComputer
        AEVC = torchani.AEVComputer(
            RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species
        )

        # Radial functions: 1
        # Angular functions: 1
        # Number of species: 3
        # AEV: 1 * 3 + 1 * 3 * (3 + 1) // 2 = 3 (R) + 6 (A) = 9
        assert AEVC.aev_length == 9

        model = models.AffinityModel(n_species, AEVC.aev_length)
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        mse = nn.MSELoss()

        # Check number of ANNs
        assert len(model) == n_species

        train_losses, valid_losses = train.train(
            model,
            optimizer,
            mse,
            AEVC,
            loader,
            loader,
            epochs=15,  # torchani.AEVComputer
        )

        # Validation loss is shifted when trainloader and testloader are the same
        assert np.allclose(train_losses[1:], valid_losses[:-1])


def test_train_small_cmap_dummy(testdata, testdir):

    # Map all elements to dummy atom
    cmap = {"X": ["C", "N", "O", "S", "P"]}

    with mlflow.start_run():

        # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
        data = loaders.PDBData(testdata, 0.1, testdir, cmap)

        batch_size = 2

        # Transform atomic numbers to species
        amap = loaders.anummap(data.species)
        data.atomicnums_to_idxs(amap)

        n_species = len(amap)

        # cmap maps everything to single dummy element
        assert n_species == 1

        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
        )

        # Define AEVComputer
        AEVC = torchani.AEVComputer(
            RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species
        )

        # Radial functions: 1
        # Angular functions: 1
        # Number of species: 1
        # AEV: 1 * 1 + 1 * 1 * (1 + 1) // 2 = 1 (R) + 1 (A) = 2
        assert AEVC.aev_length == 2

        model = models.AffinityModel(n_species, AEVC.aev_length)
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        mse = nn.MSELoss()

        # Check number of ANNs
        assert len(model) == n_species

        train_losses, valid_losses = train.train(
            model,
            optimizer,
            mse,
            AEVC,
            loader,
            loader,
            epochs=15,  # torchani.AEVComputer
        )

        # Validation loss is shifted when trainloader and testloader are the same
        assert np.allclose(train_losses[1:], valid_losses[:-1])
