import mlflow
import numpy as np
import torch
import torchani
from torch import nn

from ael import grad, loaders, models

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Radial coefficients
RcR = 5.2
EtaR = torch.tensor([16.0], device=device)
RsR = torch.tensor([0.9], device=device)

# Angular coefficients
RcA = 3.5
Zeta = torch.tensor([32], device=device)
TsA = torch.tensor([0.19634954], device=device)  # Angular shift in GA
EtaA = torch.tensor([8.0], device=device)
RsA = torch.tensor([0.9], device=device)  # Radial shift in GA


def test_grad(testdata, testdir):

    with mlflow.start_run():

        # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
        data = loaders.PDBData(testdata, 0.1, testdir)

        n_systems = len(data)

        assert n_systems == 2

        # Transform atomic numbers to species
        amap = loaders.anummap(data.species)
        data.atomicnums_to_idxs(amap)

        n_species = len(amap)

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
        loss = nn.MSELoss()

        # Move model and AEVComputer to device
        model.to(device)
        AEVC.to(device)

        # Model in evaluation mode
        model.eval()

        for i in range(n_systems):
            pdbid, label, (species, coordinates) = data[i]

            gradient = grad.gradient(
                species, coordinates, label, model, AEVC, loss, device
            )

            assert gradient.shape == coordinates.shape


def test_atomic(testdata, testdir):

    with mlflow.start_run():

        # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
        data = loaders.PDBData(testdata, 0.1, testdir)

        n_systems = len(data)

        assert n_systems == 2

        # Transform atomic numbers to species
        amap = loaders.anummap(data.species)
        data.atomicnums_to_idxs(amap)

        n_species = len(amap)

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

        # Move model and AEVComputer to device
        model.to(device)
        AEVC.to(device)

        # Model in evaluation mode
        model.eval()

        for pdbid, _, (species, coordinates) in data:

            atomic = grad.atomic(species, coordinates, model, AEVC, device)

            # Add fictitious batch dimension
            species = species.unsqueeze(0)
            coordinates = coordinates.unsqueeze(0)

            assert atomic.shape == species.shape

            aevs = AEVC.forward((species, coordinates)).aevs
            prediction = model(species, aevs)

            assert np.allclose(
                torch.sum(atomic, dim=1).cpu().detach().numpy(),
                prediction.cpu().detach().numpy(),
            )
