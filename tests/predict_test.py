import numpy as np
import torch
import torchani

from ael import loaders, models, predict

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # type: ignore

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


def test_predict(testdata, testdir):

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
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 5
    # AEV: 1 * 5 + 1 * 5 * (5 + 1) // 2 = 5 (R) + 15 (A) = 20
    assert AEVC.aev_length == 20

    model = models.AffinityModel(n_species, AEVC.aev_length)

    ids, true, predicted = predict.predict(model, AEVC, loader)

    assert isinstance(true, np.ndarray)
    assert len(true) == batch_size

    assert isinstance(predicted, np.ndarray)
    assert len(predicted) == batch_size


def test_predict_baseline(testdata, testdir):

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
    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 5
    # AEV: 1 * 5 + 1 * 5 * (5 + 1) // 2 = 5 (R) + 15 (A) = 20
    assert AEVC.aev_length == 20

    model = models.AffinityModel(n_species, AEVC.aev_length)

    ids, true, predicted = predict.predict(model, AEVC, loader)

    assert isinstance(true, np.ndarray)
    assert len(true) == batch_size

    assert isinstance(predicted, np.ndarray)
    assert len(predicted) == batch_size

    # Systems are the other way around with respect to file order
    # This is to test that deltas are added to the correct ID
    delta_ids = np.array(["1a4w", "1a4r"])
    delta_baseline = np.array([500, 600])
    delta = np.array([5.92, 6.66])
    s = np.argsort(delta_ids)

    ids_b, true_b, predicted_b = predict.predict(
        model, AEVC, loader, baseline=(delta_ids, delta_baseline, delta)
    )

    sort = np.argsort(ids)
    bsort = np.argsort(ids_b)

    assert (ids[sort] == ids_b[bsort]).all()
    assert np.allclose(true[sort], true[bsort])
    assert np.allclose(predicted[sort], predicted_b[bsort] - delta_baseline[s])
