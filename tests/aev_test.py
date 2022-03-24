import numpy as np
import torch
import torchani

from ael import loaders

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


def fc(Rij, Rc):
    """
    Cutoff function.
    """
    return torch.where(
        Rij < Rc,
        0.5 * torch.cos(Rij * (np.pi / Rc)) + 0.5,
        torch.tensor(0.0, dtype=Rij.dtype, device=device),
    )


def GR(Rij, Rc, eta, Rs, axis=0):
    """
    Radial function.

    Notes
    -----
    The factor of 0.25 is not in the paper, but shows up in the TorchANI implementation.
    """
    # Factor 0.25 is not in the paper
    # Shows up in the TorchANI code
    return 0.25 * torch.sum(torch.exp(-eta * (Rij - Rs) ** 2) * fc(Rij, Rc), axis=axis)


def GA(Rij, Rik, Tijk, Rc, eta, Rs, zeta, Ts, axis=0):
    """
    Angular function.
    """
    a = ((1.0 + torch.cos(Tijk - Ts)) / 2.0) ** zeta
    r = torch.exp(-eta * ((Rij + Rik) / 2.0 - Rs) ** 2)
    c = fc(Rij, Rc) * fc(Rik, Rc)

    return 2.0 * torch.sum(a * r * c, axis=axis)


def test_h2():
    """
    Test TorchANI AEV for H2.
    """
    n_species = 1  # H

    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 1
    # AEV: 1 * 1 + 1 * 1 * (1 + 1) // 2 = 1 (R) + 1 (A)
    # Radial: H
    # Angular: HH
    assert AEVC.aev_length == 2

    # Converts species (atomic numbers) to indices
    SC = torchani.SpeciesConverter("H")

    for atomicnum in [2, 3, 4, 5]:
        sc = SC((torch.tensor([[atomicnum]]), torch.tensor([[0.0, 0.0, 0.0]])))

        # Elements not present in SpeciesConverter are assigned -1
        assert sc.species.item() == -1

    # Define H2
    R = 1.0
    atomicnums = torch.tensor([[1, 1]], device=device)
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [R, 0.0, 0.0]]], device=device)

    # Map atomic numbers to index
    sc = SC((atomicnums, coordinates))

    assert torch.allclose(sc.species, torch.tensor([0, 0], device=device))
    assert torch.allclose(sc.coordinates, coordinates)

    aev = AEVC.forward((sc.species, sc.coordinates))

    assert torch.allclose(sc.species, aev.species)

    # Remove batch dimension and store as numpy array
    aevs = aev.aevs.squeeze(0).cpu().numpy()

    gr = GR(torch.tensor(R, device=device), RcR, EtaR, RsR).cpu().numpy()

    assert np.allclose(aevs[0, :], [gr, 0.0])  # AEV of atom 1
    assert np.allclose(aevs[1, :], [gr, 0.0])  # AEV of atom 2


def test_h2o():
    """
    Test TorchANI AEV for H2O.

    Notes
    -----
    The i-th component of the AEV correspond to the atomic environment of atom i. For
    H2O each AEV contains a [H, O] radial part and a [HH, HO, OO] radial part. This
    means that for O the first element of the radial part is a sum of the radial
    functions with both H1 and H2 while the second element of the radial part is zero
    """
    n_species = 2  # H, O

    AEVC = torchani.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)

    # Radial functions: 1
    # Angular functions: 1
    # Number of species: 2
    # AEV: 1 * 2 + 1 * 2 * (2 + 1) // 2 = 2 (R) + 3 (A)
    # Radial: H, O
    # Angular: HH, HO, OO
    assert AEVC.aev_length == 5

    # Converts species (atomic numbers) to indices
    SC = torchani.SpeciesConverter("HO")

    # H and O
    for idx, atomicnum in enumerate([1, 8]):
        sc = SC((torch.tensor([[atomicnum]]), torch.tensor([[0.0, 0.0, 0.0]])))
        assert sc.species.item() == idx

    # Other elements
    for atomicnum in [2, 3, 4, 5, 6, 7, 9]:
        sc = SC((torch.tensor([[atomicnum]]), torch.tensor([[0.0, 0.0, 0.0]])))

        # Elements not present in SpeciesConverter are assigned -1
        assert sc.species.item() == -1

    # Define H2O
    R = 1.0
    a = 104.5 / 180 * np.pi
    atomicnums = torch.tensor([[8, 1, 1]], device=device)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [R, 0.0, 0.0], [R * np.cos(a), R * np.sin(a), 0.0]]],
        device=device,
    )

    # Map atomic numbers to index
    sc = SC((atomicnums, coordinates))

    assert torch.allclose(sc.species, torch.tensor([1, 0, 0], device=device))
    assert torch.allclose(sc.coordinates, coordinates)

    aev = AEVC.forward((sc.species, sc.coordinates))

    assert torch.allclose(sc.species, aev.species)

    # Remove batch dimension and store as numpy array
    aevs = aev.aevs.squeeze(0).cpu().numpy()

    # Pairwise distances
    Rij = torch.nn.functional.pdist(coordinates.squeeze(0), p=2)

    # Rij contains the following distances: H1-O, H2-O, H1-H2
    d_HH = torch.norm(coordinates[:, 1] - coordinates[:, 2]).cpu().numpy()
    assert np.allclose(Rij.cpu().numpy(), [R, R, d_HH])

    # print(f"\n{aevs}")

    gr_H1H = GR(Rij[2], RcR, EtaR, RsR).cpu().numpy()  # H1 with H (H2)
    gr_H2H = GR(Rij[2], RcR, EtaR, RsR).cpu().numpy()  # H2 with H (H2)
    gr_H1O = GR(Rij[0], RcR, EtaR, RsR).cpu().numpy()  # H1 with O
    gr_H2O = GR(Rij[1], RcR, EtaR, RsR).cpu().numpy()  # H2 with O
    gr_OH = GR(Rij[:2], RcR, EtaR, RsR).cpu().numpy()  # O with H (H1 and H2)
    gr_OO = 0.00

    # Check radial function (H, O)
    assert np.allclose([gr_H1H], [gr_H2H])
    assert np.allclose([gr_H1O], [gr_H2O])
    assert np.allclose(aevs[0, :2], [gr_OH, gr_OO])  # Atom 0: O
    assert np.allclose(aevs[1, :2], [gr_H1H, gr_H1O])  # Atom 1: H1
    assert np.allclose(aevs[2, :2], [gr_H2H, gr_H2O])  # Atom 2: H1

    def GA_indices(i, j, k):
        vRij = coordinates.squeeze()[j, :] - coordinates.squeeze()[i, :]
        vRik = coordinates.squeeze()[k, :] - coordinates.squeeze()[i, :]

        # Factor 0.95 is not in the paper
        # Shows up in the TorchANI code
        Tijk = torch.acos(
            0.95 * torch.nn.functional.cosine_similarity(vRij, vRik, dim=0)
        )

        return GA(vRij.norm(), vRik.norm(), Tijk, RcA, EtaA, RsA, Zeta, TsA)

    ga_OH1H2 = GA_indices(0, 1, 2).item()  # Equivalent to (0, 2, 1)
    ga_H1OH2 = GA_indices(1, 0, 2).item()  # Equivalent to (1, 2, 0)
    ga_H2OH1 = GA_indices(2, 0, 1).item()  # Equivalent to (2, 1, 0)

    # Check angular function (HH, HO, OO)
    assert np.allclose(aevs[0, 2:], [ga_OH1H2, 0.0, 0.0])  # O with H & H
    assert np.allclose(aevs[1, 2:], [0.0, ga_H1OH2, 0.0])  # H1 with H & O
    assert np.allclose(aevs[2, 2:], [0.0, ga_H2OH1, 0.0])  # H2 with H & O


def test_aev_from_loader(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Compute map of atomic numbers to indices from species
    amap = loaders.anummap(data.species)

    # Transform atomic number to species in data
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
