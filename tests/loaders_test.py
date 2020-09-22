import os

import MDAnalysis as mda
import numpy as np
import pytest
import torch

from ael import loaders


@pytest.mark.parametrize(
    "system, n_ligand, n_receptor", [("1a4r", 28, 6009), ("1a4w", 42, 4289)]
)
def test_load_pdbs(testdir, system, n_ligand, n_receptor):

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    ulig = mda.Universe(os.path.join(testdir, ligname))
    urec = mda.Universe(os.path.join(testdir, recname))

    assert len(ulig.atoms) == n_ligand
    assert len(urec.atoms) == n_receptor

    system = loaders.load_pdbs(ligname, recname, testdir)

    assert len(system.atoms) == n_ligand + n_receptor

    lig = system.select_atoms("resname LIG")

    assert len(lig.atoms) == n_ligand


def test_load_pdbs_fail_lig(testdir):

    system = "1a4r"

    ligname = os.path.join(system, f"{system}_FAIL.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    with pytest.raises(Exception):
        system = loaders.load_pdbs(ligname, recname, testdir)


def test_load_pdbs_fail_rec(testdir):

    system = "1a4r"

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_FAIL.pdb")

    with pytest.raises(Exception):
        system = loaders.load_pdbs(ligname, recname, testdir)


@pytest.mark.parametrize(
    "system, distance, n_ligand, n_receptor",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        ("1a4r", 0.1, 28, 0),
        ("1a4r", 1.8, 28, 10),
        ("1a4r", 2.0, 28, 42),
        ("1a4r", 2.5, 28, 91),
        ("1a4r", 3.0, 28, 188),
        ("1a4w", 0.1, 42, 0),
        ("1a4w", 1.95, 42, 22),
        ("1a4w", 2.0, 42, 29),
        ("1a4w", 2.5, 42, 60),
        ("1a4w", 3.0, 42, 156),
    ],
)
def test_select(testdir, system, distance, n_ligand, n_receptor):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND
    """

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    system = loaders.load_pdbs(ligname, recname, testdir)

    atoms, coordinates = loaders.select(system, distance)

    # assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize(
    "system, distance, n_ligand, n_receptor",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        ("1a4r", 0.1, 28, 0),
        ("1a4r", 2.5, 28, 45),
        ("1a4w", 0.1, 42, 0),
        ("1a4w", 2.5, 42, 31),
    ],
)
def test_select_removeHs(testdir, system, distance, n_ligand, n_receptor):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    system = loaders.load_pdbs(ligname, recname, testdir)

    atoms, coordinates = loaders.select(system, distance, removeHs=True)

    # assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize(
    "system, distance, n_ligand, n_receptor",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        ("1a4r", 0.1, 28, 0),
        ("1a4r", 1.8, 28, 10),
        ("1a4r", 2.0, 28, 42),
        ("1a4r", 2.5, 28, 91),
        ("1a4r", 3.0, 28, 188),
        ("1a4w", 0.1, 42, 0),
        ("1a4w", 1.95, 42, 22),
        ("1a4w", 2.0, 42, 29),
        ("1a4w", 2.5, 42, 60),
        ("1a4w", 3.0, 42, 156),
    ],
)
def test_load_pdbs_and_select(testdir, system, distance, n_ligand, n_receptor):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND
    """

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    atoms, coordinates = loaders.load_pdbs_and_select(
        ligname, recname, distance, testdir
    )

    # assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize(
    "system, distance, n_ligand, n_receptor",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        ("1a4r", 0.1, 28, 0),
        ("1a4r", 2.5, 28, 45),
        ("1a4w", 0.1, 42, 0),
        ("1a4w", 2.5, 42, 31),
    ],
)
def test_load_pdbs_and_select_removeHs(testdir, system, distance, n_ligand, n_receptor):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    atoms, coordinates = loaders.load_pdbs_and_select(
        ligname, recname, distance, testdir, removeHs=True
    )

    # assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize(
    "els, zs",
    [
        (["H", "H"], [1, 1]),
        (["H", "O", "H"], [1, 8, 1]),
        (["C", "N", "S", "O", "P"], [6, 7, 16, 8, 15]),
    ],
)
def test_elements_to_atomicnums(els, zs):

    Zs = loaders.elements_to_atomicnums(els)

    assert np.allclose(Zs, zs)


@pytest.mark.parametrize(
    "distance, n1_atoms, n2_atoms",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        (0.1, 28 + 0, 42 + 0),
        (2.0, 28 + 42, 42 + 29),
        (2.5, 28 + 91, 42 + 60),
        (3.0, 28 + 188, 42 + 156),
    ],
)
def test_pdbloader(testdata, testdir, distance, n1_atoms, n2_atoms):

    data = loaders.PDBData(testdata, distance, testdir)

    batch_size = 1

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    n_atoms_iter = iter([n1_atoms, n2_atoms])

    for ids, label, (species, coordinates) in iloader:

        n_atoms = next(n_atoms_iter)

        assert isinstance(ids, np.ndarray)
        assert ids.shape == (batch_size,)

        assert isinstance(label, torch.Tensor)
        assert label.shape == (batch_size,)

        assert isinstance(species, torch.Tensor)
        assert species.shape == (batch_size, n_atoms)

        assert isinstance(coordinates, torch.Tensor)
        assert coordinates.shape == (batch_size, n_atoms, 3)


@pytest.mark.parametrize(
    "distance, n1_atoms, n2_atoms",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        (0.1, 28 + 0, 42 + 0),
        (2.5, 28 + 45, 42 + 31),
    ],
)
def test_pdbloader_removeHs(testdata, testdir, distance, n1_atoms, n2_atoms):

    data = loaders.PDBData(testdata, distance, testdir, removeHs=True)

    batch_size = 1

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    n_atoms_iter = iter([n1_atoms, n2_atoms])

    for ids, label, (species, coordinates) in iloader:

        n_atoms = next(n_atoms_iter)

        assert isinstance(ids, np.ndarray)
        assert ids.shape == (batch_size,)

        assert isinstance(label, torch.Tensor)
        assert label.shape == (batch_size,)

        assert isinstance(species, torch.Tensor)
        assert species.shape == (batch_size, n_atoms)

        assert isinstance(coordinates, torch.Tensor)
        assert coordinates.shape == (batch_size, n_atoms, 3)


@pytest.mark.parametrize(
    "distance, n1_atoms, n2_atoms",
    [
        # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
        (0.1, 28 + 0, 42 + 0),
        (2.0, 28 + 42, 42 + 29),
        (2.5, 28 + 91, 42 + 60),
        (3.0, 28 + 188, 42 + 156),
    ],
)
def test_pdbloader_batch(testdata, testdir, distance, n1_atoms, n2_atoms):

    data = loaders.PDBData(testdata, distance, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    ids, labels, (species, coordinates) = next(iloader)

    assert isinstance(ids, np.ndarray)
    assert ids.shape == (batch_size,)

    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (batch_size,)

    assert isinstance(species, torch.Tensor)
    assert species.shape == (batch_size, max(n1_atoms, n2_atoms))

    assert isinstance(coordinates, torch.Tensor)
    assert coordinates.shape == (batch_size, max(n1_atoms, n2_atoms), 3)


def test_atomicnum_map(testdata, testdir):

    data = loaders.PDBData(testdata, 2.0, testdir)

    amap = loaders.anummap(data.species)

    # Elements: H, C, N, O, P, S
    assert len(amap) == 6
    assert [1, 6, 7, 8, 15, 16] == list(amap.keys())
    assert list(range(6)) == list(amap.values())


def test_pdbloader_labels(testdata, testdir):
    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    ids, labels, (species, coordinates) = next(iloader)

    assert ids.shape == (batch_size,)
    assert (ids == np.array(["1a4r", "1a4w"])).all()
    assert labels.shape == (batch_size,)
    assert torch.allclose(labels, torch.tensor([6.66, 5.92]))


def elements_to_idxs(elements, amap):
    anums = loaders.elements_to_atomicnums(list(elements))

    return [amap[anum] for anum in anums]


def test_pdbloader_ligand_species(testdata, testdir):

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    ids, labels, (species, coordinates) = next(iloader)

    assert (ids == np.array(["1a4r", "1a4w"])).all()

    assert species.shape == (batch_size, 42)  # Ligand 1a4w is the largest

    # Test ligand 1a4r (padded with -1)
    assert torch.allclose(
        species[0, :],
        torch.tensor(
            elements_to_idxs("NPOOOPOOOCCOCOCOCNCNCCONCNNC", amap) + 14 * [-1]
        ),
    )

    # Test ligand 1a4w (no padding)
    assert torch.allclose(
        species[1, :],
        torch.tensor(
            elements_to_idxs("CCCCCCCCCCNCCSOONCCOCCCNCNNNCCCCCCCSOCCCCN", amap)
        ),
    )


def test_pdbloader_species_cmap_toX(testdata, testdir):

    # Map all elements to dummy atom
    cmap = {"X": ["C", "N", "O", "S", "P"]}

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir, cmap)

    # TODO: Access to data loader is quite ugly... NamedTuple?
    assert np.allclose(
        data[0][2][0], np.zeros(28),  # Species for first ligand  # Element X maps to 0
    )

    assert np.allclose(
        data[1][2][0], np.zeros(42),  # Species for second ligand  # Element X maps to 0
    )

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    assert len(amap) == 1

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    ids, labels, (species, coordinates) = next(iloader)

    assert (ids == np.array(["1a4r", "1a4w"])).all()

    assert species.shape == (batch_size, 42)  # Ligand 1a4w is the largest

    # Test ligand 1a4r (padded with -1)
    assert torch.allclose(species[0, :], torch.tensor([0] * 28 + 14 * [-1]),)

    # Test ligand 1a4w (no padding)
    assert torch.allclose(species[1, :], torch.zeros(42, dtype=int),)


def test_pdbloader_species_cmap_OtoS(testdata, testdir):

    # Map all elements to dummy atom
    cmap = {"S": "O"}

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(testdata, 0.1, testdir, cmap)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    # Check O is not in amap
    with pytest.raises(KeyError):
        amap[8]

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    ids, labels, (species, coordinates) = next(iloader)

    assert (ids == np.array(["1a4r", "1a4w"])).all()

    assert species.shape == (batch_size, 42)  # Ligand 1a4w is the largest

    # Test ligand 1a4r (padded with -1)
    assert torch.allclose(
        species[0, :],
        torch.tensor(
            elements_to_idxs("NPSSSPSSSCCSCSCSCNCNCCSNCNNC", amap) + 14 * [-1]
        ),
    )

    # Test ligand 1a4w (no padding)
    assert torch.allclose(
        species[1, :],
        torch.tensor(
            elements_to_idxs("CCCCCCCCCCNCCSSSNCCSCCCNCNNNCCCCCCCSSCCCCN", amap)
        ),
    )


@pytest.mark.skip(reason="To be implemented")
def test_pdbloader_ligand_coordinates(testdata):
    pass
