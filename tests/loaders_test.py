import os

import MDAnalysis as mda
import numpy as np
import pytest
import qcelemental as qcel
import torch
from openbabel import pybel

from ael import loaders


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
@pytest.mark.parametrize(
    "system, n_ligand, n_receptor", [("1a4r", 28, 6009), ("1a4w", 42, 4289)]
)
def test_load_mols(testdir, system, n_ligand, n_receptor, ext):

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    ulig = mda.Universe(os.path.join(testdir, ligname))
    urec = mda.Universe(os.path.join(testdir, recname))

    assert len(ulig.atoms) == n_ligand
    assert len(urec.atoms) == n_receptor

    systems = loaders.load_mols(ligname, recname, testdir)

    assert len(systems) == 1
    system = systems[0]

    assert len(system.atoms) == n_ligand + n_receptor

    lig = system.select_atoms("resname LIG")

    assert len(lig.atoms) == n_ligand
    assert set(lig.resnames) == set(["LIG"])
    assert set(lig.resnums) == set([1])
    assert set(lig.resids) == set([1])
    assert set(lig.record_types) == set(["HETATM"])
    assert set(lig.segids) == set([""])


def test_load_mols_fail_lig(testdir):

    system = "1a4r"

    ligname = os.path.join(system, f"{system}_FAIL.pdb")
    recname = os.path.join(system, f"{system}_protein.pdb")

    with pytest.raises(Exception):
        system = loaders.load_mols(ligname, recname, testdir)


def test_load_mols_fail_rec(testdir):

    system = "1a4r"

    ligname = os.path.join(system, f"{system}_ligand.pdb")
    recname = os.path.join(system, f"{system}_FAIL.pdb")

    with pytest.raises(Exception):
        system = loaders.load_mols(ligname, recname, testdir)


@pytest.mark.parametrize("ext", ["sdf", "mol2"])
@pytest.mark.parametrize("system, n_atoms", [("1a4r", 36), ("1a4w", 48)])
def test_universe_from_openbabel(testdir, system, n_atoms, ext):
    ligfile = os.path.join(testdir, system, f"{system}_docking.{ext}")

    obmols = [obmol for obmol in pybel.readfile(ext, ligfile)]

    assert len(obmols) == 9

    for obmol in obmols:
        u = loaders._universe_from_openbabel(obmol)

        assert len(u.atoms) == n_atoms

        for idx, atom in enumerate(obmol):
            assert np.allclose(u.atoms.positions[idx], atom.coords)
            assert qcel.periodictable.to_Z(u.atoms.elements[idx]) == atom.atomicnum
            assert u.atoms.elements[idx] == u.atoms.types[idx]
            assert u.atoms.resnames[idx] == "LIG"


@pytest.mark.parametrize("ext", ["sdf", "mol2"])
@pytest.mark.parametrize(
    "system, n_ligand, n_receptor", [("1a4r", 36, 6009), ("1a4w", 48, 4289)]
)
def test_load_mols_multiple(testdir, system, n_ligand, n_receptor, ext):

    ligname = os.path.join(system, f"{system}_docking.{ext}")
    ligfile = os.path.join(testdir, ligname)

    recname = os.path.join(system, f"{system}_protein.pdb")

    obmols = [obmol for obmol in pybel.readfile(ext, ligfile)]

    systems = loaders.load_mols(ligname, recname, testdir)

    assert len(systems) == 9

    for system, obmol in zip(systems, obmols):
        assert len(system.atoms) == n_ligand + n_receptor

        lig = system.select_atoms("resname LIG")

        assert len(lig.atoms) == n_ligand

        # Check ligand coordinates
        for idx, atom in enumerate(obmol):
            assert np.allclose(lig.atoms.positions[idx], atom.coords)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_select(testdir, system, distance, n_ligand, n_receptor, ext):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols(ligname, recname, testdir)

    assert len(systems) == 1
    system = systems[0]

    atoms, coordinates = loaders.select(system, distance)

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_select_ligmask(testdir, system, distance, n_ligand, n_receptor, ext):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols(ligname, recname, testdir)

    assert len(systems) == 1
    system = systems[0]

    atoms, coordinates, ligmask = loaders.select(system, distance, ligmask=True)

    assert len(ligmask) == n_ligand + n_receptor
    assert sum(ligmask) == n_ligand
    assert sum(system.atoms.resnames == "LIG") == sum(ligmask)

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_select_removeHs(testdir, system, distance, n_ligand, n_receptor, ext):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols(ligname, recname, testdir)

    assert len(systems) == 1
    system = systems[0]

    atoms, coordinates = loaders.select(system, distance, removeHs=True)

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_select_removeHs_ligmask(testdir, system, distance, n_ligand, n_receptor, ext):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols(ligname, recname, testdir)

    assert len(systems) == 1
    system = systems[0]

    atoms, coordinates, ligmask = loaders.select(
        system, distance, removeHs=True, ligmask=True
    )

    assert len(atoms) == len(ligmask)
    assert sum(ligmask) == sum(system.atoms.resnames == "LIG") == n_ligand

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_load_mols_and_select(testdir, system, distance, n_ligand, n_receptor, ext):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols_and_select(ligname, recname, distance, testdir)

    assert len(systems) == 1

    atoms, coordinates = systems[0]

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_load_mols_and_select_ligmask(
    testdir, system, distance, n_ligand, n_receptor, ext
):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols_and_select(
        ligname, recname, distance, testdir, ligmask=True
    )

    assert len(systems) == 1

    atoms, coordinates, ligmask = systems[0]

    assert len(ligmask) == n_ligand + n_receptor
    assert sum(ligmask) == n_ligand

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["pdb", "mol2"])
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
def test_load_mols_and_select_removeHs(
    testdir, system, distance, n_ligand, n_receptor, ext
):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_ligand.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    systems = loaders.load_mols_and_select(
        ligname, recname, distance, testdir, removeHs=True
    )

    assert len(systems) == 1
    atoms, coordinates = systems[0]

    assert len(atoms) == n_ligand + n_receptor
    assert coordinates.shape == (n_ligand + n_receptor, 3)


@pytest.mark.parametrize("ext", ["sdf", "mol2"])
@pytest.mark.parametrize(
    "system, distance, n_ligand",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        ("1a4r", 0.1, 28),
        ("1a4w", 0.1, 42),
    ],
)
def test_load_mols_and_select_removeHs_multiple(
    testdir, system, distance, n_ligand, ext
):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_docking.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    atoms_and_coordinates = loaders.load_mols_and_select(
        ligname, recname, distance, testdir, removeHs=True
    )

    for atoms, coordinates in atoms_and_coordinates:
        assert len(atoms) == n_ligand
        assert coordinates.shape == (n_ligand, 3)


@pytest.mark.parametrize("ext", ["sdf", "mol2"])
@pytest.mark.parametrize(
    "system, distance, n_ligand",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        ("1a4r", 0.1, 28),
        ("1a4w", 0.1, 42),
    ],
)
def test_load_mols_and_select_removeHs_ligmask_multiple(
    testdir, system, distance, n_ligand, ext
):
    """
    Selection compared with PyMol selection:

        sele byres PROTEIN within DISTANCE of LIGAND

    Hydrogen atoms were counted by hand and removed
    (H* does not select 1HD1 while *H* also selects CH2).
    """

    ligname = os.path.join(system, f"{system}_docking.{ext}")
    recname = os.path.join(system, f"{system}_protein.pdb")

    atoms_and_coordinates = loaders.load_mols_and_select(
        ligname,
        recname,
        distance,
        testdir,
        removeHs=True,
        ligmask=True,
    )

    for atoms, coordinates, ligmask in atoms_and_coordinates:
        assert len(ligmask) == n_ligand
        assert sum(ligmask) == n_ligand
        assert ligmask.all()

        assert len(atoms) == n_ligand
        assert coordinates.shape == (n_ligand, 3)


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
    "distance, n_atoms, f_label, l_label",
    [
        # Distance 0.0 produces a segmentation fault (see #2656)
        (0.1, [36 + 0, 48 + 0], [78.490, 69.210], [12.34, 43.21]),
    ],
)
def test_vsloader(testvsdata, testdir, distance, n_atoms, f_label, l_label):

    data = loaders.VSData(testvsdata, distance, testdir, labelspath=testdir)

    # One batch here corresponds to one target
    batch_size = 10

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    n_atoms_iter = iter(n_atoms)
    f_label_iter = iter(f_label)  # Iterator over first label (in batch)
    l_label_iter = iter(l_label)  # Iterator over last label (in batch)

    for ids, label, (species, coordinates) in iloader:

        n_atoms = next(n_atoms_iter)
        f_label = next(f_label_iter)
        l_label = next(l_label_iter)

        assert isinstance(ids, np.ndarray)
        assert ids.shape == (batch_size,)
        assert ids[0][4:] == "_pose_1"
        assert ids[-2][4:] == "_pose_9"
        assert ids[-1][4:] == "_ligand"

        assert isinstance(label, torch.Tensor)
        assert label.shape == (batch_size,)
        assert label[0].item() == pytest.approx(f_label)
        assert label[-1].item() == pytest.approx(l_label)

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
        data[0][2][0],
        np.zeros(28),  # Species for first ligand  # Element X maps to 0
    )

    assert np.allclose(
        data[1][2][0],
        np.zeros(42),  # Species for second ligand  # Element X maps to 0
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
    assert torch.allclose(
        species[0, :],
        torch.tensor([0] * 28 + 14 * [-1]),
    )

    # Test ligand 1a4w (no padding)
    assert torch.allclose(
        species[1, :],
        torch.zeros(42, dtype=int),
    )


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


def test_pdbloader_ligand_coordinates(testdata, testdir):
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
    assert coordinates.shape == (batch_size, 42, 3)  # Ligand 1a4w is the largest

    assert np.allclose(coordinates[0, 0], [102.486, 24.870, -2.909])
    assert np.allclose(coordinates[0, -1], [0.0, 0.0, 0.0])  # 1a4r is padded

    assert np.allclose(coordinates[1, 0], [17.735, -17.178, 22.612])
    assert np.allclose(coordinates[1, -1], [18.049, -13.554, 14.106])


def test_pdbloader_ligmasks(testdata, testdir):
    data = loaders.PDBData(testdata, 3.5, testdir, ligmask=True)

    batch_size = 2

    # Transform atomic numbers to species
    amap = loaders.anummap(data.species)
    data.atomicnums_to_idxs(amap)

    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=loaders.pad_collate
    )
    iloader = iter(loader)

    ids, labels, (species, coordinates, ligmasks) = next(iloader)

    assert (ids == np.array(["1a4r", "1a4w"])).all()

    assert ligmasks[0].sum() == 28
    assert ligmasks[1].sum() == 42

    # 1a4r_lignad.pdb
    assert coordinates[0, ligmasks[0]].shape == (28, 3)
    assert np.allclose(coordinates[0, ligmasks[0]][0, :], [102.486, 24.870, -2.909])
    assert np.allclose(coordinates[0, ligmasks[0]][-1, :], [104.205, 34.323, -1.866])

    # 1a4w_lignad.pdb
    assert coordinates[1, ligmasks[1]].shape == (42, 3)
    assert np.allclose(coordinates[1, ligmasks[1]][0, :], [17.735, -17.178, 22.612])
    assert np.allclose(coordinates[1, ligmasks[1]][-1, :], [18.049, -13.554, 14.106])
