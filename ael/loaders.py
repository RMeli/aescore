import os
import warnings
from typing import Collection, Dict, List, Optional, Tuple, Union

import MDAnalysis as mda
import numpy as np
import qcelemental as qcel
import torch
import torch.nn as nn
import tqdm
from torch.utils import data
from openbabel import pybel


def load_pdbs(
    ligand: str, receptor: str, datapaths: Union[str, List[str]]
) -> mda.Universe:
    """
    Load ligand and receptor PDB files in a single mda.Universe

    Parameters
    ----------
    ligand: str
        Ligand file
    receptor: str
        Receptor file
    datapaths: Union[str, List[str]]
        Paths to root directory ligand and receptors are stored

    Returns
    -------
    mda.Universe
        MDAnalysis universe for the protein-ligand complex

    Notes
    -----
    The ligand is treated as a single entity named LIG. (:code:`resname LIG`).

    The folders containing ligand and receptor files data are defined by
    :code:`datapaths`.
    """
    assert os.path.splitext(ligand)[-1].lower() == ".pdb"
    assert os.path.splitext(receptor)[-1].lower() == ".pdb"

    # Ensure list
    if isinstance(datapaths, str):
        datapaths = [datapaths]

    # TODO: Redirect warning instead of suppressing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Try to load ligand
        for path in datapaths:
            ligfile = os.path.join(path, ligand)
            if os.path.isfile(ligfile):
                try:
                    ulig = mda.Universe(ligfile)
                except Exception:
                    print(f"Problems loading {ligfile}")
                    raise

                break
        else:
            # Runs only if nothing is found
            raise RuntimeError(
                f"Could not find ligand file {ligfile} in {datapaths}..."
            )

        # Try to load receptor
        for path in datapaths:
            recfile = os.path.join(path, receptor)
            if os.path.isfile(recfile):
                try:
                    urec = mda.Universe(recfile)
                except Exception:
                    print(f"Problems loading {recfile}")
                    raise

                break
        else:
            # Runs only if nothing is found
            raise RuntimeError(
                f"Could not find receptor file {recfile} in {datapaths}..."
            )

    lig = ulig.select_atoms("all")
    rec = urec.select_atoms("all")

    # Rename ligand consistently
    lig.residues.resnames = "LIG"

    # Merge receptor and ligand in single universe
    system = mda.core.universe.Merge(lig, rec)

    return system


def _universe_from_openbabel(obmol):
    n_atoms = len(obmol.atoms)
    n_residues = 1  # LIG

    u = mda.Universe.empty(
        n_atoms,
        n_residues,
        atom_resindex=[0] * n_atoms,
        residue_segindex=[0] * n_residues,
        trajectory=True,
    )

    elements = []
    coordinates = np.zeros((n_atoms, 3))
    for idx, atom in enumerate(obmol):
        elements.append(qcel.periodictable.to_E(atom.atomicnum))
        coordinates[idx, :] = atom.coords

    u.add_TopologyAttr("elements", elements)
    u.add_TopologyAttr("type", elements)
    u.add_TopologyAttr("resname", ["LIG"] * n_residues)

    u.atoms.positions = coordinates

    return u


def load_mols(
    ligand: str, receptor: str, datapaths: Union[str, List[str]]
) -> List[mda.Universe]:
    """
    Load ligand and receptor PDB files in a single mda.Universe

    Parameters
    ----------
    ligand: str
        Ligand file (SDF)
    receptor: str
        Receptor file (PDB)
    datapaths: Union[str, List[str]]
        Paths to root directory ligand and receptors are stored

    Returns
    -------
    Lit[mda.Universe]
        MDAnalysis universe for the protein-ligand complex

    Notes
    -----
    This function allows to load multiple ligands from SDF files for a single receptor.
    This is useful for docking and virtual screening, where multiple ligands are
    associated to a single target.

    The ligand is treated as a single entity named LIG. (:code:`resname LIG`).

    The folders containing ligand and receptor files data are defined by
    :code:`datapaths`.
    """

    ext = os.path.splitext(ligand)[-1].lower()[1:]

    assert ext == "sdf" or ext == "mol2"
    assert os.path.splitext(receptor)[-1].lower() == ".pdb"

    # Ensure list
    if isinstance(datapaths, str):
        datapaths = [datapaths]

    # TODO: Redirect warning instead of suppressing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Try to load ligand
        for path in datapaths:
            ligfile = os.path.join(path, ligand)
            if os.path.isfile(ligfile):
                try:
                    uligs = [
                        _universe_from_openbabel(obmol)
                        for obmol in pybel.readfile(ext, ligfile)
                    ]
                except Exception:
                    print(f"Problems loading {ligfile}")
                    raise

                # Ligand file found in current path, no need to search further
                break
        else:
            raise RuntimeError(
                f"Could not find ligand file {ligfile} in {datapaths}..."
            )

        # Try to load receptor
        for path in datapaths:
            recfile = os.path.join(path, receptor)
            if os.path.isfile(recfile):
                try:
                    urec = mda.Universe(recfile)
                except Exception:
                    print(f"Problems loading {recfile}")
                    raise

                break
        else:
            raise RuntimeError(
                f"Could not find receptor file {recfile} in {datapaths}..."
            )

    ligs = [ulig.select_atoms("all") for ulig in uligs]
    rec = urec.select_atoms("all")

    # Merge receptor and ligand in single universe
    systems = [mda.core.universe.Merge(lig, rec) for lig in ligs]

    return systems


def select(
    system: mda.Universe, distance: float, removeHs: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select binding site.

    Parameters
    ---------
    system: mda.Universe
        Protein-ligand complex
    distance: float
        Ligand-residues distance
    removeHs: bool
        Remove hydrogen atoms

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Array of elements and array of cartesian coordinate for ligand and protein
        atoms within the binding site

    Notes
    -----
    The binding site is defined by residues with at least one atom within
    :code:`distance` from the ligand.
    """
    resselection = system.select_atoms(
        f"(byres (around {distance} (resname LIG))) or (resname LIG)"
    )

    if removeHs:
        mask = resselection.elements != "H"
        # Elements from PDB file needs MDAnalysis@develop (see #2648)
        return resselection.elements[mask], resselection.positions[mask]
    else:
        return resselection.elements, resselection.positions


def load_pdbs_and_select(
    ligand: str, receptor: str, distance: float, datapaths, removeHs: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PDB files and select binding site.

    Parameters
    ----------
    ligand: str
        Ligand file
    receptor: str
        Receptor file
    distance: float
        Ligand-residues distance
    datapaths: Union[str, List[str]]
        Paths to root directory ligand and receptors are stored
    removeHs: bool
        Remove hydrogen atoms

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Array of elements and array of cartesian coordinate for ligand and protein
        atoms within the binding site

    Notes
    -----
    Combines :func:`load_pdbs` and :func:`select`.
    """
    system = load_pdbs(ligand, receptor, datapaths)

    return select(system, distance, removeHs=removeHs)


def load_mols_and_select(
    ligand: str, receptor: str, distance: float, datapaths, removeHs: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load PDB files and select binding site.

    Parameters
    ----------
    ligand: str
        Ligand file (SDF)
    receptor: str
        Receptor file (PDB)
    distance: float
        Ligand-residues distance
    datapaths: Union[str, List[str]]
        Paths to root directory ligand and receptors are stored
    removeHs: bool
        Remove hydrogen atoms

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Array of elements and array of cartesian coordinate for ligand and protein
        atoms within the binding site

    Notes
    -----
    Combines :func:`load_pdbs` and :func:`select`.
    """
    systems = load_mols(ligand, receptor, datapaths)

    return [select(system, distance, removeHs=removeHs) for system in systems]


def elements_to_atomicnums(elements: Collection[int]) -> np.ndarray:
    """
    Convert element symbols to atomic numbers.

    Parameters
    ----------
    elements: Iterable
        Iterable object with lement symbols

    Returns
    -------
    np.ndarray
        Array of atomic numbers
    """
    atomicnums = np.zeros(len(elements), dtype=int)

    for idx, e in enumerate(elements):
        atomicnums[idx] = qcel.periodictable.to_Z(e)

    return atomicnums


def pad_collate(
    batch,
    species_pad_value=-1,
    coords_pad_value=0,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Collate function to pad batches.

    Parameters
    ----------
    batch:
        Batch
    species_pad_value:
        Padding value for species vector
    coords_pad_value:
        Padding value for coordinates
    device: Optional[Union[str, torch.device]]
        Computation device

    Returns
    -------
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

    Notes
    -----

    :code:`torch.utils.data.Dataset`'s :code:`__getitem__` returns a batch that need
    to be padded. This function can be passed to :code:`torch.utils.data.DataLoader`
    as :code:`collate_fn` argument in order to pad species and coordinates
    """

    ids, labels, species_and_coordinates = zip(*batch)
    species, coordinates = zip(*species_and_coordinates)

    pad_species = nn.utils.rnn.pad_sequence(
        species, batch_first=True, padding_value=species_pad_value
    )
    pad_coordinates = nn.utils.rnn.pad_sequence(
        coordinates, batch_first=True, padding_value=coords_pad_value
    )

    return np.array(ids), torch.tensor(labels), (pad_species, pad_coordinates)


def anummap(*args) -> Dict[int, int]:
    """
    Map atomic numbers to zero-based numbers.

    Returns
    -------
    Dict[int, int]
        Mapping between atomic numbers and indices
    """
    unique_atomicnums = set()

    for species in args:
        for s in species:
            unique_atomicnums.update(np.unique(s))

    return {anum: idx for idx, anum in enumerate(unique_atomicnums)}


def _anum_to_idx(anum: int, amap: Dict[int, int]) -> int:
    """
    Convert atomic number to index.

    Parameters
    ----------
    anum: int
        Atomic number
    amap:
        Map atomic numbers to zero-based indices

    Returns
    -------
    int
        Zero-based index for the given atomic number
    """
    try:
        return amap[anum]
    except KeyError:
        raise KeyError(
            f"Atomic number {anum} ({qcel.periodictable.to_E(anum)}) "
            "not included in {amap}."
        )


# Numpy vectorisation
anum_to_idx = np.vectorize(_anum_to_idx)


def chemap(
    atomicnums: List[np.ndarray], cmap: Union[Dict[str, str], Dict[str, List[str]]]
):
    """
    Map chemical elements into another.

    Parameters
    ----------
    atomicnum: List[np.ndarray]
        List of atomic numbers for every system
    chemap: Union[Dict[str, str],Dict[str, List[str]]
        Chemical mapping

    Notes
    -----
    This function can be used to map different elements into a single one. For
    example, map Se to S to avoid selenoproteins or map all metal atoms to a
    dummy atom X.

    :code:`species` is modified in-place.
    """
    n = len(atomicnums)

    dummy = "X"  # Element symbol for a dummy atom

    # Transform map from element symbols to atomic number
    cmapZ = {}
    for to_element, from_elements in cmap.items():
        if isinstance(from_elements, list):
            # Transform element symbols into atomic numbers
            from_elements = [qcel.periodictable.to_Z(e) for e in from_elements]
        else:
            # Transform element symbol into atomic number
            # Make a list for consistency
            from_elements = [qcel.periodictable.to_Z(from_elements)]

        # Transform destination element symbol into atomic number
        if to_element != dummy:
            cmapZ[qcel.periodictable.to_Z(to_element)] = from_elements
        else:
            # Dummy atom X is mapped to 0
            cmapZ[0] = from_elements

    # Apply map to all the atomicnums
    # TODO: Refactor to make this faster
    for idx in range(n):
        for to_element, from_elements in cmapZ.items():
            mask = np.isin(atomicnums[idx], from_elements)
            atomicnums[idx][mask] = to_element


class Data(data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        # TODO: Better way to avoid mypy complaints?
        self.n: int = -1
        self.ids: List[str] = []
        self.labels: List[float] = []
        self.species: List[torch.Tensor] = []
        self.coordinates: List[torch.Tensor] = []
        self.species_are_indices: bool = False

    def __len__(self) -> int:
        """
        Number of protein-ligand complexes in the dataset.

        Returns
        -------
        int
            Dataset length
        """
        return self.n

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get item from dataset.

        Parameters
        ----------
        idx: int
            Item index within the dataset

        Returns
        -------
        Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]
            Item from the dataset (PDB IDs, labels, species, coordinates)
        """
        return (
            self.ids[idx],
            self.labels[idx],
            (self.species[idx], self.coordinates[idx]),
        )

    def _chemap(self, cmap: Union[Dict[str, str], Dict[str, List[str]]]):
        """
        Map chemical species into another.

        Parameters
        ----------
        chemap: Union[Dict[str, str],Dict[str, List[str]]
            Chemical mapping

        Notes
        -----
        This function can be used to map different elements into a single one. For
        example, map Se to S to avoid selenoproteins or map all metal atoms to a
        dummy atom X.
        """
        if self.species_are_indices:
            raise RuntimeError("Species are indices. CHEMAP can't be computed.")

        # Compute mapping
        chemap(self.species, cmap)

    def atomicnums_to_idxs(self, atomicnums_map: Dict[int, int]) -> None:
        """
        Convert atomic numbers to indices.

        Parameters
        ----------
        atomicnums_map: Dict[int, int]
            Map of atomic number to indices

        Notes
        -----
        This function converts :attr:`species` from :code:`np.array` to
        :code:`torch.tensor`.

        The :code`atomicnums_map` needs to be shared between train/validation/test
        datasets for consistency.
        """

        if not self.species_are_indices:
            for idx in range(self.n):
                indices = anum_to_idx(self.species[idx], atomicnums_map)
                self.species[idx] = torch.from_numpy(indices)

            self.species_are_indices = True


class PDBData(Data):
    """
    PDB dataset.

    Parameters
    ----------
    fname: str
        Data file name
    distance: float
        Ligand-residues distance
    datapaths: Union[str, List[str]]
        Paths to root directory ligand and receptors are stored
    cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]]
        Chemical mapping
    desc: Optional[str]
        Dataset description (for :mod:`tqdm`)
    removeHs: bool
        Remove hydrogen atoms

    Notes
    -----
    The data file contains the label in the first column, the protein file name in
    the second column and the ligand file name in the third column.
    """

    def __init__(
        self,
        fname: str,
        distance: float,
        datapaths: Union[str, List[str]] = "",
        cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        desc: Optional[str] = None,
        removeHs: bool = False,
    ):

        super().__init__()

        self._load(fname, distance, datapaths, cmap, desc, removeHs)

    def _load(
        self,
        fname: str,
        distance: float,
        datapaths: Union[str, List[str]] = "",
        cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        desc: Optional[str] = None,
        removeHs: bool = False,
    ) -> None:

        super().__init__()

        if desc is None:
            desc = "Loading PDB data"

        self.species = []
        self.coordinates = []
        self.labels = []

        self.ids = []

        self.cmap = cmap

        with open(fname, "r") as f:
            for line in tqdm.tqdm(f, desc=desc):
                label, recfile, ligfile = line.split()

                self.ids.append(os.path.dirname(recfile))

                self.labels.append(float(label))

                els, coords = load_pdbs_and_select(
                    ligfile, recfile, distance, datapaths, removeHs=removeHs
                )

                atomicnums = elements_to_atomicnums(els)

                # Species are converted to tensors in _atomicnums_to_idx
                # Species are transformed to 0-based indices in _atomicnums_to_idx
                self.species.append(atomicnums)

                # Coordinates are transformed to tensor here and left unchanged
                self.coordinates.append(torch.from_numpy(coords))

        self.labels = np.array(self.labels, dtype=np.float32)
        self.n = len(self.labels)

        self.ids = np.array(self.ids, dtype="U4")

        self.species_are_indices = False

        # Map one element into another
        # This allows to reduce the complexity of the model
        if cmap is not None:
            self._chemap(cmap)


class VSData(Data):
    """
    Dataset for docking and virtual screening.

    Parameters
    ----------
    fname: str
        Data file name
    distance: float
        Ligand-residues distance
    datapaths: Union[str, List[str]]
        Paths to root directory ligand and receptors are stored
    cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]]
        Chemical mapping
    desc: Optional[str]
        Dataset description (for :mod:`tqdm`)
    removeHs: bool
        Remove hydrogen atoms
    labelspath: str
        Path to labels files
    idsuffix:
        Suffix to add to the PDB ID (from the receptor name)

    Notes
    -----
    The data file contains the file with labels in the first column, the protein file
    name in the second column and the ligand file name in the third column.

    The ligand file is assumed to be a SDF file with one or multiple poses, for docking
    or virtual screening tasks. All poses are against the same target, specified in
    the second column.

    :code:`idsuffix` allows to specify a suffix to the PDB ID extracted from the
    receptor. This is used when a line in :code:`fname` does not contain a label
    file with multiple systems (decoys) but a single numerical label corresponding to
    a single system (usually the active molecule or crystallographic pose).
    """

    def __init__(
        self,
        fname: str,
        distance: float,
        datapaths: Union[str, List[str]] = "",
        cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        desc: Optional[str] = None,
        removeHs: bool = False,
        labelspath: str = "",
        idsuffix="ligand",
    ):

        super().__init__()

        self._load(
            fname, distance, datapaths, cmap, desc, removeHs, labelspath, idsuffix
        )

    def _load(
        self,
        fname: str,
        distance: float,
        datapaths: Union[str, List[str]] = "",
        cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        desc: Optional[str] = None,
        removeHs: bool = False,
        labelspath: str = "",
        idsuffix: str = "ligand",
    ) -> None:

        super().__init__()

        if desc is None:
            desc = "Loading PDB data"

        self.species = []
        self.coordinates = []
        self.labels = []

        self.ids = []

        self.cmap = cmap

        with open(fname, "r") as f:
            for line in tqdm.tqdm(f, desc=desc):
                labelfile, recfile, ligfile = line.split()

                pdbid = os.path.dirname(recfile)

                # Support mixed file or numerical label
                try:
                    labels = [float(labelfile)]
                    ids = [idsuffix]

                    systems = [
                        load_pdbs_and_select(
                            ligfile, recfile, distance, datapaths, removeHs=removeHs
                        )
                    ]

                except ValueError:  # labelfile contains a file path, not a label
                    ids = np.loadtxt(
                        os.path.join(labelspath, labelfile), usecols=0, dtype="U"
                    )
                    labels = np.loadtxt(os.path.join(labelspath, labelfile), usecols=1)
                    systems = load_mols_and_select(
                        ligfile, recfile, distance, datapaths, removeHs=removeHs
                    )

                assert len(ids) == len(labels) == len(systems)

                for ID, label, (els, coords) in zip(ids, labels, systems):
                    if pdbid in ID:  # PDB code of target already in ID (pose name)
                        self.ids.append(ID)
                    else:  # Prepend PDB code of target to avoid ambiguities
                        self.ids.append(f"{pdbid}_{ID}")

                    self.labels.append(label)

                    atomicnums = elements_to_atomicnums(els)

                    # Species are converted to tensors in _atomicnums_to_idx
                    # Species are transformed to 0-based indices in _atomicnums_to_idx
                    self.species.append(atomicnums)

                    # Coordinates are transformed to tensor here and left unchanged
                    self.coordinates.append(torch.from_numpy(coords))

        self.labels = np.array(self.labels, dtype=np.float32)
        self.n = len(self.labels)

        self.ids = np.array(self.ids, dtype="U")

        self.species_are_indices = False

        # Map one element into another
        # This allows to reduce the complexity of the model
        if cmap is not None:
            self._chemap(cmap)
