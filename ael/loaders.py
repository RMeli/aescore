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

    # Ensure list
    if isinstance(datapaths, str):
        datapaths = [datapaths]

    # TODO: Redirect warning instead of suppressing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for path in datapaths:

            ligfile = os.path.join(path, ligand)
            if os.path.isfile(ligfile):
                try:
                    ulig = mda.Universe(ligfile)
                except Exception:
                    print(f"Problems loading {ligfile}")
                    raise

                recfile = os.path.join(path, receptor)
                try:
                    urec = mda.Universe(recfile)
                except Exception:
                    print(f"Problems loading {recfile}")
                    raise

                break
        else:
            # Runs only if nothing is found
            raise RuntimeError(
                f"Could not find ligand or receptor file in {datapaths}..."
            )

    lig = ulig.select_atoms("all")
    rec = urec.select_atoms("all")

    # Rename ligand consistently
    lig.residues.resnames = "LIG"

    # Merge receptor and ligand in single universe
    system = mda.core.universe.Merge(lig, rec)

    return system


def select(system: mda.Universe, distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select binding site.

    Parameters
    ---------
    system: mda.Universe
        Protein-ligand complex
    distance: float
        Ligand-residues distance

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

    # Elements from PDB file needs MDAnalysis@develop (see #2648)
    return resselection.elements, resselection.positions


def load_pdbs_and_select(
    ligand: str, receptor: str, distance: float, datapaths
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

    return select(system, distance)


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
    return amap[anum]


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


class PDBData(data.Dataset):
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

    Notes
    -----
    The data file contains the label in the first colum, the protein file name in
    the second column and the ligand file name in the third colum.
    """

    def __init__(
        self,
        fname: str,
        distance: float,
        datapaths: Union[str, List[str]] = "",
        cmap: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        desc: Optional[str] = None,
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

                # Labels are converted to tensors in pad_collate
                self.labels.append(float(label))

                els, coords = load_pdbs_and_select(
                    ligfile, recfile, distance, datapaths
                )

                atomicnums = elements_to_atomicnums(els)

                # Species are converted to tensors in _atomicnums_to_idx
                # Species are transformed to 0-based indices in _atomicnums_to_idx
                self.species.append(atomicnums)

                # Coordinates are transformed to tensor here and left unchanged
                self.coordinates.append(torch.from_numpy(coords))

        self.n = len(self.labels)

        self.ids = np.array(self.ids, dtype="U4")

        self.species_are_indices = False

        # Map one element into another
        # This allows to reduce the complexity of the model
        if cmap is not None:
            self._chemap(cmap)

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
