import os
import warnings

import MDAnalysis as mda
import numpy as np
import qcelemental as qcel
import torch
import torch.nn as nn
import tqdm
from torch.utils import data


def load_pdbs(ligand: str, receptor: str, datapaths):

    if isinstance(datapaths, str):
        # Make list
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

    lig = ulig.select_atoms("all")
    rec = urec.select_atoms("all")

    # Rename ligand consistently
    lig.residues.resnames = "LIG"

    # Merge receptor and ligand in single universe
    system = mda.core.universe.Merge(lig, rec)

    return system


def select(system, distance: float):
    resselection = system.select_atoms(
        f"(byres (around {distance} (resname LIG))) or (resname LIG)"
    )

    # Elements from PDB file needs MDAnalysis@develop (see #2648)
    return resselection.elements, resselection.positions


def load_pdbs_and_select(ligand: str, receptor: str, distance: float, datapaths):
    system = load_pdbs(ligand, receptor, datapaths)

    return select(system, distance)


def elements_to_atomicnums(elements):
    atomicnums = np.zeros(len(elements), dtype=int)

    for idx, e in enumerate(elements):
        atomicnums[idx] = qcel.periodictable.to_Z(e)

    return atomicnums


def pad_collate(batch, species_pad_value=-1, coords_pad_value=0, device=None):
    """

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


def anummap(*args):
    unique_atomicnums = set()

    for species in args:
        for s in species:
            unique_atomicnums.update(np.unique(s))

    return {anum: idx for idx, anum in enumerate(unique_atomicnums)}


class PDBData(data.Dataset):
    def __init__(self, fname, distance, datapaths="", desc=None):

        super().__init__()

        if desc is None:
            desc = "Loading PDB data"

        self.species = []
        self.coordinates = []
        self.labels = []

        self.ids = []

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

        self.ids = np.array(self.ids, dtype="U4")

        self.n = len(self.labels)

        self.species_are_indices = False

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            self.ids[idx],
            self.labels[idx],
            (self.species[idx], self.coordinates[idx]),
        )

    def atomicnums_to_idxs(self, atomicnums_map):

        if not self.species_are_indices:

            def anum_to_i(a):
                return atomicnums_map[a]

            v_anum_to_i = np.vectorize(anum_to_i)

            for idx in range(self.n):
                indices = v_anum_to_i(self.species[idx])
                self.species[idx] = torch.from_numpy(indices)
