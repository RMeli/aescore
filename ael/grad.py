import torch
from torch import nn


def gradient(species, coordinates, label, model, AEVC, loss, device=None):
    """
    Compute gradient of the loss with respect to atomic coordinates.

    Parameters
    ----------
    species: np.array
        Atomic species (mapped to indices)
    coordinates: np.array
        Atomic coordinates
    label:
        Label for the current example
    model: torch.nn.Module
        Trained model
    AEVC: torchani.AEVComputer
        AEV computer
    loss:
        Loss function
    device:
        Computation device

    Returns
    -------
    np.array
        Gradient of the loss function with respect to atomic coordinates
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to device and add batch dimension
    label = torch.tensor(label).to(device).unsqueeze(0)
    species = species.to(device).unsqueeze(0)
    coordinates = (
        coordinates.clone().detach().requires_grad_(True).to(device).unsqueeze(0)
    )

    aevs = AEVC.forward((species, coordinates)).aevs

    output = model(species, aevs)

    #ls = loss(output, label)
    # Compute gradient of the loss with respect to the coordinates
    #grad = torch.autograd.grad(ls, coordinates)[0]

    # Compute gradient of the output with respect to the coordinates
    grad = torch.autograd.grad(output, coordinates)[0]

    # Remove batch dimension
    return grad.squeeze(0)


def atomic(species, coordinates, model, AEVC, device=None):
    """
    Compute atomic contributions.

    Parameters
    ----------
    species: np.array
        Atomic species (mapped to indices)
    coordinates: np.array
        Atomic coordinates
    model: torch.nn.Module
        Trained model
    AEVC: torchani.AEVComputer
        AEV computer
    device:
        Computation device

    Returns
    -------
    np.array
        Atomic contributions
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to device and add batch dimension
    species = species.to(device).unsqueeze(0)
    coordinates = (
        coordinates.clone().detach().requires_grad_(True).to(device).unsqueeze(0)
    )

    aevs = AEVC.forward((species, coordinates)).aevs

    atomic_contributions = model._forward_atomic(species, aevs)

    assert atomic_contributions.shape == species.shape

    return atomic_contributions


if __name__ == "__main__":

    import argparse as ap
    import json
    import os
    import warnings

    import numpy as np
    import tqdm

    from ael import loaders, utils

    parser = ap.ArgumentParser(description="Loss gradient.")

    parser.add_argument("gradfile", type=str, help="Gradient file")

    parser.add_argument(
        "-v",
        "--visualization",
        type=str,
        default="gradient",
        help="Visualisation method",
    )
    parser.add_argument("-s", "--scaling", type=int, default=1, help="Scaling factor")

    # TODO: Multiple models for consensus scoring
    parser.add_argument("-m", "--model", type=str, default="best.pth", help="Model")
    parser.add_argument("-e", "--aev", type=str, default="aevc.pth", help="Model")
    parser.add_argument(
        "-am", "--amap", type=str, default="amap.json", help="Atomic mapping to indices"
    )
    parser.add_argument(
        "-cm", "--chemap", type=str, default=None, help="Chemical mapping"
    )

    parser.add_argument("-d", "--datapaths", type=str, default="", help="Path to data")

    parser.add_argument(
        "-r", "--distance", type=float, default=3.5, help="Residue selection distance"
    )  # TODO: Read from training

    parser.add_argument("-o", "--outpath", type=str, default="", help="Output path")

    parser.add_argument("--device", type=str, default=None, help="Device")

    parser.add_argument("--removeHs", action="store_true", help="Remove hydrogen atoms")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.chemap is not None:
        with open(args.chemap, "r") as fin:
            cmap = json.load(fin)
    else:
        cmap = None

    # Load and apply atom to index mapping with amap
    amap = utils.load_amap(args.amap)
    n_species = len(amap)

    # Load AEVComputer and model
    AEVC = utils.loadAEVC(args.aev)
    model = utils.loadmodel(args.model)

    AEVC.to(device)
    model.to(device)

    with open(args.gradfile, "r") as f:
        for line in tqdm.tqdm(f):
            label, recfile, ligfile = line.split()

            systems = loaders.load_mols(ligfile, recfile, args.datapaths)

            # TODO: Allow multiple systems?
            assert len(systems) == 1
            system = systems[0]

            # Select ligand and residues
            # TODO: Unify selections
            selection = system.select_atoms(
                f"(byres (around {args.distance} (resname LIG))) or (resname LIG)"
            )

            if args.removeHs:
                mask = selection.elements != "H"

                sel_idxs = selection.ix[mask]

                elements = selection.elements[mask]
                coordinates = selection.positions[mask]
            else:
                # Store indices of selection for later use
                # Needed to insert gradient as b-factors
                sel_idxs = selection.ix

                elements = selection.elements
                coordinates = selection.positions

            # Get species for selection
            # Apply mapping from elements to atomic number
            # Apply chemical mapping (if any)
            # Apply mapping from atomic numbers to indices
            anums = loaders.elements_to_atomicnums(elements)
            if cmap is not None:
                loaders.chemap([anums], cmap)  # Only one system
            species = loaders.anum_to_idx(anums, amap)

            G = np.zeros(len(system.atoms))

            if args.visualization.lower() == "gradient":

                gradt = gradient(
                    torch.from_numpy(species),
                    torch.from_numpy(coordinates),
                    float(label),
                    model,
                    AEVC,
                    nn.MSELoss(),
                    device=args.device,
                )

                # Compute gradient norm
                G[sel_idxs] = gradt.norm(dim=1).cpu().numpy()
            elif args.visualization.lower() == "atomic":
                atomic_contributions = atomic(
                    torch.from_numpy(species),
                    torch.from_numpy(coordinates),
                    model,
                    AEVC,
                    device=args.device,
                )

                # Assign atomic contributions to corresponding atoms
                G[sel_idxs] = atomic_contributions.detach().cpu().numpy()
            else:
                print(f"Visualization method {args.visualization} not supported.")
                exit(-1)

            # Register gradient as temperature factor
            system.add_TopologyAttr("tempfactors", G * args.scaling)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ligfname = os.path.splitext(os.path.basename(ligfile))[0]
                system.select_atoms("resname LIG").write(
                    os.path.join(
                        args.outpath,
                        f"{ligfname}_{args.visualization}.pdb",
                    )
                )

                recfname = os.path.splitext(os.path.basename(recfile))[0]
                system.select_atoms("not resname LIG").write(
                    os.path.join(
                        args.outpath,
                        f"{recfname}_{args.visualization}.pdb",
                    )
                )
