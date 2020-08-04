import mlflow
import numpy as np
import torch


def predict(model, AEVC, loader, baseline=None, device=None):
    """
    Binding affinity predictions.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network
    AEVC: torchani.AEVComputer
        Atomic environment vector computer
    loader:
        Data loader
    baseline: Tuple[np.ndarray, np.ndarray, , np.ndarray]
        Baseline for delta learning (PDB IDs, Vina, logK)
    device: torch.device
        Computation device

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        System identifiers, true valudes and predicted values

    Notes
    -----
    The baseline for ∆-learning consists in the Autodock Vina score.
    It is passed together with the corresponding PDB IDs and the
    experimental :math:`\\log(K)`.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model.to(device)

    # Put model in evaluation mode
    model.eval()

    true = []
    predictions = []
    identifiers = []

    if baseline is not None:
        baseline_ids, baseline_values, logK = baseline

    with torch.no_grad():  # Turn off gradient computation during inference
        for ids, labels, (species, coordinates) in loader:

            # Move data to device
            labels = labels.to(device)
            species = species.to(device)
            coordinates = coordinates.to(device)

            # Compute AEV
            aevs = AEVC.forward((species, coordinates)).aevs

            # Forward pass
            output = model(species, aevs)

            # Transform output and labels to numpy arrays
            output = output.cpu().numpy()
            labels = labels.cpu().numpy()

            if baseline is None:
                # Store predicted values
                predictions += output.tolist()

                # Store true values
                true += labels.tolist()
            else:  # Delta learning
                # Store predicted values (delta) plus baseline
                # This corresponds to the final prediction

                # Mask baseline_ids with prediction identifiers
                # This allows to make b_ids and ids identical when sorted
                mask = np.isin(baseline_ids, ids)

                # Select relevant baseline
                b_ids = baseline_ids[mask]
                b_vals = baseline_values[mask]
                b_logK = logK[mask]

                # Sort baseline values according to IDs
                bsort = np.argsort(b_ids)
                b_vals = b_vals[bsort]
                b_logK = b_logK[bsort]

                # Sort output values according to IDs
                outsort = np.argsort(ids)
                ids = ids[outsort]  # IDs
                output = output[outsort]  # Deltas

                # Compute final predicitons: output plus baseline
                predictions += (output + b_vals).tolist()

                # True values are stored in baseline file
                # The labels are deltas, not true values
                true += b_logK.tolist()

            # Store systems identifiers
            identifiers += ids.tolist()

    # TODO: Work with numpy array directly instead of lists
    return np.array(identifiers), np.array(true), np.array(predictions)


if __name__ == "__main__":

    import argparse as ap

    import os

    import pandas as pd
    import json

    from ael import loaders, utils, plot

    from torch.utils import data

    parser = ap.ArgumentParser(description="Affinity prediction.")

    parser.add_argument("experiment", type=str, help="MLFlow experiment")

    parser.add_argument("testfile", type=str, help="Test set file")

    # TODO: Multiple models for consensus scoring
    parser.add_argument("-m", "--model", type=str, default="best.pth", help="Model")
    parser.add_argument("-e", "--aev", type=str, default="aevc.pth", help="Model")
    parser.add_argument(
        "-am", "--amap", type=str, default="amap.json", help="Atomic mapping to indices"
    )
    parser.add_argument(
        "-cm", "--chemap", type=str, default="cmap.json", help="Chemical mapping"
    )

    parser.add_argument(
        "-t", "--trainfile", type=str, default=None, help="Training set file"
    )
    parser.add_argument(
        "-v", "--validfile", type=str, default=None, help="Validation set file"
    )

    parser.add_argument("-d", "--datapaths", type=str, default="", help="Path to data")

    parser.add_argument(
        "-r", "--distance", type=float, default=0.1, help="Residue selection distance"
    )  # TODO: Read from test output file

    parser.add_argument("-b", "--batchsize", type=int, default=64, help="Batch size")

    parser.add_argument("-o", "--outpath", type=str, default="", help="Output path")

    parser.add_argument("--device", type=str, default=None, help="Device")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    mlflow.set_experiment(args.experiment)

    # Start MLFlow run (named predict)
    with mlflow.start_run(run_name="predict"):

        mlflow.log_param("device", args.device)

        mlflow.log_param("distance", args.distance)
        mlflow.log_param("testfile", args.testfile)
        mlflow.log_param("datapaths", args.datapaths)

        mlflow.log_param("batchsize", args.batchsize)

        with open(args.chemap, "r") as fin:
            cmap = json.load(fin)

        testdata = loaders.PDBData(
            args.testfile, args.distance, args.datapaths, cmap, desc="Test set"
        )

        amap = utils.load_amap(args.amap)

        testdata.atomicnums_to_idxs(amap)

        n_species = len(amap)

        mlflow.log_param("n_species", n_species)

        testloader = data.DataLoader(
            testdata,
            batch_size=args.batchsize,
            shuffle=False,
            collate_fn=loaders.pad_collate,
        )

        AEVC = utils.loadAEVC(args.aev)

        model = utils.loadmodel(args.model)

        ids_test, true_test, predicted_test = predict(model, AEVC, testloader)

        plot.regplot(true_test, predicted_test, "test", path=args.outpath)

        results_test = {
            "ture": pd.Series(index=ids_test, data=true_test),
            "predicted": pd.Series(index=ids_test, data=predicted_test),
        }

        df_test = pd.DataFrame(results_test)

        test_csv = os.path.join(args.outpath, "test.csv")
        df_test.to_csv(test_csv)

        # TODO
        # if args.trainfile is not None:
        # true_train, predicted_train = utils.predict(model, AEVC, trainloader)
        # plot.regplot(true_train, predicted_train, name="train")

        # TODO
        # if args.validfile is not None:
        # true_valid, predicted_valid = utils.predict(model, AEVC, validloader)
        # plot.regplot(true_valid, predicted_valid, name="valid")
