import os

import mlflow
import numpy as np
import torch
import tqdm

from ael import utils


def train(
    model,
    optimizer,
    loss_function,
    AEVC,  # torchani.AEVComputer
    trainloader,
    testloader,
    epochs=15,
    savepath=None,
    idx=None,
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log parameters
    mlflow.log_param("aev_size", AEVC.aev_length)
    mlflow.log_param("trainloader_size", len(trainloader))
    mlflow.log_param("validloader_size", len(testloader))
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("loss_fn", loss_function.__class__.__name__)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)

    # Move model to CUDA
    # Labels and densities are on the GPU
    model.to(device)

    train_losses, valid_losses = [], []

    best_valid_loss = np.inf
    best_epoch = 0

    # Loop over epochs
    for epoch in tqdm.trange(epochs, desc="Training"):

        # Put model in training mode
        model.train()

        # Initialize total epoch loss
        epoch_loss = 0

        # Training
        for _, labels, (species, coordinates) in trainloader:

            # Move everything to device
            labels = labels.to(device)
            species = species.to(device)
            coordinates = coordinates.to(device)

            aevs = AEVC.forward((species, coordinates)).aevs

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Perform forward pass
            output = model(species, aevs)

            # Compute the loss
            # TODO: Exponential loss function?
            loss = loss_function(output, labels)

            # Perform backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate total epoch loss
            epoch_loss += loss.item()

        else:
            valid_loss = 0

            # Put model in evaluation mode
            model.eval()

            # Validation
            # No need to track gradients during validation
            with torch.no_grad():
                for _, labels, (species, coordinates) in testloader:

                    # Move everything to device
                    labels = labels.to(device)
                    species = species.to(device)
                    coordinates = coordinates.to(device)

                    aevs = AEVC.forward((species, coordinates)).aevs

                    # Perform forward pass
                    output = model(species, aevs)

                    # Compute the loss
                    valid_loss += loss_function(output, labels).item()

            # Normalise loss
            epoch_loss /= len(trainloader)
            valid_loss /= len(testloader)

            if valid_loss < best_valid_loss and savepath is not None:
                # TODO: Save Optimiser
                if idx is None:
                    modelname = os.path.join(savepath, "best.pth")
                else:
                    modelname = os.path.join(savepath, f"best_{idx}.pth")

                utils.savemodel(model, modelname)

                best_epoch = epoch
                best_valid_loss = valid_loss

            # Store losses
            train_losses.append(epoch_loss)
            valid_losses.append(valid_loss)

            # Log losses
            if idx is None:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            else:
                mlflow.log_metric(f"train_loss_{idx}", epoch_loss, step=epoch)
                mlflow.log_metric(f"valid_loss_{idx}", valid_loss, step=epoch)

    # Save best model
    if best_epoch != 0:
        mlflow.log_artifact(modelname)

        if idx is None:
            mlflow.log_param("best _epoch", best_epoch)
        else:
            mlflow.log_param(f"best_epoch_{idx}", best_epoch)

    return train_losses, valid_losses


if __name__ == "__main__":

    from torch import optim, nn
    import torchani

    from torch.backends import cudnn
    from torch.utils import data

    from ael import loaders, models, plot, predict, argparsers

    import pandas as pd

    from matplotlib import pyplot as plt

    from typing import Optional, Tuple, Any

    args = argparsers.trainparser(default="BP")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True  # type: ignore
        cudnn.benchmark = False  # type: ignore

    mlflow.set_experiment(args.experiment)

    # Start MLFlow run (named train)
    with mlflow.start_run(run_name="train"):

        mlflow.log_param("device", device)
        mlflow.log_param("random_seed", args.seed)

        mlflow.log_param("distance", args.distance)
        mlflow.log_param("trainfile", args.trainfile)
        mlflow.log_param("validfile", args.validfile)
        mlflow.log_param("datapaths", args.datapaths)

        mlflow.log_param("batchsize", args.batchsize)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("layers", args.layers)
        mlflow.log_param("dropout", args.dropout)

        mlflow.log_param("consensus", args.consensus)

        traindata = loaders.PDBData(
            args.trainfile, args.distance, args.datapaths, desc="Training set"
        )
        validdata = loaders.PDBData(
            args.validfile, args.distance, args.datapaths, desc="Validation set"
        )

        # Get combined atomic numbers map
        if args.testfile is not None:
            testdata = loaders.PDBData(
                args.testfile, args.distance, args.datapaths, desc="Test set"
            )

            amap = loaders.anummap(
                traindata.species, validdata.species, testdata.species
            )
        else:
            amap = loaders.anummap(traindata.species, validdata.species)

        n_species = len(amap)

        # Transform atomic numbers to 0-based indices
        traindata.atomicnums_to_idxs(amap)
        validdata.atomicnums_to_idxs(amap)

        if args.testfile is not None:
            testdata.atomicnums_to_idxs(amap)

        # Save amap to JSON file
        utils.save_amap(amap, path=os.path.join(args.outpath, "amap.json"))

        mlflow.log_param("nspecies", n_species)

        trainloader = data.DataLoader(
            traindata,
            batch_size=args.batchsize,
            shuffle=True,
            collate_fn=loaders.pad_collate,
        )

        validloader = data.DataLoader(
            validdata,
            batch_size=args.batchsize,
            shuffle=True,
            collate_fn=loaders.pad_collate,
        )

        if args.testfile is not None:
            testloader = data.DataLoader(
                testdata,
                batch_size=args.batchsize,
                shuffle=False,
                collate_fn=loaders.pad_collate,
            )

        # Radial coefficients
        EtaR = torch.tensor([args.EtaR], device=device)
        RsR = torch.tensor(args.RsR, device=device)

        mlflow.log_param("RcR", args.RcR)
        mlflow.log_param("EtaR", args.EtaR)
        mlflow.log_param("RsR", args.RsR)

        # Angular coefficients
        RsA = torch.tensor(args.RsA, device=device)
        EtaA = torch.tensor([args.EtaA], device=device)
        TsA = torch.tensor(args.TsA, device=device)
        Zeta = torch.tensor([args.Zeta], device=device)

        mlflow.log_param("RcA", args.RcA)
        mlflow.log_param("RsA", args.RsA)
        mlflow.log_param("EtaA", args.EtaA)
        mlflow.log_param("TsA", args.TsA)
        mlflow.log_param("Zeta", args.Zeta)

        # Define AEVComputer
        AEVC = torchani.AEVComputer(
            args.RcR, args.RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species
        )

        # Save AEVComputer
        utils.saveAEVC(AEVC, n_species, path=os.path.join(args.outpath, "aevc.pth"))

        # Define models
        models_list = []
        optimizers_list = []
        for idx in range(args.consensus):
            models_list.append(
                models.AffinityModel(
                    n_species,
                    AEVC.aev_length,
                    layers_sizes=args.layers,
                    dropp=args.dropout,
                )
            )

            # Define optimizer
            optimizers_list.append(optim.Adam(models_list[-1].parameters(), lr=args.lr))

            # Define loss
            mse = nn.MSELoss()

            # Train model
            train_losses, valid_losses = train(
                models_list[-1],
                optimizers_list[-1],
                mse,
                AEVC,
                trainloader,
                validloader,
                epochs=args.epochs,
                savepath=args.outpath,
                idx=None if args.consensus == 1 else idx,
            )

            # Save training and validation losses
            if args.consensus == 1:
                fname_loss = os.path.join(args.outpath, "loss.dat")
            else:
                fname_loss = os.path.join(args.outpath, f"loss_{idx}.dat")

            np.savetxt(
                fname_loss,
                np.stack((train_losses, valid_losses), axis=-1),
                fmt="%.6e",
                header="tran_losses, valid_losses",
            )

            mlflow.log_artifact(fname_loss)

            if args.plot:
                for ext in [".png", ".pdf"]:
                    e = np.arange(args.epochs)

                    plt.figure()
                    plt.plot(e, train_losses, label="train")
                    plt.plot(e, valid_losses, label="validation")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()

                    # Save figure and store as MLFlow artifact
                    if args.consensus == 1:
                        plot.savefig("losses", path=args.outpath)
                    else:
                        plot.savefig(f"losses_{idx}", path=args.outpath)

                    plt.close()

        # Baseline for delta learning
        bl: Optional[Tuple[Any, Any, Any]] = None
        if args.baseline is not None:
            pdbids = np.loadtxt(args.baseline, usecols=(0,), dtype="U4")
            vina_score, logK = np.loadtxt(args.baseline, usecols=(1, 2), unpack=True)

            vina_pK = utils.vina_to_pK(vina_score)

            bl = (pdbids, vina_pK, logK)

        # Load best models
        best_models = []
        for idx in range(args.consensus):
            if args.consensus == 1:
                best_models.append(
                    utils.loadmodel(os.path.join(args.outpath, "best.pth"))
                )
            else:
                best_models.append(
                    utils.loadmodel(os.path.join(args.outpath, f"best_{idx}.pth"))
                )

        results_train = {}
        results_valid = {}
        results_test = {}

        for idx, bestmodel in enumerate(best_models):
            # Training set predictions
            ids_train, true_train, predicted_train = predict.predict(
                bestmodel, AEVC, trainloader, bl
            )

            # Validation set predictions
            ids_valid, true_valid, predicted_valid = predict.predict(
                bestmodel, AEVC, validloader, bl
            )

            # Store results
            if idx == 0:
                results_train["true"] = pd.Series(index=ids_train, data=true_train)
                results_valid["true"] = pd.Series(index=ids_valid, data=true_valid)
            results_train[f"predicted_{idx}"] = pd.Series(
                index=ids_train, data=predicted_train
            )
            results_valid[f"predicted_{idx}"] = pd.Series(
                index=ids_valid, data=predicted_valid
            )

            # Test set predictions
            if args.testfile is not None:
                ids_test, true_test, predicted_test = predict.predict(
                    bestmodel, AEVC, testloader, bl
                )

                # Store results
                if idx == 0:
                    results_test["true"] = pd.Series(index=ids_test, data=true_test)
                results_test[f"predicted_{idx}"] = pd.Series(
                    index=ids_test, data=predicted_test
                )

        # Build dataframes
        # This takes care of possible different order of data in different models
        df_train = pd.DataFrame(results_train)
        df_valid = pd.DataFrame(results_valid)
        df_test = pd.DataFrame(results_test)

        # Compute averages and stds
        df_train["avg"] = df_train.drop("true", axis="columns").mean(axis="columns")
        df_train["std"] = df_train.drop("true", axis="columns").std(axis="columns")
        df_valid["avg"] = df_valid.drop("true", axis="columns").mean(axis="columns")
        df_valid["std"] = df_valid.drop("true", axis="columns").std(axis="columns")

        train_csv = os.path.join(args.outpath, "train.csv")
        df_train.to_csv(train_csv)
        mlflow.log_artifact(train_csv)

        valid_csv = os.path.join(args.outpath, "valid.csv")
        df_valid.to_csv(valid_csv)
        mlflow.log_artifact(valid_csv)

        # Plot
        plot.regplot(
            df_train["true"].to_numpy(),
            df_train["avg"].to_numpy(),
            std=df_train["std"].to_numpy(),
            name="train",
            path=args.outpath,
        )

        plot.regplot(
            df_valid["true"].to_numpy(),
            df_valid["avg"].to_numpy(),
            std=df_valid["std"].to_numpy(),
            name="valid",
            path=args.outpath,
        )

        if args.testfile is not None:
            # Compute averages and stds
            df_test["avg"] = df_test.drop("true", axis="columns").mean(axis="columns")
            df_test["std"] = df_test.drop("true", axis="columns").std(axis="columns")

            plot.regplot(
                df_test["true"].to_numpy(),
                df_test["avg"].to_numpy(),
                std=df_test["std"].to_numpy(),
                name="test",
                path=args.outpath,
            )

            test_csv = os.path.join(args.outpath, "test.csv")
            df_test.to_csv(test_csv)
            mlflow.log_artifact(test_csv)
