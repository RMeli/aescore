import argparse as ap

import numpy as np


def trainparser(default="BP"):
    """
    Command line arguments parser.

    Returns
    -------
    argarse.Namespace
        Parsed arguments
    """

    if default.strip().upper() == "BP":
        TsA_default = [0.0, np.pi]
        RsA_default = [0.0]
    elif default.strip().upper() == "ANI":
        TsA_default = ([0.196, 0.589, 0.982, 1.37, 1.77, 2.16, 2.55, 2.95],)
        RsA_default = [0.90, 1.55, 2.2, 2.85]
    elif default.strip().upper() == "MIX":
        TsA_default = [0.0, np.pi]
        RsA_default = [0.90, 1.55, 2.2, 2.85]
    else:
        raise NameError(f"{default}: default not implemented.")

    parser = ap.ArgumentParser(description="Train affinity prediction model.")

    parser.add_argument("experiment", type=str, help="MLFlow experiment")
    parser.add_argument("trainfile", type=str, help="Training set file")
    parser.add_argument("validfile", type=str, default=None, help="Validation set file")

    parser.add_argument(
        "-t", "--testfile", type=str, default=None, help="Test set file"
    )
    parser.add_argument("-d", "--datapaths", type=str, default="", help="Path to data")

    parser.add_argument(
        "-r", "--distance", type=float, default=0.1, help="Residue selection distance"
    )  # TODO: Change to larger distance

    # Radial
    parser.add_argument("-RcR", type=float, default=5.2, help="Radial cutoff")
    parser.add_argument("-EtaR", type=float, default=16.0, help="Radial decay")
    parser.add_argument(
        "-RsR",
        nargs="+",
        type=float,
        default=[
            0.90,
            1.17,
            1.44,
            1.71,
            1.98,
            2.24,
            2.51,
            2.78,
            3.05,
            3.32,
            3.59,
            3.86,
            4.13,
            4.39,
            4.66,
            4.93,
        ],
        help="Radial shift",
    )

    # Angular
    parser.add_argument("-RcA", type=float, default=5.2, help="Angular cutoff")
    parser.add_argument("-EtaA", type=float, default=3.5, help="Angular decay")
    parser.add_argument(
        "-RsA", nargs="+", type=float, default=RsA_default, help="Angular radial shift",
    )
    parser.add_argument(
        "-TsA", nargs="+", type=float, default=TsA_default, help="Angular shift",
    )
    parser.add_argument(
        "-Zeta", type=float, default=32.0, help="Angular multiplicity",
    )

    parser.add_argument("-b", "--batchsize", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("-lr", type=float, default=0.0005, help="Leanring rate")
    parser.add_argument(
        "-l", "--layers", type=int, nargs="+", default=None, help="Atomic NN layers"
    )
    parser.add_argument(
        "-p", "--dropout", type=float, default=None, help="Dropout probability"
    )

    parser.add_argument(
        "-c",
        "--consensus",
        type=int,
        default=1,
        help="Number of models for consensus scoring",
    )

    parser.add_argument(
        "-cm", "--chemap", type=str, default=None, help="Chemical elements mapping"
    )

    parser.add_argument("-o", "--outpath", type=str, default="", help="Output path")

    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--baseline", type=str, help="Vina baseline")

    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    return args
