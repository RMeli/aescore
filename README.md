# AEScore: Protein-Ligand Binding Affinity with Atomic Environment Vectors

![flake8](https://github.com/RMeli/ael/workflows/flake8/badge.svg)
![mypy](https://github.com/RMeli/ael/workflows/mypy/badge.svg)
![pytest](https://github.com/RMeli/ael/workflows/pytest/badge.svg)

Learning protein-ligand binding affinity using atomic environment vectors.

## Installation

```bash
conda create -f devtools/conda-envs/ael-test.yaml
```

```bash
pip install .
```

## Usage

### Training

```bash
python -m ael.train --help
```

### Inference

```bash
python -m ael.predict --help
```

## Test

Run tests:

```bash
pytest
```

## References

* Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost. Chemical Science,(2017), DOI: [10.1039/C6SC05720A](https://doi.org/10.1039/C6SC05720A)
* Gao, Xiang; Ramezanghorbani, Farhad; Isayev, Olexandr; Smith, Justin; Roitberg, Adrian (2020): TorchANI: A Free and Open Source PyTorch Based Deep Learning Implementation of the ANI Neural Network Potentials. ChemRxiv. Preprint. DOI: [10.26434/chemrxiv.12218294.v1](https://doi.org/10.26434/chemrxiv.12218294.v1)
* R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler, D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein. MDAnalysis: A Python package for the rapid analysis of molecular dynamics simulations. In S. Benthall and S. Rostrup, editors, Proceedings of the 15th Python in Science Conference, pages 98-105, Austin, TX, 2016. SciPy, DOI: [10.25080/majora-629e541a-00e](https://doi.org/10.25080/majora-629e541a-00e).
* N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations. J. Comput. Chem. 32 (2011), 2319-2327, DOI: [10.1002/jcc.21787](https://doi.org/10.1002/jcc.21787)
* O'Boyle, N.M., Banck, M., James, C.A. et al. Open Babel: An open chemical toolbox. J Cheminform 3, 33 (2011). DOI: [10.1186/1758-2946-3-33](https://doi.org/10.1186/1758-2946-3-33)
* N.M. O’Boyle, C. Morley and G.R. Hutchison. Pybel: a Python wrapper for the OpenBabel cheminformatics toolkit. Chem. Cent. J. 2008, 2, 5. DOI: [10.1186/1752-153X-2-5](https://bmcchem.biomedcentral.com/articles/10.1186/1752-153X-2-5)

## Copyright

Copyright (c) 2020-2021, Rocco Meli

### Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
