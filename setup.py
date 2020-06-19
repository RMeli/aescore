import sys

from setuptools import find_packages, setup

import versioneer

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup(
    name="ael",
    author="Rocco Meli",
    author_email="rocco.meli@biodtp.ox.ac.uk",
    description="Learning Protein-Ligand Properties from Atomic Environment Vectors",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    # install_requires=[],
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],
    python_requires=">=3.6",
)
