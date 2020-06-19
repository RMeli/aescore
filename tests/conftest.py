import os

import pytest


@pytest.fixture
def testdir():
    wdir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(wdir, "testdata")


@pytest.fixture
def testdata():
    wdir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(wdir, "testdata/systems.dat")
