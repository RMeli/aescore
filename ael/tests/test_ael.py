"""
Unit and regression test for the ael package.
"""

# Import package, test suite, and other packages as needed
import ael
import pytest
import sys

def test_ael_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ael" in sys.modules
