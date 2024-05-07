"""
Unit and regression test for the ensemblr package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ensemblr


def test_ensemblr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ensemblr" in sys.modules
