"""
Smoke tests: the package, every submodule and the CLI import and start up.
"""

import importlib
import pkgutil
import subprocess
import sys
from importlib import resources

import pytest

import ensemblr

SUBMODULES = [m.name for m in pkgutil.iter_modules(ensemblr.__path__)
              if m.name not in ('tests', '_version')]


@pytest.mark.parametrize('name', SUBMODULES)
def test_submodule_imports(name):
    importlib.import_module(f'ensemblr.{name}')


def test_public_api():
    for name in ensemblr.__all__:
        assert hasattr(ensemblr, name), name


def test_template_notebook_installed():
    template = resources.files('ensemblr') / 'templates' / 'umbrella_sampling_2d_grid_template.ipynb'
    assert template.is_file()


def test_generate_cli_help():
    out = subprocess.run([sys.executable, '-m', 'ensemblr.generate', '--help'],
                         capture_output=True, text=True, check=True)
    assert '--colabfold-dir' in out.stdout


def test_colabfold_dir_missing_is_clear(tmp_path):
    from ensemblr.generate import _colabfold_prefix
    with pytest.raises(FileNotFoundError, match='colabfold_batch'):
        _colabfold_prefix(str(tmp_path))
