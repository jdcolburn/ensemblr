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


def test_trajectory_parse_name():
    from ensemblr.trajectory import _parse_name
    assert _parse_name('r0_1024-2048_rank_001.pdb') == (0, '', '1024:2048')
    assert _parse_name('r12_cell_3_14_181-362_rank_010.pdb') == (12, 'cell_3_14', '181:362')
    assert _parse_name('model.pdb') == (None, '', '')


def _toy_ensemble():
    import numpy as np
    import pandas as pd
    from ensemblr import Ensemble
    ens = Ensemble('/nonexistent/')
    rng = np.random.default_rng(0)
    ens.df = pd.DataFrame({'pLDDT': rng.uniform(70, 95, 100),
                           'PC1': np.linspace(-5, 5, 100),
                           'PC2': rng.normal(size=100),
                           'cluster': np.repeat([0, 1, -1, 2], 25)},
                          index=pd.Index([f's{i:03d}.pdb' for i in range(100)], name='structure'))
    return ens


def test_select_nearest_representatives():
    ens = _toy_ensemble()
    box = ens.select(PC1=(2, None), PC2=(-1, 1))
    assert len(box) and (box.PC1 >= 2).all() and (box.PC2.abs() <= 1).all()
    assert set(ens.select(cluster=[0, 2]).cluster) == {0, 2}
    assert (ens.select('PC1 > 0', pLDDT=(80, None)).PC1 > 0).all()
    nn = ens.nearest(PC1=0, PC2=0, n=3)
    assert len(nn) == 3 and nn.distance.is_monotonic_increasing
    reps = ens.representatives()                    # HDBSCAN-style labels, noise skipped
    assert list(reps.group) == [0, 1, 2] and list(reps.n_members) == [25, 25, 25]
    bins = ens.representatives('PC1', n_bins=5)
    assert len(bins) == 5 and bins.PC1.is_monotonic_increasing


def test_names_accepts_selections():
    ens = _toy_ensemble()
    top = ens.df['PC1'].nlargest(3)
    assert ens._names(top) == ['s099.pdb', 's098.pdb', 's097.pdb']
    assert ens._names(ens.df.PC1 > 4.9) == ['s099.pdb']
    assert ens._names(['/x/s001.pdb']) == ['s001.pdb']
    assert ens.paths('s001.pdb')[0].endswith('/nonexistent/s001.pdb')
    with pytest.raises(KeyError):
        ens._names(['nope.pdb'])
