# ensemblr

A notebook-driven workflow for using **synthetic ensembles** of AlphaFold predictions to derive collective variables (CVs) and generate umbrella sampling (US) setups.

The idea is to treat the **synthetic ensemble** as a coarse or approximate prior, and to recover the thermodynamics of the real ensemble using enhanced sampling.

Spiritual successor to [MEMENTO](https://github.com/simonlichtinger/PyMEMENTO).

---

## Overview

This repository documents a workflow to:

- Analyse AF2 structural ensembles  
- Identify collective variables  
- Select representative seed structures  
- Construct smooth transition paths  
- Generate umbrella sampling inputs  

A lightweight embedding routine is also provided to insert selected structures into simulation-ready systems. This code prioritises **transparency and reproducibility** over general-purpose packaging.

---

## Pipeline

1. **Generate ensemble**
   - External (e.g. `localcolabfold`)
   - Adaptive generator: `ensemblr-generate` (see `GENERATE_AF2_ENSEMBLE_V2.md`)

2. **Preprocess structures**
   - Filtering (e.g. pLDDT)
   - Alignment and atom selection
   - Feature calculation (RMSD, distances, SASA)

3. **Dimensionality reduction**
   - PCA on selected atoms

4. **Clustering / binning**
   - Structures grouped along chosen CV

5. **Matrix construction**
   - RMSD matrix (JAX-accelerated)
   - CV difference matrix

6. **Path optimisation**
   - Monte Carlo simulated annealing (MCSA)
   - Objective combines:
     - Structural smoothness (RMSD)
     - CV smoothness / spacing
     - Optional orthogonal DOFs or density of states terms

7. **Umbrella sampling setup**
   - Generate window structures
   - Embed into template system
   - Write PLUMED input files

---

## 2D Umbrella Sampling (in development)

A 2D extension of the pipeline exists (`GridPathFinder`, `GridUmbrellaSamplingPrep`, `GridSolventSystemPrep`) that places windows on a regular grid in (PC1, PC2) space to recover a 2D PMF. The MC objective uses Ramachandran smoothness + optional Euclidean CV distance between 4-connected grid neighbours. Window directories are named `window_N_M` (bin indices).

**What works:**
- Grid construction and MC optimisation
- GROMACS solvent system prep (`GridSolventSystemPrep`)
- 2D PLUMED input generation (two PCAVARS + two RESTRAINTs per window)
- PLUMED driver cross-check cell in the notebook

**Still missing:**
- `GridUmbrellaSamplingPrep` (membrane/template path) is implemented but untested
- No GROMACS submission scripts for the 2D case

---

## Installation

Python ≥ 3.10. The conda environment is the easiest route, because it also provides the external
programs (`mkdssp`, `plumed`):

```bash
git clone https://github.com/jdcolburn/ensemblr.git
cd ensemblr
conda env create -f environment.yml
conda activate ensemblr
pytest --pyargs ensemblr        # quick check that the install works
```

Or with pip, into an environment you already have (Python packages only):

```bash
pip install git+https://github.com/jdcolburn/ensemblr.git
```

Use `pip install -e ".[notebook]"` from a clone if you want to edit the code or run the notebooks.
The JAX installed by default runs on the CPU; for GPU see the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html) (e.g. `pip install "jax[cuda12]"`).

### External programs

These are not Python packages, so pip can't install them. Install the ones you need:

| Program | Needed for | How to get it |
|---|---|---|
| ColabFold (`colabfold_batch`) | generating ensembles (`ensemblr-generate`) | [localcolabfold](https://github.com/YoshitakaMo/localcolabfold); then put it on `PATH` or set `COLABFOLD_DIR` |
| `mkdssp` | secondary-structure atom selection (`smart_selector`, generator PCA) | in `environment.yml` (`conda install -c conda-forge dssp`) |
| `plumed` | checking US window CVs (`plumed.verify_*`) | in `environment.yml` (`conda install -c conda-forge plumed`) |
| `gmx` | solvating / equilibrating windows (`SolventSystemPrep`, `gmx_stages`) | your GROMACS install |
| `hole`, `memembed` | pore profiles (`PoreProfiler`) | `conda install -c conda-forge hole2`; [memembed](https://github.com/psipred/memembed) |

---

## Usage

1. Generate an ensemble: `ensemblr-generate --dry-run`, then `ensemblr-generate`, from a directory
   containing `input.fasta`. See `GENERATE_AF2_ENSEMBLE_V2.md` for options. At the end of a run the
   generator also writes `structures.xtc` (every accepted model as one trajectory, superposed on the
   PCA selection), its topology `structures.pdb`, and `structures_index.csv` (frame → file, round,
   depth, cell, pLDDT): `vmd structures.pdb structures.xtc`. For a run made before this, run
   `ensemblr-xtc <output_dir>`.
2. Analyse it and set up umbrella sampling in a notebook: copy one of the example notebooks, edit the
   settings at the top, and run it top to bottom. `example_notebook_1d_clean.ipynb` (also installed as
   `ensemblr/templates/umbrella_sampling_1d_template.ipynb`) is the 1D path;
   `example_notebook_2d_grid_clean.ipynb` (also installed as
   `ensemblr/templates/umbrella_sampling_2d_grid_template.ipynb`) is the 2D grid.
3. To explore an ensemble your own way (pick structures by PC value or cluster, write them out, add your
   own per-structure properties or clustering), start from `example_notebook_explore.ipynb` (also
   installed as `ensemblr/templates/explore_ensemble_template.ipynb`): set `ENSEMBLE_DIR` and run it.

### Getting at the structures

`ens.df` has one row per structure (indexed by file name) with pLDDT, `PC1…PCn`, `cluster` and any
metrics. Pick rows any way you like, then write or locate them:

```python
from ensemblr import Ensemble
ens = Ensemble.from_run('synthetic_ensemble/run_01')   # reopen a notebook run (no recomputation)

hi = ens.df.nlargest(20, 'PC1')                        # plain pandas works
box = ens.select(PC1=(10, None), PC2=(-2, 2))          # ranges; None = open end
near = ens.nearest(PC1=5.0, PC2=-1.0, n=5)             # closest to a point (adds 'distance')
bins = ens.representatives('PC1', n_bins=12)           # one structure per PC1 bin
reps = ens.representatives('cluster')                  # one per cluster (closest to centroid)

ens.write(hi, 'pc1_high.xtc')     # pc1_high.xtc + .pdb topology + _index.csv, superposed
ens.write(reps, 'reps.pdb')       # one multi-model PDB
ens.write(near, 'near_dir/')      # one PDB per structure
ens.paths(box)                    # absolute PDB paths
```

For your own clustering, add a label column and reuse the same calls:
`ens.df['km'] = KMeans(5).fit_predict(ens.df[['PC1', 'PC2']]); ens.representatives('km')`.
`write()` reads frames from the run's `structures.xtc` when it exists (much faster than opening
thousands of symlinked PDBs on NFS). `ens.universe` is the whole ensemble as one aligned
MDAnalysis Universe, frame *i* = row *i* of `ens.df`. `ens.save(run_dir)` saves an analysis for
`from_run`.

---

## Modules

| Module | Contents |
|---|---|
| `generate.py` | Adaptive AF2 ensemble generator (`ensemblr-generate`) |
| `ensemble.py` | `Ensemble`: loads structures, pLDDT filtering, alignment, PCA and other CVs, clustering |
| `path_finder.py`, `grid_path_finder.py` | `MCPathFinder` (1D path) and `GridPathFinder` (2D grid) |
| `monte_carlo.py` | Monte Carlo simulated annealing used by the path finders |
| `calc_matrices.py` | Pairwise RMSD and CV-difference matrices (JAX) |
| `us_prep.py` | `UmbrellaSamplingPrep`, `GridUmbrellaSamplingPrep`: embed window structures into a template system |
| `solvent_prep.py` | `SolventSystemPrep`, `GridSolventSystemPrep`: water-box systems and Slurm scripts |
| `structure_prep.py` | Coordinate replacement, overlap and threaded-lipid fixes |
| `plumed.py` | PLUMED input writers and `plumed driver` checks |
| `gmx_stages.py` | Run grompp/mdrun stage chains over window directories |
| `smart_selector.py` | DSSP-based atom selection |
| `metrics.py` | RMSD, SASA and distance CVs |
| `adaptive_reweighting.py` | Importance-sampling reweighting of adaptive ensembles |
| `pore_profiles.py` | `PoreProfiler`: HOLE pore profiles |
| `pdb_fetch.py` | Fetch experimental structures for a UniProt ID |

---

#### Acknowledgements 

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
