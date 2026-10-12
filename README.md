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
   containing `input.fasta`. See `GENERATE_AF2_ENSEMBLE_V2.md` for options.
2. Analyse it and set up umbrella sampling in a notebook: copy one of the example notebooks, edit the
   settings at the top, and run it top to bottom. `example_notebook_clean.ipynb` is the 1D workflow;
   `example_notebook_2d_grid_clean.ipynb` (also installed as
   `ensemblr/templates/umbrella_sampling_2d_grid_template.ipynb`) is the 2D grid.

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
