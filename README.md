==============================
# ensemblr

SBCB notebook-driven workflow for using AlphaFold (AF) structural ensembles to perform umbrella sampling (US).

This repository is intended to document the workflow used to:
- analyse structural ensembles
- identify collective variables 
- select seed structures 

Also included is a basic routine that embeds these seed structures into user-provided coordinates. This code is primarily designed for transparency and reuse by other researchers, rather than as a fully packaged, general-purpose software tool.

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

---

## Overview

The key idea is to treat the synthetic (AF) ensemble as a coarse or approximate representation of the real conformational landscape, and to use it to find various optimal paths between states of interest.

### Pipeline

1. **Generate ensemble**
   - AlphaFold ensemble (e.g. via localcolabfold)

2. **Preprocess structures**
   - filtering (e.g. pLDDT)
   - alignment / atom selection
   - feature calculation (RMSD, distances, SASA, etc.)

3. **Dimensionality reduction**
   - PCA on selected atoms

4. **Clustering / binning**
   - structures grouped along a chosen CV

5. **Matrix construction**
   - RMSD matrix (JAX-accelerated)  
   - CV difference matrix

6. **Path optimisation**
   - Monte Carlo simulated annealing
   - objective combines:
     - structural smoothness (RMSD)
     - CV smoothness / spacing
     - optional orthogonal DOFs or density of states terms

7. **Umbrella sampling setup**
   - generate window structures
   - embed into template system
   - write PLUMED input files

---

## Module Overview

The repository is organised into a small set of modules corresponding to each stage of the workflow.

### `calc_matrices.py`
Construction of pairwise matrices used for path optimisation.

- JAX-accelerated RMSD matrix computation
- CV difference matrices (for ordering and spacing)
- Optional similarity metrics (e.g. dihedral-based comparisons)

---

### `monte_carlo.py`
Monte Carlo optimisation of pathways through the ensemble.

- Simulated annealing over discrete structure selections
- Composite objective function combining:
  - RMSD smoothness (structural continuity)
  - CV smoothness (even window spacing)
  - optional orthogonal DOFs and density terms

---

### `us_window_setup.py`
Preparation of umbrella sampling systems.

- Alignment of structures to a template system
- Coordinate replacement into a simulation-ready topology
- PLUMED input file generation (PCA or distance CVs)
- Heuristic fixes for clashes and geometric artefacts

---

### `misc_functions.py`
Utility functions for structural analysis.

- RMSD to reference structures
- SASA calculations
- Distance-based collective variables

---

### `smart_selector.py`
Automatic generation of atom selection strings.

- Uses DSSP-derived secondary structure to "intelligently" define selections
- Allows inclusion/exclusion of specific residues

---

## Installation

Clone and add to your Python path:

```bash
git clone <repo_url>
cd <repo>
export PYTHONPATH=$PYTHONPATH:$(pwd)
