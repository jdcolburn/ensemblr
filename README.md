# ensemblr

A notebook-driven workflow for using **AlphaFold (AF) structural ensembles** to construct collective variables (CVs) and generate umbrella sampling (US) setups.

The idea is to treat the "synthetic" AF ensemble as a coarse or approximate prior, and to recover the thermodynamics of the real ensemble using enhanced sampling.

---

## Overview

This repository documents a workflow to:

- Analyse AF2 structural ensembles  
- Identify collective variables  
- Select representative seed structures  
- Construct smooth transition paths  
- Generate umbrella sampling inputs  

A lightweight embedding routine is also provided to insert selected structures into simulation-ready systems.

> This code prioritises **transparency and reproducibility** over general-purpose packaging.

---

## Pipeline

1. **Generate ensemble**
   - External (e.g. `localcolabfold`)
   - Example script: `generate_af2_ensemble.sh`

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

## Files

| File | Description |
|---|---|
| `calc_matrices.py` | Constructs pairwise RMSD and CV-difference matrices (JAX-accelerated) |
| `monte_carlo.py` | Monte Carlo simulated annealing for path optimisation |
| `us_window_setup.py` | Embeds structures and prepares umbrella sampling inputs |
| `misc_functions.py` | Utility functions (RMSD, SASA, distance CVs) |
| `smart_selector.py` | DSSP-based intelligent atom selection |

---

## Dependencies

### Core

- numpy  
- pandas  
- matplotlib  
- seaborn  
- MDAnalysis  
- Biopython  

### Optional (performance)

- jax  
- jaxlib  

### Utilities

- tqdm (progress bars)  
- IPython (display utilities)  

---

## Installation

Clone and add to your Python path:

```bash
git clone <repo_url>
cd <repo>
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

#### Acknowledgements 

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
