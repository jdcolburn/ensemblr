import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import seaborn as sns
import nglview as nv                            # for visualisation
from MDAnalysis.analysis.align import alignto   # for aligning structures
from MDAnalysis.analysis.pca import PCA         # for PCA
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP                   # for secondary structure selection

# define function to get rmsd to a structure
def get_rmsd_to_structure(structure):
    mobile = mda.Universe(structure, structure) # make universe
    mobile.atoms.residues.resids += resid_offset                                            # renumber residues in mobile
    rmsds = alignto(mobile, ref, select=rmsd_selection, match_atoms=True, weights=None)     # these ref selections are different becasue ref has different residue numbering
    return [structure, rmsds[1]]                                                            # [1] = rmsd after alignment

# define a function to calculate the SASA of a structure
def get_sasa(structure):
    p = PDBParser(QUIET=1)
    struct = p.get_structure(structure, structure)
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(struct, level="S")
    return [structure, struct.sasa]  



