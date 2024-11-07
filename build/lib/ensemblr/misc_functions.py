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
from Bio.PDB.SASA import ShrakeRupley           # for SASA calculation
from IPython.display import display             # for data frame display
from multiprocessing import Pool                # for multiprocessing
from tqdm import tqdm                           # for progress bars

# define function to get rmsd to a structure
def get_rmsd_to_ref(structure, ref_structure, selection, resid_offset=0):
    mobile = mda.Universe(structure, structure) # make universe
    ref = mda.Universe(ref_structure, ref_structure)    # make universe
    mobile.atoms.residues.resids += resid_offset                                            # renumber residues in mobile
    rmsds = alignto(mobile, ref, select=selection, match_atoms=True, weights=None)     # these ref selections are different becasue ref has different residue numbering
    return [structure, rmsds[1]]                                                            # [1] = rmsd after alignment

# define a function to calculate the SASA of a structure
def get_sasa(structure):
    p = PDBParser(QUIET=1)
    struct = p.get_structure(structure, structure)
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(struct, level="S")
    return [structure, struct.sasa]  

# calculate certain distances e.g. to use as a CV
def calc_distance(structure, selection1, selection2, resid_offset=0):
    universe = mda.Universe(structure, structure) # make universe
    universe.atoms.residues.resids += resid_offset      
    sel1_coords = universe.select_atoms(selection1).positions
    sel2_coords = universe.select_atoms(selection2).positions
    sel1_coords_avg = np.mean(sel1_coords, axis=0)  # geometric average = COM when all atoms are Ca
    sel2_coords_avg = np.mean(sel2_coords, axis=0)  # geometric average = COM when all atoms are Ca
    distance = np.linalg.norm(sel1_coords_avg - sel2_coords_avg)
    return distance