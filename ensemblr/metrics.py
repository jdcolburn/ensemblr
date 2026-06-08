# Miscellaneous functions

import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

# function to get rmsd to a structure
def get_rmsd_to_ref(structure, ref_structure, selection, resid_offset=0, ref_resid_offset=0):
    mobile = mda.Universe(structure, structure)
    ref = mda.Universe(ref_structure, ref_structure)
    mobile.atoms.residues.resids += resid_offset
    ref.atoms.residues.resids += ref_resid_offset
    rmsds = alignto(mobile, ref, select=selection, match_atoms=True, weights=None)
    return [structure, rmsds[1]]

# function to calculate the SASA of a structure
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


def calc_all_distances(structure, distances_items, resid_offset=0):
    """Compute all named distances for one structure in a single universe load.

    Parameters
    ----------
    structure : str
        Path to the PDB file.
    distances_items : list of (name, sel1, sel2)
        Each entry is a (column_name, mda_selection_1, mda_selection_2) tuple.
    resid_offset : int

    Returns
    -------
    list of (name, distance)
    """
    universe = mda.Universe(structure, structure)
    universe.atoms.residues.resids += resid_offset
    results = []
    for name, sel1, sel2 in distances_items:
        sel1_coords = universe.select_atoms(sel1).positions
        sel2_coords = universe.select_atoms(sel2).positions
        dist = np.linalg.norm(np.mean(sel1_coords, axis=0) - np.mean(sel2_coords, axis=0))
        results.append((name, dist))
    return structure, results
