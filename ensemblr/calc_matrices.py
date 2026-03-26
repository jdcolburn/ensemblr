# Matrix calculation and handling functions

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit

from Bio.PDB import PDBParser

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Janin

def get_rmsdmat_jax(ensemble_pdb, ensemble_dataframe):

    """
    Calculate RMSD matrix for the ensemble using jax.

    Parameters
    ----------
    ensemble_coordinates : numpy array, Required, default: None
        array consisting of the coordinates of teh entire ensemble of structures
    ensemble_dataframe : pandas dataframe, Required, default: None
        the pandas dataframe with all the structures in the ensemble.

    Returns
    -------
    results : numpy array
        The resulting RMSD matrix.
    """

    jnp_rmsd_parallel = jax.jit(jax.vmap(jnp_rmsd,(None,0)))
    data = ensemble_pdb

    # populate the RMSD matrix
    results = []
    for n in range(len(ensemble_dataframe)):
      results.append(np.array(jnp_rmsd_parallel(data[n],data)))

    # turn the "results" list into a numpy array float 64
    results = np.array(results, dtype=np.float32)

    return results

def pdb_to_coordinates(pdb_file):
    
    """
    Extract atomic coordinates from a PDB file into a NumPy array.

    Parameters
    ----------
    pdb_file : string, Required, default: None
        PDB file to convert to coordinate matrix.

    Returns
    -------
    coordinates : numpy array
        PDB coordinates as a numpy array.
    """

    parser = PDBParser()
    structure = parser.get_structure('trajectory', pdb_file)
    # Initialize an empty list to store coordinates
    coordinates = []
    for model in structure:
        model_coordinates = []
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Extract coordinates
                    atom_coordinates = atom.get_coord()
                    model_coordinates.append(atom_coordinates)
        coordinates.append(model_coordinates)
    # Convert the list of coordinates into a numpy array
    return np.array(coordinates)

def jnp_kabsch(a, b):

    """
    Compute the optimal rotation matrix that aligns two point sets
    using the Kabsch algorithm.

    Parameters
    ----------
    a : jax.numpy.ndarray, shape (N, 3)
        Reference coordinate set. Coordinates should be centred
        (i.e., centroid removed) prior to calling this function.
    b : jax.numpy.ndarray, shape (N, 3)
        Coordinate set to be aligned to `a`. Must correspond one-to-one
        with `a` and also be centred.

    Returns
    -------
    R : jax.numpy.ndarray, shape (3, 3)
        Optimal rotation matrix that minimizes the RMSD between `a`
        and `b`, such that:
            a ≈ b @ R

    """

    u, s, vh = jnp.linalg.svd(a.T @ b, full_matrices=False)
    u = jnp.where(jnp.linalg.det(u @ vh) < 0, u.at[:,-1].set(-u[:,-1]), u)
    return u @ vh

def jnp_rmsd(true, pred):
        
    """
    Compute the root-mean-square deviation (RMSD) between two coordinate
    sets after optimal rigid-body alignment.

    Parameters
    ----------
    true : jax.numpy.ndarray, shape (N, 3)
        Reference coordinates. Each row corresponds to a point (e.g., atom).
    pred : jax.numpy.ndarray, shape (N, 3)
        Coordinates to compare against `true`. Must have a one-to-one
        correspondence with `true`.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the aligned coordinate sets.
    """

    p = true - true.mean(0,keepdims=True)
    q = pred - pred.mean(0,keepdims=True)
    p = p @ jnp_kabsch(p, q)
    return jnp.sqrt(jnp.square(p-q).sum(-1).mean())

def get_diffmat(dataframe, property_name):
    
    """
    Make a matrix of differences in collective variable between structures

    Parameters
    ----------
    dataframe : pandas dataframe, Required, default: None
        Pandas dataframe in which to look up property_name.
    property_name : string, Required, default: None
        Name of the property (column) in the pandas dataframe for which to make the difference matrix.

    Returns
    -------
    matrix : numpy array
        The resulting difference matrix.
    """

    matrix = np.zeros((len(dataframe), len(dataframe)))
    n_elements = len(dataframe) * len(dataframe)
    for i in range(len(dataframe)):
        for j in range(len(dataframe)):
            matrix[i,j] = (dataframe[property_name].iloc[i] - dataframe[property_name].iloc[j])
    return matrix

def get_summat(dataframe, property_name):
    
    """
    Make a matrix of sums of collective variable between structures, usful for energies where they are summed over a path

    Parameters
    ----------
    dataframe : pandas dataframe, Required, default: None
        Pandas dataframe in which to look up property_name.
    property_name : string, Required, default: None
        Name of the property (column) in the pandas dataframe for which to make the difference matrix.

    Returns
    -------
    matrix : numpy array
        The resulting difference matrix.
    """

    matrix = np.zeros((len(dataframe), len(dataframe)))
    n_elements = len(dataframe) * len(dataframe)
    for i in range(len(dataframe)):
        for j in range(len(dataframe)):
            matrix[i,j] = (dataframe[property_name].iloc[i] + dataframe[property_name].iloc[j])
    return matrix

# JIT-compiled cosine similarity computation using JAX
@jit
def compute_cosine_similarity(u1_angles, u2_angles):
    """
    Compute cosine similarity between two angle arrays using JAX.

    Parameters:
    - u1_angles: JAX array, flattened angle array of the first structure.
    - u2_angles: JAX array, flattened angle array of the second structure.

    Returns:
    - cosine_similarity: float, similarity between the two angle sets.
    """
    dot_product = jnp.dot(u1_angles, u2_angles)
    norm_u1 = jnp.linalg.norm(u1_angles)
    norm_u2 = jnp.linalg.norm(u2_angles)
    return dot_product / (norm_u1 * norm_u2)

# Function to compute cosine similarity between two structures
def get_janin_similarity(structure_directory, structure1, structure2, selection='protein'):
    """
    Calculate the cosine similarity between Janin angle sets of two protein structures.

    Parameters:
    - structure_directory: str, directory containing the structure files.
    - structure1: str, filename of the first structure.
    - structure2: str, filename of the second structure.
    - selection: str, atom selection string for MDAnalysis (default: 'protein').

    Returns:
    - cosine_similarity: float, similarity between the two angle sets.
    """
    # Load structures into MDAnalysis Universes
    u1 = mda.Universe(structure_directory + structure1, structure_directory + structure1)
    janin_u1 = Janin(u1.select_atoms(selection))
    u2 = mda.Universe(structure_directory + structure2, structure_directory + structure2)
    janin_u2 = Janin(u2.select_atoms(selection))
    
    # Run the Janin analysis
    janin_u1.run()
    janin_u2.run()
    
    # Extract and flatten angle sets
    u1_flattened = jnp.array(janin_u1.angles.flatten())  # Convert to JAX array
    u2_flattened = jnp.array(janin_u2.angles.flatten())  # Convert to JAX array
    
    # Compute cosine similarity using JAX JIT
    return compute_cosine_similarity(u1_flattened, u2_flattened)

# Function to compute pairwise similarity matrix using JAX
def get_janin_overlap_matrix(dataframe, structure_directory):
    """
    Compute a pairwise cosine similarity matrix for all structures in a dataframe using JAX.

    Parameters:
    - dataframe: pandas.DataFrame, contains structure filenames in the 'structure' column.
    - structure_directory: str, directory containing the structure files.

    Returns:
    - matrix: np.ndarray, pairwise similarity matrix of shape (n_structures, n_structures).
    """
    # Initialize an empty square matrix for storing pairwise similarities
    n_structures = len(dataframe)
    matrix = np.zeros((n_structures, n_structures))
    
    # Compute pairwise similarities with JAX JIT
    for i in range(n_structures):
        for j in range(i, n_structures):  # Upper triangular only (symmetry optimization)
            similarity = get_janin_similarity(
                structure_directory,
                dataframe.iloc[i]['structure'],
                dataframe.iloc[j]['structure']
            )
            matrix[i, j] = similarity
            matrix[j, i] = similarity  # Exploit symmetry
    
    return matrix

