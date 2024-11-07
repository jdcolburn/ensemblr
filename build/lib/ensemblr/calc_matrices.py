# alternate faster way to calculate rmsd matrix

import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import jax
from Bio.PDB import PDBParser

def test():
    print('Hello world')


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
    Robbie - fill in these docs please.

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
    Robbie - fill in these docs please.

    Parameters
    ----------
    ?

    Returns
    -------
    ?
    """

    u, s, vh = jnp.linalg.svd(a.T @ b, full_matrices=False)
    u = jnp.where(jnp.linalg.det(u @ vh) < 0, u.at[:,-1].set(-u[:,-1]), u)
    return u @ vh

def jnp_rmsd(true, pred):
        
    """
    Robbie - fill in these docs please.

    Parameters
    ----------
    ?

    Returns
    -------
    ?
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