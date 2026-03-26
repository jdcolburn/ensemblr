# MC path finding by exchanging structures within bins

import gc
import numpy as np

# function to calculate the energy of a given path 
def calc_energy(path, rmsd_matrix, cv_matrix, dataframe, odf_matrix=None, dos_matrix=None, wf_rmsd=0.1, wf_cv=1, wf_odf=1, wf_dos=0.1):   #  wf_rmsd=0.01, wf_cv=1, wf_odf=0.5, wf_dos=0.001

    """
    Function to calculate the "energy" of a path. Currently hardcoded to work with the ensemble dataframe.

    Energy function consists of:
    - average of pairwise RMSDs along the path (wf_rmsd)
    - average of pairwise differences in CV value along the path (wf_cv)

    you can set the coefficients for each using the wf_rmsd and wf_cv arguments.

    Parameters
    ----------
    path : list, Required, default: None
        A list of dataframe indices corresponding to structures in dataframe.

    path_length : int, Required, default: None
        The total number of windows in the path. This should be deprecated later because it will (always?) correspond to the number of elements in path.

    rmsd_matrix : numpy array, Required, default: None
        The RMSD matrix for the ensemble.
    cv_matrix : numpy array, Required, default: None
        The CV matrix for the ensemble.
    dataframe : pandas dataframe, Required, default: None
        The pandas dataframe with all the structures in the ensemble.
    wf_rmsd : float, Optional, default: 1
        Coefficient for the rmsd part of the energy expression. Set to zero for a path with random smoothness, or to -1 for a maximally "unsmooth" path.
    wf_cv : float, Optional, default: 1
        Coefficient for the CV part of the energy expression. Determines "smoothness" w.r.t. values of the CV along the path. 1 should be ideal for umbrella sampling.
    cv2_matrix : numpy array, Optional, default: None
        The matrix for an optional second CV the ensemble. e.g. for PC2. If not provided, the energy will be calculated without this term.

    Returns
    -------
    total_energy : float
        The total energy of the path.
    """

    path_length = len(path)

    # contribution from the RMSD of the structures (ensures smoothness in cartesian space, for good phase-space overlap between windows)
    energy_rmsd = 0
    for i in range(path_length - 1):
        energy_rmsd += (rmsd_matrix[np.where(dataframe.index.values == path[i])[0][0], np.where(dataframe.index.values == path[i+1])[0][0]]**2)
    energy_rmsd = wf_rmsd * np.sqrt ( 1/path_length * energy_rmsd )

    # contribution from the CV of interest (ensures smoothness in the CV space and even distribution of windows)
    energy_cv = 0
    for i in range(path_length -1):
        energy_cv += (cv_matrix[np.where(dataframe.index.values == path[i])[0][0], np.where(dataframe.index.values == path[i+1])[0][0]]**2)
    energy_cv = wf_cv * np.sqrt ( 1/path_length * energy_cv )

    # contribution from a second variable representing largest orthogonal DOF (e.g. largest PC or second largest if PC1 is used as CV)
    energy_odf = 0
    if odf_matrix is not None:
        for i in range(path_length -1):
            energy_odf += (odf_matrix[np.where(dataframe.index.values == path[i])[0][0], np.where(dataframe.index.values == path[i+1])[0][0]]**2)
        energy_odf = wf_odf * np.sqrt ( 1/path_length * energy_odf ) 

    # contribution from the density of states of the AF2 predictions (NB this will depend on your ensemble, AF2 is not a boltzmann generator but this can be used as a proxy for the real DOS in the absence of another prior)
    energy_dos = 0
    if dos_matrix is not None:
        for i in range(path_length -1):
            energy_dos += (dos_matrix[np.where(dataframe.index.values == path[i])[0][0], np.where(dataframe.index.values == path[i+1])[0][0]])
        energy_dos = wf_dos * ( 1/path_length * energy_dos )

    total_energy = 0
    total_energy = energy_rmsd + energy_cv + energy_odf + energy_dos

    return total_energy

# define a function that runs monte carlo simulated annealing to optimise smoothness
def mc_path_optimisation(seed, initial_guess_indices, rmsd_matrix, cv_matrix, dataframe, fixed_endpoints=True, odf_matrix=None, dos_matrix=None, mc_n_steps=1000, initial_temperature=0.0001, cooling_factor=10000, debug=True):

    """
    Function to select the optimal set of N structures for a path of length N between two endpoints. Currently hardcoded to work with the ensemble dataframe.
    
    Works by exchanging structures within bins (as indicated in the dataframe)

    Parameters
    ----------
    seed : float, Required, default: None
        Seed for the random draws.
    initial_guess_indices : list, Required, default: None
        A list of dataframe indices corresponding to structures in dataframe.
    fixed_endpoints : Bool, Optional, default: True
        Whether or not the endpoints will be optimised. Set to False if you want to keep the end states structures consistent.
    mc_n_steps : int, Optional, default: 1000
        Total number of MC iterations (moves proposed).
    initial_temperature : float, Optional, default: 0.01
        Initial temperature for metropolis criterion.
    cooling_factor : float, Optional, default: 10000
        Defines temperature schedule. A list of N logarithmically decreasing temperatures is generated between initial_temperature and initial_temperature/cooling_factor.
    cv2_matrix : numpy array, Optional, default: None
        The matrix for an optional second CV the ensemble. e.g. for PC2. If not provided, the energy will be calculated without this term.
        
    Returns
    -------
    total_energy : float
        The total energy of the path.
    final_energy : float
        The final energy of the optimised path
    final_path: list
        list of indices corresponding to structures in the ensemble dataframe
    final_path_structures: tuple
        The sequence of structure filenames corresponding to the indices in final_path as they appear in the ensemble dataframe
    relaxation_energies: list
        Record of the total path energies from initial path to the final optimised path (shows relaxation)
    """

    #fixed_endpoints = True
    #mc_n_steps = 1000
    #initial_temperature = 0.01
    #cooling_factor = 10000 # initial temp is divided by this to get the final temperature

    np.random.seed(seed)

    # list of temperatures mapped to mc_n_steps decreasing stepwise in 10 increments
    final_temperature = initial_temperature/cooling_factor
    temperatures = np.logspace(np.log10(initial_temperature), np.log10(final_temperature), num=mc_n_steps)

    #mc_path_length = mc_n_bins

    mcpath = []

    #initial_guess_indices = []
    #initial_guess_indices = ideal_initial_path

    mcpath = initial_guess_indices

    # dictionary of paths and energies
    path_energies = {}

    # run n_mc_runs steps
    for i in range(mc_n_steps):

        temperature = temperatures[i]

        # calculate energy of the initial path
        energy = 0
        energy = calc_energy(path=mcpath, rmsd_matrix=rmsd_matrix, cv_matrix=cv_matrix, dataframe=dataframe, odf_matrix=odf_matrix, dos_matrix=dos_matrix)
        
        # propose an exchange of a random structure in the path with a random structure from the pool
        new_mcpath = mcpath.copy()

        if fixed_endpoints == True:
            start_point = 1
            end_point = len(new_mcpath)-1
        else:
            start_point = 0
            end_point = len(new_mcpath)

        # loop throguh new_mcpath
        for point in range(start_point, end_point):

            new_mcpath = mcpath.copy()
            
            structure_in_path = mcpath[point]

            # select a random structure in the same bin as the structure in the path
            random_structure_in_pool = np.random.choice(dataframe[dataframe['bin'] == dataframe.loc[structure_in_path]['bin']].index.values)

            # replace the random structure in the new path with the random structure from the pool
            new_mcpath[np.where(new_mcpath == structure_in_path)[0][0]] = random_structure_in_pool

            # calculate the energy of the original path
            energy = 0
            energy = calc_energy(mcpath, rmsd_matrix=rmsd_matrix, cv_matrix=cv_matrix, dataframe=dataframe, odf_matrix=odf_matrix, dos_matrix=dos_matrix)

            # calcualte energy of proposed path
            new_energy = 0
            new_energy = calc_energy(path=new_mcpath, rmsd_matrix=rmsd_matrix, cv_matrix=cv_matrix, dataframe=dataframe, odf_matrix=odf_matrix, dos_matrix=dos_matrix)

            # calculate the difference between the two energies
            delta_energy = 0
            delta_energy = new_energy - energy

            # metropolis criterion

            # if lower, accept the new path
            if delta_energy < 0:
                descision = 'accepted'
                mcpath = new_mcpath
                path_energies[new_energy] = mcpath
            else:
            
                # if higher, accept with probability e^(-difference/temp)
                if np.random.rand() < np.exp(-delta_energy / temperature):
                    descision = 'accepted'
                    mcpath = new_mcpath
                    path_energies[new_energy] = mcpath
                else:
                    descision = 'rejected'
                    pass
            
            # for debugging - do not uncomment this when running in parallel, will break your editor
            #print('dE:',delta_energy,' T:', temperature, ' rand:', np.exp(-delta_energy / temperature), descision)
            
    # get the path with the lowest energy
    final_energy = min(path_energies.keys())
    
    final_path = path_energies[final_energy]
    final_path_structures = tuple(dataframe.loc[final_path]['structure'].values)

    if debug == True:
        relaxation_energies = list(path_energies.keys())
    else:
        relaxation_energies = None

    gc.collect()
    return [final_energy, final_path, final_path_structures, relaxation_energies]
