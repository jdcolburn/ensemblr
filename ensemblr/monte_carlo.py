# MC path finding by exchanging structures within bins

import gc
import numpy as np


def calc_energy(path, rmsd_matrix, cv_matrix, struct_to_idx,
                odf_matrix=None, dos_matrix=None,
                wf_rmsd=0.1, wf_cv=1, wf_odf=0.0, wf_dos=0.0):
    """
    Calculate the "energy" of a path.

    Uses plain dict lookups (struct_to_idx) instead of a DataFrame so the
    function is safe to call from multiprocessing workers.

    Parameters
    ----------
    path : sequence of str
        Ordered structure filenames.
    rmsd_matrix : np.ndarray  (N x N)
    cv_matrix : np.ndarray    (N x N)
    struct_to_idx : dict
        {structure_filename: row/col index in the matrices}
    odf_matrix : np.ndarray or None
        Optional orthogonal DOF matrix (e.g. PC2).
    dos_matrix : np.ndarray or None
        Optional density-of-states matrix.
    wf_rmsd, wf_cv, wf_odf, wf_dos : float
        Weighting coefficients for each energy term.

    Returns
    -------
    float
    """
    path_length = len(path)

    energy_rmsd = 0.0
    for i in range(path_length - 1):
        ii = struct_to_idx[path[i]]
        jj = struct_to_idx[path[i + 1]]
        energy_rmsd += rmsd_matrix[ii, jj] ** 2
    energy_rmsd = wf_rmsd * np.sqrt(energy_rmsd / path_length)

    energy_cv = 0.0
    for i in range(path_length - 1):
        ii = struct_to_idx[path[i]]
        jj = struct_to_idx[path[i + 1]]
        energy_cv += cv_matrix[ii, jj] ** 2
    energy_cv = wf_cv * np.sqrt(energy_cv / path_length)

    energy_odf = 0.0
    if odf_matrix is not None:
        for i in range(path_length - 1):
            ii = struct_to_idx[path[i]]
            jj = struct_to_idx[path[i + 1]]
            energy_odf += odf_matrix[ii, jj] ** 2
        energy_odf = wf_odf * np.sqrt(energy_odf / path_length)

    energy_dos = 0.0
    if dos_matrix is not None:
        for i in range(path_length - 1):
            ii = struct_to_idx[path[i]]
            jj = struct_to_idx[path[i + 1]]
            energy_dos += dos_matrix[ii, jj]
        energy_dos = wf_dos * (energy_dos / path_length)

    return energy_rmsd + energy_cv + energy_odf + energy_dos


def mc_path_optimisation(seed, initial_guess_indices,
                         rmsd_matrix, cv_matrix,
                         struct_to_idx, bin_lookup, bin_members,
                         fixed_endpoints=True,
                         odf_matrix=None, dos_matrix=None,
                         mc_n_steps=1000, initial_temperature=0.0001,
                         cooling_factor=10000, debug=True,
                         wf_step=0.1, wf_cv=1.0, wf_odf=0.0):
    """
    Optimize a path using Monte Carlo simulated annealing.

    Parameters
    ----------
    seed : int
    initial_guess_indices : array-like of str
        Initial path as an ordered sequence of structure filenames.
    rmsd_matrix, cv_matrix : np.ndarray  (N x N)
    struct_to_idx : dict
        {structure: matrix index}  — precomputed from ensemble.df.
    bin_lookup : dict
        {structure: bin_id}        — precomputed from ensemble.df['bin'].
    bin_members : dict
        {bin_id: [structure, ...]} — precomputed from ensemble.df['bin'].
    fixed_endpoints : bool
    odf_matrix, dos_matrix : np.ndarray or None
    mc_n_steps : int
    initial_temperature, cooling_factor : float

    Returns
    -------
    [final_energy, final_path, final_path_structures, relaxation_energies]
    """
    np.random.seed(seed)

    final_temperature = initial_temperature / cooling_factor
    temperatures = np.logspace(
        np.log10(initial_temperature), np.log10(final_temperature),
        num=mc_n_steps,
    )

    mcpath = list(initial_guess_indices)   # plain Python list throughout
    path_energies = {}                     # energy → tuple(path)

    for i in range(mc_n_steps):
        temperature = temperatures[i]

        if fixed_endpoints:
            start_point, end_point = 1, len(mcpath) - 1
        else:
            start_point, end_point = 0, len(mcpath)

        for point in range(start_point, end_point):
            structure_in_path = mcpath[point]

            # pick a random structure from the same bin
            random_structure = np.random.choice(bin_members[bin_lookup[structure_in_path]])

            new_mcpath = mcpath.copy()
            new_mcpath[point] = random_structure

            energy     = calc_energy(mcpath,     rmsd_matrix, cv_matrix, struct_to_idx,
                                     odf_matrix, dos_matrix, wf_rmsd=wf_step, wf_cv=wf_cv, wf_odf=wf_odf)
            new_energy = calc_energy(new_mcpath, rmsd_matrix, cv_matrix, struct_to_idx,
                                     odf_matrix, dos_matrix, wf_rmsd=wf_step, wf_cv=wf_cv, wf_odf=wf_odf)
            delta_energy = new_energy - energy

            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                mcpath = new_mcpath
                path_energies[float(new_energy)] = tuple(mcpath)

    if path_energies:
        final_energy = min(path_energies)
        final_path   = list(path_energies[final_energy])
    else:
        # no moves accepted — return the initial path
        final_energy = float(calc_energy(mcpath, rmsd_matrix, cv_matrix, struct_to_idx,
                                         odf_matrix, dos_matrix, wf_rmsd=wf_step, wf_cv=wf_cv, wf_odf=wf_odf))
        final_path   = list(mcpath)

    final_path_structures = tuple(final_path)

    relaxation_energies = sorted(path_energies.keys()) if debug else None

    gc.collect()
    return [final_energy, final_path, final_path_structures, relaxation_energies]
