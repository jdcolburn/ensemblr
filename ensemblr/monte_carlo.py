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
        Optional orthogonal DOF matrix (e.g. a second PC).
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


def mc_grid_optimisation(seed, initial_grid, active_cells, rama_matrix, struct_to_idx,
                         bin_members, mc_n_steps=1000, initial_temperature=0.01,
                         cooling_factor=10000, debug=True,
                         wf_rama=1.0, wf_cv=0.0, cv_matrix=None, cv2_matrix=None):
    """
    Optimise a 2D grid of structure assignments using Monte Carlo simulated annealing.

    Each active cell holds one structure; the energy is a weighted sum of
    Ramachandran and (optionally) CV difference terms over all 4-connected
    neighbour edges.  Local energy updates are used so each per-cell move
    costs O(neighbours) rather than O(grid).

    Parameters
    ----------
    seed : int
    initial_grid : dict  {(i, j): structure_filename}
        Starting assignment for every active cell.
    active_cells : sequence of (int, int)
        Populated grid cells, ordered as returned by GridPathFinder.build_grid().
    rama_matrix : np.ndarray  (N x N)
        Pairwise Ramachandran distance matrix.
    struct_to_idx : dict  {structure_filename: int}
        Row/column index in the matrices for each structure.
    bin_members : dict  {(i, j): [structure_filename, ...]}
        All structures that fall in each active cell.
    mc_n_steps : int
        Number of annealing sweeps (each sweep visits every active cell once).
    initial_temperature : float
        Starting temperature in energy² units.
    cooling_factor : float
        final_temperature = initial_temperature / cooling_factor.
    debug : bool
        If True, record energy after each sweep.
    wf_rama : float
        Weight for the Ramachandran distance term (default 1.0).
    wf_cv : float
        Weight for the CV difference term (default 0.0 — disabled).
    cv_matrix : np.ndarray or None
        Pairwise primary-CV difference matrix.  Required when wf_cv > 0.
    cv2_matrix : np.ndarray or None
        Pairwise secondary-CV difference matrix (e.g. a second PC).
        Combined with cv_matrix as Euclidean distance: sqrt(cv1² + cv2²).
        Optional even when wf_cv > 0.

    Returns
    -------
    [best_energy, best_grid, energy_history]
        best_energy  : float — weighted RMSE over all active edges.
        best_grid    : dict  {(i, j): structure_filename}
        energy_history : list of float or None
    """
    np.random.seed(seed)

    final_temperature = initial_temperature / cooling_factor
    temperatures = np.logspace(
        np.log10(initial_temperature), np.log10(final_temperature),
        num=mc_n_steps,
    )

    grid = dict(initial_grid)
    active_set = set(map(tuple, active_cells))
    active_list = list(map(tuple, active_cells))

    # precompute 4-connected neighbours for each active cell
    cell_neighbors = {
        cell: [
            (cell[0] + di, cell[1] + dj)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (cell[0] + di, cell[1] + dj) in active_set
        ]
        for cell in active_list
    }

    # count undirected edges (right + down only to avoid double-counting)
    n_edges = sum(
        1
        for (i, j) in active_set
        for ni, nj in [(i + 1, j), (i, j + 1)]
        if (ni, nj) in active_set
    )

    # initialise running weighted sum over all undirected edges — must match delta_sq terms exactly
    running_sq = 0.0
    for (i, j) in active_set:
        ii = struct_to_idx[grid[(i, j)]]
        for ni, nj in [(i + 1, j), (i, j + 1)]:
            if (ni, nj) in active_set:
                jj = struct_to_idx[grid[(ni, nj)]]
                running_sq += wf_rama * float(rama_matrix[ii, jj]) ** 2
                if cv_matrix is not None:
                    cv1 = float(cv_matrix[ii, jj]) ** 2
                    cv2 = float(cv2_matrix[ii, jj]) ** 2 if cv2_matrix is not None else 0.0
                    running_sq += wf_cv * (cv1 + cv2)

    best_sq = running_sq
    best_grid = dict(grid)
    energy_history = []

    for step_idx in range(mc_n_steps):
        temperature = temperatures[step_idx]

        for cell in active_list:
            nb_list = cell_neighbors[cell]
            if not nb_list:
                continue

            structure = grid[cell]
            random_structure = np.random.choice(bin_members[tuple(cell)])

            ii_old = struct_to_idx[structure]
            ii_new = struct_to_idx[random_structure]

            # local delta: only edges incident on this cell change
            delta_sq = 0.0
            for nb in nb_list:
                jj = struct_to_idx[grid[nb]]
                delta_sq += wf_rama * (float(rama_matrix[ii_new, jj]) ** 2
                                       - float(rama_matrix[ii_old, jj]) ** 2)
                if cv_matrix is not None:
                    cv1_new = float(cv_matrix[ii_new, jj]) ** 2
                    cv1_old = float(cv_matrix[ii_old, jj]) ** 2
                    cv2_new = float(cv2_matrix[ii_new, jj]) ** 2 if cv2_matrix is not None else 0.0
                    cv2_old = float(cv2_matrix[ii_old, jj]) ** 2 if cv2_matrix is not None else 0.0
                    delta_sq += wf_cv * ((cv1_new + cv2_new) - (cv1_old + cv2_old))

            if delta_sq < 0 or np.random.rand() < np.exp(-delta_sq / temperature):
                grid[cell] = random_structure
                running_sq += delta_sq

        if running_sq < best_sq:
            best_sq = running_sq
            best_grid = dict(grid)

        if debug:
            rmse = float(np.sqrt(max(running_sq, 0.0) / n_edges)) if n_edges else 0.0
            energy_history.append(rmse)

    best_energy = float(np.sqrt(max(best_sq, 0.0) / n_edges)) if n_edges else 0.0
    gc.collect()
    return [best_energy, best_grid, energy_history if debug else None]
