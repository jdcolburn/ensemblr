import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import MDAnalysis as mda
import MDAnalysis.analysis.align
from sklearn.cluster import KMeans
from tqdm import tqdm

from .monte_carlo import mc_path_optimisation


class MCPathFinder:
    """
    Finds an optimal ordered path through the ensemble for umbrella sampling.

    Bins structures along a collective variable (CV) with KMeans, then uses
    Monte Carlo simulated annealing to minimise path energy (smooth in both
    Cartesian and CV space).

    Attributes
    ----------
    runs_df : pd.DataFrame
        Results of all MC runs, sorted by energy (best first). Columns:
        energy, path, path_structures, relaxation_energies.
    """

    def __init__(self, ensemble, n_bins=32, cv='PC1', cv2=None):
        """
        Parameters
        ----------
        ensemble : Ensemble
            A fully prepared Ensemble (run_pca and compute_matrices done).
        n_bins : int
            Number of umbrella sampling windows / path length.
        cv : str
            Column in ensemble.df to use as the primary collective variable.
        cv2 : str, optional
            Column for an orthogonal DOF (passed as odf_matrix to MC).
        """
        self.ensemble = ensemble
        self.n_bins = n_bins
        self.cv = cv
        self.cv2 = cv2
        self.runs_df = None
        self._initial_path = None
        self._guess_window_values = None

    def bin_structures(self):
        """
        Assign structures to bins along the CV axis using KMeans and build
        an initial path (one representative per bin, ordered by CV value).
        Adds a 'bin' column to ensemble.df.

        Returns
        -------
        self
        """
        kmeans = KMeans(n_clusters=self.n_bins, random_state=1).fit(
            self.ensemble.df[[self.cv]])
        self.ensemble.df['bin'] = kmeans.labels_

        guess_values = np.sort(
            self.ensemble.df.groupby('bin')[self.cv].mean().values)
        initial_path = [self.ensemble.df[self.cv].sub(v).abs().idxmin()
                        for v in guess_values]

        # dtype=object avoids fixed-width string truncation in numpy arrays
        self._initial_path = np.array(initial_path, dtype=object)
        self._guess_window_values = guess_values
        return self

    def run(self, n_runs=100, n_steps=1000, num_processes=6,
            reproducible_seeds=False, fixed_endpoints=True, step_matrix=None,
            wf_step=1.0, wf_cv=1.0, wf_odf=0.0):
        """
        Run MC simulated annealing in parallel and store results in runs_df.

        Parameters
        ----------
        n_runs : int
            Number of independent MC runs.
        n_steps : int
            Cooling steps per run (each step sweeps all path positions).
        num_processes : int
        reproducible_seeds : bool
            If True use seeds 0…n_runs-1, otherwise draw randomly.
        fixed_endpoints : bool
            If True, keep the first and last structures fixed during MC.
        wf_step : float
            Weight for the step-cost (RMSD or Rama) energy term relative to
            the CV term (wf_cv=1). Default 1.0 balances both contributions.

        Returns
        -------
        self
        """
        if self._initial_path is None:
            raise RuntimeError('Call bin_structures() before run()')
        if self.ensemble.cv_matrix is None:
            raise RuntimeError('Call ensemble.compute_matrices() before run()')

        if step_matrix is None:
            step_matrix = (self.ensemble.rama_matrix
                           if self.ensemble.rmsd_matrix is None
                           else self.ensemble.rmsd_matrix)
        if step_matrix is None:
            raise RuntimeError('No step matrix available — call ensemble.compute_matrices() first')

        seeds = (range(n_runs) if reproducible_seeds
                 else np.random.randint(0, 1000, size=n_runs))

        cv2_mat = self.ensemble.cv2_matrix  # may be None

        # Precompute plain-Python lookups so workers receive no DataFrame
        df = self.ensemble.df
        struct_to_idx = {s: i for i, s in enumerate(df.index)}
        bin_lookup    = dict(zip(df.index, df['bin']))
        bin_members   = {}
        for s, b in bin_lookup.items():
            bin_members.setdefault(int(b), []).append(s)

        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(mc_path_optimisation, zip(
                    seeds,
                    [self._initial_path] * n_runs,
                    [step_matrix] * n_runs,
                    [self.ensemble.cv_matrix] * n_runs,
                    [struct_to_idx] * n_runs,
                    [bin_lookup] * n_runs,
                    [bin_members] * n_runs,
                    [fixed_endpoints] * n_runs,
                    [cv2_mat] * n_runs,
                    [None] * n_runs,     # dos_matrix
                    [n_steps] * n_runs,  # mc_n_steps
                    [0.0001] * n_runs,   # initial_temperature
                    [10000] * n_runs,    # cooling_factor
                    [True] * n_runs,     # debug
                    [wf_step] * n_runs,
                    [wf_cv] * n_runs,
                    [wf_odf] * n_runs,
                )),
                total=n_runs, desc='MC optimisation'
            ))

        self.runs_df = (
            pd.DataFrame(results,
                         columns=['energy', 'path', 'path_structures', 'relaxation_energies'])
            .sort_values('energy')
            .reset_index(drop=True)
        )
        print(f'Best path energy: {self.runs_df.iloc[0]["energy"]:.4f}')
        return self

    def get_paths(self, n=5):
        """Return the top n path_structures tuples (best-energy first)."""
        return list(self.runs_df.head(n)['path_structures'])

    def get_window_values(self, path_rank=0):
        """
        Extract CV values along a path and compute the ideal (linearly
        interpolated) target values.

        Parameters
        ----------
        path_rank : int
            Index into runs_df (0 = best).

        Returns
        -------
        actual : np.ndarray  shape (n_bins,)
        ideal : np.ndarray   shape (n_bins,)
        """
        path = self.runs_df.iloc[path_rank]['path']
        actual = np.array([self.ensemble.df.loc[s, self.cv] for s in path])
        ideal = np.interp(
            np.linspace(0, self.n_bins - 1, self.n_bins),
            [0, self.n_bins - 1],
            [actual[0], actual[-1]],
        )
        return actual, ideal

    def write_paths(self, output_dir, n=5, resid_offset=0):
        """
        Write the top n paths as multi-model PDB files.

        Parameters
        ----------
        output_dir : str
        n : int
        resid_offset : int
            Added to residue IDs in written structures.

        Returns
        -------
        self
        """
        os.makedirs(output_dir, exist_ok=True)
        rmsd_sel = self.ensemble.rmsd_selection
        for rank, (_, row) in enumerate(self.runs_df.head(n).iterrows(), start=1):
            path_file = os.path.join(output_dir, f'path_rank_{rank}.pdb')
            first = self.ensemble.structure_directory + row['path_structures'][0]
            ref = mda.Universe(first, first)
            n_protein_atoms = len(ref.select_atoms('protein'))
            with mda.Writer(path_file, n_protein_atoms) as W:
                for structure in row['path_structures']:
                    full = self.ensemble.structure_directory + structure
                    u = mda.Universe(full, full)
                    if rmsd_sel is not None:
                        MDAnalysis.analysis.align.alignto(u, ref, select=rmsd_sel)
                    u.atoms.segments.segids = 'A'
                    u.atoms.chainIDs = 'A'
                    u.atoms.residues.resids += resid_offset
                    W.write(u.select_atoms('protein'))
            print(f'Rank {rank} (energy={row["energy"]:.3f}) → {path_file}')
        return self
