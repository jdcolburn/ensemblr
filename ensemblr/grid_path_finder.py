import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import MDAnalysis as mda
import MDAnalysis.analysis.align
from tqdm import tqdm

from .monte_carlo import mc_grid_optimisation


class GridPathFinder:
    """
    Assigns structures to a regular 2D grid in (PC1, PC2) space for
    2D umbrella sampling.

    Bins structures into a rectangular grid, then uses Monte Carlo
    simulated annealing to minimise the Ramachandran RMSE between
    4-connected grid neighbours, ensuring structural continuity.

    Typical usage::

        gfinder = GridPathFinder(ensemble, n_bins_pc1=12, n_bins_pc2=10)
        gfinder.build_grid()
        gfinder.run(n_runs=50)
        targets = gfinder.get_window_targets()

    Attributes
    ----------
    best_grid : dict  {(i, j): structure_filename}
        Grid assignment from the lowest-energy MC run.
    runs_list : list of [energy, grid, history]
        All MC results, sorted by energy (best first).
    """

    def __init__(self, ensemble, n_bins_pc1=16, n_bins_pc2=16,
                 cv1='PC1', cv2='PC2'):
        """
        Parameters
        ----------
        ensemble : Ensemble
            A fully prepared Ensemble (run_pca and compute_matrices done).
        n_bins_pc1, n_bins_pc2 : int
            Number of grid cells along each axis.
        cv1, cv2 : str
            Columns in ensemble.df for the two collective variables.
        """
        self.ensemble = ensemble
        self.n_bins_pc1 = n_bins_pc1
        self.n_bins_pc2 = n_bins_pc2
        self.cv1 = cv1
        self.cv2 = cv2

        self._bin_members = None    # {(i,j): [structure, ...]}
        self._active_cells = None   # sorted list of populated (i,j)
        self._initial_grid = None   # {(i,j): structure} — closest to cell centre
        self._pc1_centers = None    # shape (n_bins_pc1,)
        self._pc2_centers = None    # shape (n_bins_pc2,)
        self.runs_list = None
        self.best_grid = None

    # ── Grid construction ─────────────────────────────────────────────────────

    def build_grid(self):
        """
        Assign structures to a regular (PC1, PC2) grid using np.digitize.

        Adds 'bin_i' and 'bin_j' columns to ensemble.df and sets the initial
        grid assignment (structure nearest to each cell centre).

        Returns
        -------
        self
        """
        df = self.ensemble.df
        pc1 = df[self.cv1].values
        pc2 = df[self.cv2].values

        pc1_edges = np.linspace(pc1.min(), pc1.max(), self.n_bins_pc1 + 1)
        pc2_edges = np.linspace(pc2.min(), pc2.max(), self.n_bins_pc2 + 1)

        self._pc1_centers = 0.5 * (pc1_edges[:-1] + pc1_edges[1:])
        self._pc2_centers = 0.5 * (pc2_edges[:-1] + pc2_edges[1:])

        bin_i = np.clip(np.digitize(pc1, pc1_edges) - 1, 0, self.n_bins_pc1 - 1)
        bin_j = np.clip(np.digitize(pc2, pc2_edges) - 1, 0, self.n_bins_pc2 - 1)

        self.ensemble.df['bin_i'] = bin_i
        self.ensemble.df['bin_j'] = bin_j

        self._bin_members = {}
        for s, i, j in zip(df.index, bin_i, bin_j):
            self._bin_members.setdefault((int(i), int(j)), []).append(s)

        self._active_cells = sorted(self._bin_members.keys())

        # initial grid: per cell, pick the structure closest to the cell centre
        self._initial_grid = {}
        for (i, j) in self._active_cells:
            c1 = self._pc1_centers[i]
            c2 = self._pc2_centers[j]
            members = self._bin_members[(i, j)]
            dists = [
                (df.loc[s, self.cv1] - c1) ** 2 + (df.loc[s, self.cv2] - c2) ** 2
                for s in members
            ]
            self._initial_grid[(i, j)] = members[int(np.argmin(dists))]

        n_total = self.n_bins_pc1 * self.n_bins_pc2
        n_active = len(self._active_cells)
        print(f'Grid: {self.n_bins_pc1}×{self.n_bins_pc2} = {n_total} cells, '
              f'{n_active} active ({100 * n_active / n_total:.0f}%)')
        return self

    # ── MC optimisation ───────────────────────────────────────────────────────

    def run(self, n_runs=50, n_steps=1000, num_processes=6,
            reproducible_seeds=False,
            initial_temperature=0.01, cooling_factor=10000,
            wf_rama=1.0, wf_cv=0.0):
        """
        Run MC simulated annealing in parallel across multiple independent runs.

        Minimises a weighted sum of Ramachandran and CV terms over all
        4-connected neighbour pairs in the grid.  Requires ensemble.rama_matrix;
        also requires ensemble.cv_matrix when wf_cv > 0.

        Parameters
        ----------
        n_runs : int
        n_steps : int
            Annealing sweeps per run (each sweep visits every active cell once).
        num_processes : int
        reproducible_seeds : bool
        initial_temperature : float
            Starting temperature in energy² units.
        cooling_factor : float
            final_temperature = initial_temperature / cooling_factor.
        wf_rama : float
            Weight for the Ramachandran distance term.
        wf_cv : float
            Weight for the CV difference term.  Uses ensemble.cv_matrix (the
            primary CV set by compute_matrices).  Default 0.0 — disabled.

        Returns
        -------
        self
        """
        if self._initial_grid is None:
            raise RuntimeError('Call build_grid() before run()')

        rama_matrix = self.ensemble.rama_matrix
        if rama_matrix is None:
            raise RuntimeError(
                'Rama matrix not available — call '
                'ensemble.compute_matrices(step_matrix="rama") first'
            )

        cv_matrix = None
        cv2_matrix = None
        if wf_cv > 0:
            cv_matrix = self.ensemble.cv_matrix
            if cv_matrix is None:
                raise RuntimeError(
                    'cv_matrix not available — call ensemble.compute_matrices() first'
                )
            cv2_matrix = self.ensemble.cv2_matrix  # may be None; combined if present

        df = self.ensemble.df
        struct_to_idx = {s: i for i, s in enumerate(df.index)}
        bin_members = {k: list(v) for k, v in self._bin_members.items()}

        seeds = (list(range(n_runs)) if reproducible_seeds
                 else list(np.random.randint(0, 10000, size=n_runs)))

        args = [
            (seed, self._initial_grid, self._active_cells, rama_matrix,
             struct_to_idx, bin_members, n_steps,
             initial_temperature, cooling_factor, True,
             wf_rama, wf_cv, cv_matrix, cv2_matrix)
            for seed in seeds
        ]

        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.starmap(mc_grid_optimisation, args),
                total=n_runs, desc='MC grid optimisation',
            ))

        self.runs_list = sorted(results, key=lambda x: x[0])
        self.best_grid = self.runs_list[0][1]
        print(f'Best grid energy (Rama RMSE): {self.runs_list[0][0]:.4f}')
        return self

    # ── Results ───────────────────────────────────────────────────────────────

    def get_window_targets(self):
        """
        Return the ordered list of windows from the best grid.

        Returns
        -------
        list of (structure_filename, pc1_target, pc2_target)
            One entry per active cell, ordered by (bin_i, bin_j).
        """
        if self.best_grid is None:
            raise RuntimeError('Call run() before get_window_targets()')
        return [
            (self.best_grid[(i, j)],
             float(self._pc1_centers[i]),
             float(self._pc2_centers[j]))
            for (i, j) in self._active_cells
        ]

    def get_grid_summary(self):
        """
        Return a DataFrame summarising the best-grid window assignments.

        Columns: window_idx, bin_i, bin_j, pc1_target, pc2_target, structure.
        """
        if self.best_grid is None:
            raise RuntimeError('Call run() before get_grid_summary()')
        rows = []
        for idx, (i, j) in enumerate(self._active_cells):
            rows.append({
                'window_idx': idx,
                'bin_i': i,
                'bin_j': j,
                'pc1_target': float(self._pc1_centers[i]),
                'pc2_target': float(self._pc2_centers[j]),
                'structure': self.best_grid[(i, j)],
            })
        return pd.DataFrame(rows)

    def write_grid_summary(self, output_path):
        """Write the grid summary CSV to output_path."""
        self.get_grid_summary().to_csv(output_path, index=False)
        print(f'Grid summary written to {output_path}')
        return self
