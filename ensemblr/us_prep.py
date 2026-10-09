import os
import shutil

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm

from .structure_prep import (align_universe, protein_coordinate_replacer,
                              write_pca_ref_pdb, fix_overlapping_atoms,
                              prevent_threaded_lipids, fix_long_bonds)
from .plumed import plumed_input_writer, plumed_input_writer_2d
from .gmx_stages import run_staged_mdrun


class UmbrellaSamplingPrep:
    """
    Embeds path structures into a membrane/solvent template and writes
    PLUMED input files for umbrella sampling.

    Typical usage::

        prep = UmbrellaSamplingPrep(finder, ensemble,
                                    template_pdb='template.pdb',
                                    output_dir='umbrella_sampling/')
        prep.prepare_windows()
        prep.write_plumed_input()
    """

    def __init__(self, path_finder, ensemble, template_pdb, output_dir):
        """
        Parameters
        ----------
        path_finder : MCPathFinder
            A completed MCPathFinder (run() already called).
        ensemble : Ensemble
            The parent Ensemble (needs pca and rmsd_selection set).
        template_pdb : str
            Path to a template PDB (protein + membrane + solvent) from a
            prior simulation or CHARMM-GUI.
        output_dir : str
            Root directory for all window subdirectories and PLUMED files.
        """
        self.path_finder = path_finder
        self.ensemble = ensemble
        self.template_pdb = template_pdb
        self.output_dir = output_dir.rstrip('/') + '/'
        self._window_dirs = []
        self._u_template = None
        self._ref_for_plumed = None

    def prepare_windows(self, path_rank=0, resid_offset=0,
                        aln_selection='name CA',
                        lipid_selection='resname DLPC',
                        fix_bond_thresh=10.0):
        """
        Create one window_N/ directory per path structure, embed each AF2
        structure into the template, and apply geometry fixes.

        Parameters
        ----------
        path_rank : int
            Which MC run to use (0 = best energy).
        resid_offset : int
            Added to residue IDs when replacing protein coordinates.
        aln_selection : str
            MDA selection for initial alignment of structure onto template.
        lipid_selection : str
            Lipid residue name selection for threaded-lipid fixes.
        fix_bond_thresh : float
            Distance threshold (Å) above which bonds are flagged as too long.

        Returns
        -------
        self
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path_structures = self.path_finder.runs_df.iloc[path_rank]['path_structures']

        u_template = mda.Universe(self.template_pdb, self.template_pdb)
        self._u_template = u_template

        self._window_dirs = []
        for i, structure in enumerate(tqdm(path_structures, desc='Preparing windows')):
            window_dir = os.path.join(self.output_dir, f'window_{i}')
            os.makedirs(window_dir, exist_ok=True)
            self._window_dirs.append(window_dir)

            shutil.copy(self.ensemble.structure_directory + structure, window_dir)

            u = u_template.copy()
            full_path = self.ensemble.structure_directory + structure
            u_structure = mda.Universe(full_path, full_path)
            u_structure = align_universe(u_structure, u, aln_selection)

            protein_coordinate_replacer(
                u_structure, u,
                resid_offset=resid_offset,
                selection_token='protein and chainID A',
            )
            prevent_threaded_lipids(u, selection_token=lipid_selection)
            fix_long_bonds(u, thresh=fix_bond_thresh, fudge_dist=1)
            fix_overlapping_atoms(
                u,
                selection_token='protein or (around 10 protein)',
                tolerance=0.2, step=0.1,
            )
            u.atoms.write(os.path.join(window_dir, f'window_{i}.pdb'))

        self._ref_for_plumed = os.path.join(self._window_dirs[0], 'window_0.pdb')
        return self

    def write_plumed_input(self, path_rank=0, cv_values=None,
                           force_constant=1000, distances_dict=None,
                           resid_offset=0):
        """
        Write per-window plumed.dat files and a combined plumed_multidir.dat.

        Detects whether the CV is a named distance or a PCA projection and
        writes the appropriate PLUMED syntax.

        Parameters
        ----------
        path_rank : int
            Which MC run to use for window values (0 = best).
        cv_values : np.ndarray, optional
            Override the ideal window values computed from the path.
        force_constant : float
            Harmonic restraint force constant (kJ mol⁻¹ nm⁻²).
        distances_dict : dict, optional
            {cv_name: (mda_selection_1, mda_selection_2)} — required when
            the CV is a named distance rather than a PC.
        resid_offset : int
            Offset applied when writing the PCA reference PDB.

        Returns
        -------
        self
        """
        if self._ref_for_plumed is None:
            raise RuntimeError('Call prepare_windows() before write_plumed_input()')

        if cv_values is None:
            _, cv_values = self.path_finder.get_window_values(path_rank)

        cv = self.path_finder.cv

        if distances_dict and cv in distances_dict:
            self._write_plumed_distance_cv(cv_values, distances_dict[cv], force_constant)

        elif cv.startswith('PC') and cv[2:].isdigit():
            self._write_plumed_pca_cv(cv_values, force_constant, resid_offset)

        else:
            print(f"Unknown CV '{cv}' — PLUMED input not written. "
                  "Pass distances_dict or use a PC-based CV.")

        return self

    def run_staged_equilibration(self, stages, topol_path, index_path=None,
                                 gmx='gmx', env=None, maxwarn=5,
                                 mdrun_flags=None, continue_on_error=True):
        """
        Run a chain of grompp + mdrun stages (e.g. EM, then equilibration,
        then production) across every prepared window, against one shared
        topology built externally — a CHARMM-GUI template's topol.top, or a
        constant-pH-retitrated one — rather than a per-window pdb2gmx
        topology.

        Call after prepare_windows().  Each window's own window_N.pdb
        (written by prepare_windows()) is used as the first stage's
        coordinate source.

        Parameters
        ----------
        stages : list of dict
            See gmx_stages.run_staged_mdrun — each dict is
            {'name': str, 'mdp': str, 'grompp_flags': list (optional),
            'mdrun_flags': list (optional)}.  Stage N's output feeds stage
            N+1 as -c/-r/-t automatically.
        topol_path : str
            Path to the shared topol.top — absolute, or relative to each
            window_dir.
        index_path : str, optional
            Path to the shared index file, same convention as topol_path.
        gmx : str
            gmx executable name or full path.
        env : dict, optional
            Environment variables for the gmx subprocess calls.
        maxwarn : int
            Passed to every grompp call.
        mdrun_flags : list of str, optional
            Flags appended to every mdrun call, e.g.
            ['-ntmpi', '1', '-ntomp', '8', '-nb', 'gpu', '-update', 'cpu'].
        continue_on_error : bool
            If True (default), a window that fails a stage is logged and
            left there, but the loop continues with the rest — inspect the
            per-window grompp/mdrun logs afterward to decide what to do
            with any failures, rather than losing progress on every other
            window over one bad case.  Set False to stop at the first
            failure instead.

        Returns
        -------
        dict
            {window_dir: {'ok': bool, 'failed_stage': str or None,
                          'error': str or None}}
        """
        if not self._window_dirs:
            raise RuntimeError(
                'Call prepare_windows() before run_staged_equilibration()')

        initial_structures = [
            os.path.basename(d) + '.pdb' for d in self._window_dirs
        ]

        return run_staged_mdrun(
            self._window_dirs, stages=stages,
            topol_path=topol_path, index_path=index_path,
            initial_structures=initial_structures,
            gmx=gmx, env=env, maxwarn=maxwarn, mdrun_flags=mdrun_flags,
            continue_on_error=continue_on_error,
            desc='Running staged equilibration',
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_plumed_distance_cv(self, cv_values, com_selections, force_constant):
        com1, com2 = com_selections

        plumed_input_writer(
            cv_window_values=cv_values,
            cv_reference_pdb=self._ref_for_plumed,
            molinfo_pdb=self._ref_for_plumed,
            COM_1=com1, COM_2=com2, multidir=True,
            force_constant=force_constant,
            output_file=os.path.join(self.output_dir, 'plumed_multidir.dat'),
        )
        for i, window_dir in enumerate(self._window_dirs):
            plumed_input_writer(
                cv_window_values=cv_values[i],
                cv_reference_pdb=self._ref_for_plumed,
                molinfo_pdb=self._ref_for_plumed,
                COM_1=com1, COM_2=com2, multidir=False,
                force_constant=force_constant,
                output_file=os.path.join(window_dir, 'plumed.dat'),
            )

    def _write_plumed_pca_cv(self, cv_values, force_constant, resid_offset):
        if self.ensemble.pca is None:
            raise RuntimeError('PCA not available on Ensemble — call run_pca() first')
        if self._u_template is None:
            raise RuntimeError('Template universe not set — call prepare_windows() first')

        pc_idx = int(self.path_finder.cv[2:]) - 1
        eigenvector = self.ensemble.pca.p_components[:, pc_idx].reshape(-1, 3)
        pca_mean = self.ensemble.pca.mean.reshape(-1, 3)
        n_pca_atoms = len(eigenvector)

        # build average-structure universe for the PCA reference PDB
        u_avg = self._u_template.copy()
        u_avg.trajectory[0]
        # rmsd_selection uses ensemble residue numbering (no offset); subtract the
        # offset from the template copy so the selection matches the right atoms.
        if resid_offset:
            u_avg.select_atoms('protein').residues.resids -= resid_offset

        template_sel = u_avg.select_atoms(self.ensemble.rmsd_selection)
        n_template = len(template_sel)

        if n_template != n_pca_atoms:
            # Template doesn't cover every residue in rmsd_selection (e.g. the
            # N-terminal region was excluded from the membrane embedding).  Find
            # which of the n_pca_atoms atoms are actually present in the template
            # and restrict the eigenvector / mean to that subset.
            u_ref = mda.Universe(
                self.ensemble.structure_directory + self.ensemble.df.index[0],
                self.ensemble.structure_directory + self.ensemble.df.index[0],
            )
            ens_sel = u_ref.select_atoms(self.ensemble.rmsd_selection)
            tmpl_id_set = set(zip(template_sel.resids.tolist(), template_sel.names.tolist()))
            mask = np.array(
                [(r, n) in tmpl_id_set
                 for r, n in zip(ens_sel.resids.tolist(), ens_sel.names.tolist())]
            )
            if mask.sum() != n_template:
                raise RuntimeError(
                    f"PCA selection has {n_pca_atoms} atoms but template has "
                    f"{n_template} after offset correction; only {mask.sum()} "
                    f"(resid, atom-name) pairs match. Check template PDB coverage "
                    f"and resid_offset."
                )
            print(f"  Note: template covers {n_template}/{n_pca_atoms} PCA atoms — "
                  f"subsetting eigenvector to residues present in template.")
            pca_mean = pca_mean[mask]
            eigenvector = eigenvector[mask]
            n_pca_atoms = n_template

        template_sel.positions = pca_mean
        u_avg.atoms.write(os.path.join(self.output_dir, 'average_structure.pdb'))

        pca_ref_pdb = os.path.join(self.output_dir, 'pca_ref.pdb')
        write_pca_ref_pdb(
            u_structure=u_avg,
            u_template=self._u_template.copy(),  # copy so stored template isn't corrupted
            eigenvector=eigenvector,
            pca_selection_token=self.ensemble.rmsd_selection,
            offset=resid_offset,
            output_file=pca_ref_pdb,
        )

        plumed_input_writer(
            cv_window_values=cv_values,
            cv_reference_pdb=pca_ref_pdb,
            molinfo_pdb=self._ref_for_plumed,
            scale_factor=n_pca_atoms, multidir=True,
            force_constant=force_constant,
            output_file=os.path.join(self.output_dir, 'plumed_multidir.dat'),
        )
        for i, window_dir in enumerate(self._window_dirs):
            plumed_input_writer(
                cv_window_values=cv_values[i],
                cv_reference_pdb=os.path.join(self.output_dir, '../pca_ref.pdb'),
                molinfo_pdb=self._ref_for_plumed,
                scale_factor=n_pca_atoms, multidir=False,
                force_constant=force_constant,
                output_file=os.path.join(window_dir, 'plumed.dat'),
            )


class GridUmbrellaSamplingPrep:
    """
    Embeds grid structures into a membrane/solvent template and writes
    2D PLUMED input files for umbrella sampling on a (PC1, PC2) grid.

    Typical usage::

        prep = GridUmbrellaSamplingPrep(grid_finder, ensemble,
                                        template_pdb='template.pdb',
                                        output_dir='us_2d/')
        prep.prepare_windows()
        prep.write_plumed_input()
    """

    def __init__(self, grid_finder, ensemble, template_pdb, output_dir):
        """
        Parameters
        ----------
        grid_finder : GridPathFinder
            A completed GridPathFinder (run() already called).
        ensemble : Ensemble
            The parent Ensemble (needs pca and rmsd_selection set).
        template_pdb : str
            Path to a template PDB (protein + membrane + solvent).
        output_dir : str
            Root directory for window subdirectories and PLUMED files.
        """
        self.grid_finder = grid_finder
        self.ensemble = ensemble
        self.template_pdb = template_pdb
        self.output_dir = output_dir.rstrip('/') + '/'
        self._window_dirs = []
        self._window_targets = []   # [(structure, pc1_val, pc2_val), ...]
        self._u_template = None
        self._ref_for_plumed = None

    def prepare_windows(self, resid_offset=0, aln_selection='name CA',
                        lipid_selection='resname DLPC', fix_bond_thresh=10.0):
        """
        Create one window_N/ directory per active grid cell, embed each
        structure into the template, and apply geometry fixes.

        Windows are numbered sequentially in (bin_i, bin_j) order; the
        mapping to grid coordinates is written to window_map.csv.

        Parameters
        ----------
        resid_offset : int
        aln_selection : str
        lipid_selection : str
        fix_bond_thresh : float

        Returns
        -------
        self
        """
        os.makedirs(self.output_dir, exist_ok=True)
        targets = self.grid_finder.get_window_targets()
        self._window_targets = targets

        u_template = mda.Universe(self.template_pdb, self.template_pdb)
        self._u_template = u_template
        self._window_dirs = []

        for i, (structure, pc1_val, pc2_val) in enumerate(
                tqdm(targets, desc='Preparing windows')):
            window_dir = os.path.join(self.output_dir, f'window_{i}')
            os.makedirs(window_dir, exist_ok=True)
            self._window_dirs.append(window_dir)

            shutil.copy(self.ensemble.structure_directory + structure, window_dir)

            u = u_template.copy()
            full_path = self.ensemble.structure_directory + structure
            u_structure = mda.Universe(full_path, full_path)
            u_structure = align_universe(u_structure, u, aln_selection)

            protein_coordinate_replacer(
                u_structure, u,
                resid_offset=resid_offset,
                selection_token='protein and chainID A',
            )
            prevent_threaded_lipids(u, selection_token=lipid_selection)
            fix_long_bonds(u, thresh=fix_bond_thresh, fudge_dist=1)
            fix_overlapping_atoms(
                u,
                selection_token='protein or (around 10 protein)',
                tolerance=0.2, step=0.1,
            )
            u.atoms.write(os.path.join(window_dir, f'window_{i}.pdb'))

        # write the grid → window index mapping for downstream analysis
        self.grid_finder.write_grid_summary(
            os.path.join(self.output_dir, 'window_map.csv'))

        self._ref_for_plumed = os.path.join(self._window_dirs[0], 'window_0.pdb')
        return self

    def write_plumed_input(self, force_constant=1000, resid_offset=0):
        """
        Write per-window plumed.dat files and a combined plumed_multidir.dat
        with two PCA restraints (PC1 and PC2).

        Parameters
        ----------
        force_constant : float
            Harmonic restraint force constant (kJ mol⁻¹ nm⁻²).
        resid_offset : int

        Returns
        -------
        self
        """
        if self._ref_for_plumed is None:
            raise RuntimeError('Call prepare_windows() before write_plumed_input()')
        if self.ensemble.pca is None:
            raise RuntimeError('PCA not available on Ensemble — call run_pca() first')

        cv1 = self.grid_finder.cv1
        cv2 = self.grid_finder.cv2
        pc1_idx = int(cv1[2:]) - 1 if cv1.startswith('PC') and cv1[2:].isdigit() else 0
        pc2_idx = int(cv2[2:]) - 1 if cv2.startswith('PC') and cv2[2:].isdigit() else 1

        ev1 = self.ensemble.pca.p_components[:, pc1_idx].reshape(-1, 3)
        ev2 = self.ensemble.pca.p_components[:, pc2_idx].reshape(-1, 3)
        pca_mean = self.ensemble.pca.mean.reshape(-1, 3)
        n_pca_atoms = len(ev1)

        u_avg = self._u_template.copy()
        u_avg.trajectory[0]
        if resid_offset:
            u_avg.select_atoms('protein').residues.resids -= resid_offset

        template_sel = u_avg.select_atoms(self.ensemble.rmsd_selection)
        n_template = len(template_sel)

        if n_template != n_pca_atoms:
            u_ref = mda.Universe(
                self.ensemble.structure_directory + self.ensemble.df.index[0],
                self.ensemble.structure_directory + self.ensemble.df.index[0],
            )
            ens_sel = u_ref.select_atoms(self.ensemble.rmsd_selection)
            tmpl_id_set = set(zip(template_sel.resids.tolist(), template_sel.names.tolist()))
            mask = np.array(
                [(r, n) in tmpl_id_set
                 for r, n in zip(ens_sel.resids.tolist(), ens_sel.names.tolist())]
            )
            if mask.sum() != n_template:
                raise RuntimeError(
                    f"PCA selection has {n_pca_atoms} atoms but template has "
                    f"{n_template} after offset correction; only {mask.sum()} "
                    f"(resid, atom-name) pairs match. Check template PDB coverage "
                    f"and resid_offset."
                )
            print(f"  Note: template covers {n_template}/{n_pca_atoms} PCA atoms — "
                  f"subsetting eigenvectors.")
            pca_mean = pca_mean[mask]
            ev1 = ev1[mask]
            ev2 = ev2[mask]
            n_pca_atoms = n_template

        template_sel.positions = pca_mean
        u_avg.atoms.write(os.path.join(self.output_dir, 'average_structure.pdb'))

        pca_ref_pc1 = os.path.join(self.output_dir, 'pca_ref_pc1.pdb')
        pca_ref_pc2 = os.path.join(self.output_dir, 'pca_ref_pc2.pdb')

        write_pca_ref_pdb(
            u_structure=u_avg, u_template=self._u_template.copy(),
            eigenvector=ev1, pca_selection_token=self.ensemble.rmsd_selection,
            offset=resid_offset, output_file=pca_ref_pc1,
        )
        write_pca_ref_pdb(
            u_structure=u_avg, u_template=self._u_template.copy(),
            eigenvector=ev2, pca_selection_token=self.ensemble.rmsd_selection,
            offset=resid_offset, output_file=pca_ref_pc2,
        )

        pc1_values = np.array([t[1] for t in self._window_targets])
        pc2_values = np.array([t[2] for t in self._window_targets])

        plumed_input_writer_2d(
            cv1_window_values=pc1_values, cv2_window_values=pc2_values,
            cv1_reference_pdb=pca_ref_pc1, cv2_reference_pdb=pca_ref_pc2,
            molinfo_pdb=self._ref_for_plumed,
            scale_factor=n_pca_atoms, force_constant=force_constant,
            multidir=True,
            output_file=os.path.join(self.output_dir, 'plumed_multidir.dat'),
        )
        for i, (window_dir, (_, pc1_val, pc2_val)) in enumerate(
                zip(self._window_dirs, self._window_targets)):
            plumed_input_writer_2d(
                cv1_window_values=pc1_val, cv2_window_values=pc2_val,
                cv1_reference_pdb='../pca_ref_pc1.pdb',
                cv2_reference_pdb='../pca_ref_pc2.pdb',
                molinfo_pdb=self._ref_for_plumed,
                scale_factor=n_pca_atoms, force_constant=force_constant,
                multidir=False,
                output_file=os.path.join(window_dir, 'plumed.dat'),
            )
        return self

    def run_staged_equilibration(self, stages, topol_path, index_path=None,
                                 gmx='gmx', env=None, maxwarn=5,
                                 mdrun_flags=None, continue_on_error=True):
        """
        Run a chain of grompp + mdrun stages (e.g. EM, then equilibration,
        then production) across every prepared window, against one shared
        topology built externally — a CHARMM-GUI template's topol.top, or a
        constant-pH-retitrated one — rather than a per-window pdb2gmx
        topology.

        Call after prepare_windows().  Each window's own window_N.pdb
        (written by prepare_windows()) is used as the first stage's
        coordinate source.

        Parameters
        ----------
        stages : list of dict
            See gmx_stages.run_staged_mdrun — each dict is
            {'name': str, 'mdp': str, 'grompp_flags': list (optional),
            'mdrun_flags': list (optional)}.  Stage N's output feeds stage
            N+1 as -c/-r/-t automatically.
        topol_path : str
            Path to the shared topol.top — absolute, or relative to each
            window_dir.
        index_path : str, optional
            Path to the shared index file, same convention as topol_path.
        gmx : str
            gmx executable name or full path.
        env : dict, optional
            Environment variables for the gmx subprocess calls.
        maxwarn : int
            Passed to every grompp call.
        mdrun_flags : list of str, optional
            Flags appended to every mdrun call, e.g.
            ['-ntmpi', '1', '-ntomp', '8', '-nb', 'gpu', '-update', 'cpu'].
        continue_on_error : bool
            If True (default), a window that fails a stage is logged and
            left there, but the loop continues with the rest — inspect the
            per-window grompp/mdrun logs afterward to decide what to do
            with any failures, rather than losing progress on every other
            window over one bad case.  Set False to stop at the first
            failure instead.

        Returns
        -------
        dict
            {window_dir: {'ok': bool, 'failed_stage': str or None,
                          'error': str or None}}
        """
        if not self._window_dirs:
            raise RuntimeError(
                'Call prepare_windows() before run_staged_equilibration()')

        initial_structures = [
            os.path.basename(d) + '.pdb' for d in self._window_dirs
        ]

        return run_staged_mdrun(
            self._window_dirs, stages=stages,
            topol_path=topol_path, index_path=index_path,
            initial_structures=initial_structures,
            gmx=gmx, env=env, maxwarn=maxwarn, mdrun_flags=mdrun_flags,
            continue_on_error=continue_on_error,
            desc='Running staged equilibration',
        )
