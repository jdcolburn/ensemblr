import os
import shutil

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm

from .structure_prep import (align_universe, protein_coordinate_replacer,
                              write_pca_ref_pdb, fix_overlapping_atoms,
                              prevent_threaded_lipids, fix_long_bonds)
from .plumed import plumed_input_writer


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
