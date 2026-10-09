import os
import shutil
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import MDAnalysis as mda
import MDAnalysis.analysis.align
from MDAnalysis.analysis.pca import PCA
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from .smart_selector import generate_selection_token
from .metrics import get_rmsd_to_ref, get_sasa, calc_distance, calc_all_distances
from .calc_matrices import pdb_to_coordinates, get_rmsdmat_jax, get_diffmat, get_ramachandran_matrix
from .pdb_fetch import fetch_uniprot_pdb_ids, download_pdb


def _count_atoms(path):
    with open(path) as f:
        return sum(1 for line in f if line.startswith(('ATOM  ', 'HETATM')))


class Ensemble:
    """
    Central data object representing an AlphaFold2 structural ensemble.

    Holds the ensemble dataframe, computed matrices, and PCA results.
    Methods return self to allow chaining.

    Attributes
    ----------
    df : pd.DataFrame
        Indexed by structure filename. Grows as metrics are added.
    rmsd_selection : str
        MDAnalysis selection token used for RMSD / PCA.
    rmsd_matrix : np.ndarray  (N x N)
        Pairwise RMSD matrix, set by compute_matrices().
    cv_matrix : np.ndarray  (N x N)
        Pairwise CV difference matrix, set by compute_matrices().
    cv2_matrix : np.ndarray or None
        Optional second CV difference matrix.
    pca : MDAnalysis PCA object
        Set by run_pca().
    """

    def __init__(self, structure_directory, thresh_pLDDT=90):
        self.structure_directory = structure_directory.rstrip('/') + '/'
        self.thresh_pLDDT = thresh_pLDDT
        self.df = None
        self.rmsd_selection = None
        self.rmsd_matrix = None
        self.rama_matrix = None
        self.cv_matrix = None
        self.cv2_matrix = None
        self.pca = None
        self._universe = None       # aligned in-memory MDA universe
        self._ensemble_pdb = None   # path to the written aligned ensemble PDB
        self._ref_structures = []

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_from_af2(self, base_directory, msa_depths, relaxed=True):
        """
        Gather AF2 PDBs from output_* subdirectories, parse pLDDT
        scores from log files, filter by pLDDT and atom count, then build
        the main ensemble dataframe.

        Parameters
        ----------
        base_directory : str
            Root directory containing output_{msa} subdirectories.
        msa_depths : list of str
            MSA depth labels, e.g. ['8-16', '16-32', '32-64'].
        relaxed : bool
            If True (default), load relaxed PDBs (_relaxed_). If False, load
            unrelaxed PDBs (_unrelaxed_).

        Returns
        -------
        self
        """
        tag = '_relaxed_' if relaxed else '_unrelaxed_'
        base_directory = base_directory.rstrip('/') + '/'
        os.makedirs(self.structure_directory, exist_ok=True)

        for msa in msa_depths:
            output_dir = os.path.join(base_directory, 'output_' + msa)
            if not os.path.isdir(output_dir):
                print(f'Warning: output directory not found: {output_dir}')
                continue
            for fname in os.listdir(output_dir):
                if not (fname.endswith('.pdb') and tag in fname):
                    continue
                parts = fname.split('_')
                if 'rank' not in parts:
                    continue
                rank = 'rank_' + parts[parts.index('rank') + 1]
                dest = os.path.join(self.structure_directory, f'msa-{msa}_{rank}.pdb')
                if not os.path.exists(dest):
                    shutil.copy(os.path.join(output_dir, fname), dest)

        structure_and_plddt = {}
        for msa in msa_depths:
            output_dir = os.path.join(base_directory, 'output_' + msa)
            if not os.path.isdir(output_dir):
                continue
            for logfile in [f for f in os.listdir(output_dir) if f.endswith('log.txt')]:
                shutil.copy(os.path.join(output_dir, logfile),
                            os.path.join(self.structure_directory, msa + '_log.txt'))

            log_file = os.path.join(self.structure_directory, msa + '_log.txt')
            if not os.path.exists(log_file):
                print(f'Warning: log file missing for MSA depth {msa}')
                continue

            rank_plddt = {}
            with open(log_file) as f:
                for line in f:
                    if 'rank' not in line:
                        continue
                    lparts = line.split()
                    for token in lparts:
                        if token.startswith('rank_'):
                            try:
                                # token may be 'rank_001_alphafold2_...' — keep only 'rank_NNN'
                                short_rank = '_'.join(token.split('_')[:2])
                                rank_plddt[short_rank] = float(lparts[3].replace('pLDDT=', ''))
                            except (IndexError, ValueError):
                                pass
                            break

            structures_msa = [f for f in os.listdir(self.structure_directory)
                              if f.endswith('.pdb') and msa in f]
            n_skipped = 0
            for structure in structures_msa:
                parts = structure.split('_')
                if 'rank' not in parts:
                    continue
                rank = 'rank_' + parts[parts.index('rank') + 1].split('.')[0]
                plddt = rank_plddt.get(rank)
                if plddt is not None:
                    structure_and_plddt[structure] = plddt
                else:
                    n_skipped += 1
            if n_skipped:
                print(f'  Note: skipped {n_skipped} msa-{msa} file(s) with no log entry '
                      f'(stale files from a previous run — delete the ensemble directory to reset)')

        if not structure_and_plddt:
            raise RuntimeError('No structures with parseable pLDDT found. Check your output directories and log files.')

        self._build_df(structure_and_plddt)
        return self

    def load(self, plddt_dict=None):
        """
        Build the ensemble dataframe from PDB files already in structure_directory.

        Parameters
        ----------
        plddt_dict : dict, optional
            Mapping of filename → pLDDT. If None, all PDBs are loaded
            without pLDDT filtering.

        Returns
        -------
        self
        """
        if plddt_dict is None:
            structures = sorted(f for f in os.listdir(self.structure_directory)
                                if f.endswith('.pdb'))
            plddt_dict = {s: float('inf') for s in structures}
        self._build_df(plddt_dict)
        return self

    def _build_df(self, structure_and_plddt):
        df = pd.DataFrame({'pLDDT': structure_and_plddt})
        df.index.name = 'structure'
        df = df[df['pLDDT'] > self.thresh_pLDDT]

        n_atoms = {s: _count_atoms(self.structure_directory + s) for s in df.index}
        df['n_atoms'] = pd.Series(n_atoms)
        highest = max(n_atoms.values())
        df = df[df['n_atoms'] == highest].drop(columns='n_atoms')

        self.df = df
        print(f'Loaded {len(self.df)} structures (pLDDT > {self.thresh_pLDDT}, {highest} atoms)')

    # ── Selection ──────────────────────────────────────────────────────────────

    def build_selection(self, conserved_residues=None, excluded_residues=None,
                        always_include=None, output_file=None, include_loops=True):
        """
        Generate an MDAnalysis selection token using DSSP on the highest-pLDDT
        structure. Stores result in self.rmsd_selection.

        Parameters
        ----------
        conserved_residues : str, optional
            MDA token for residues whose sidechain atoms should be included.
        excluded_residues : str, optional
            MDA token for residues to exclude (overrides conserved_residues).
        always_include : str, optional
            MDA token for residues to always include by CA.
        output_file : str, optional
            Write the selected atoms to this PDB path for inspection.
        include_loops : bool, optional
            Include loop residues in the selection (default True).

        Returns
        -------
        self
        """
        ref_pdb = self.structure_directory + self.df['pLDDT'].idxmax()
        self.rmsd_selection = generate_selection_token(
            ref_pdb,
            conserved_residues=conserved_residues,
            excluded_residues=excluded_residues,
            explicitly_include=always_include,
            include_loops=include_loops,
        )
        if output_file:
            mda.Universe(ref_pdb).select_atoms(self.rmsd_selection).write(output_file)
        return self

    # ── PCA ───────────────────────────────────────────────────────────────────

    def run_pca(self, n_components=3, output_dir=None, n_morph_frames=20):
        """
        Build a multi-model ensemble PDB, run PCA, and add PC1…PCn columns
        to self.df.

        Parameters
        ----------
        n_components : int
        output_dir : str, optional
            If given, write ensemble.pdb and pca{i}.pdb visualisation files
            here. If None, a temporary file is used for ensemble.pdb.
        n_morph_frames : int
            Number of frames in each pca{i}.pdb morph file (default 20).

        Returns
        -------
        self
        """
        if self.rmsd_selection is None:
            raise RuntimeError('Call build_selection() before run_pca()')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            ensemble_pdb = os.path.join(output_dir, 'ensemble.pdb')
        else:
            fd, ensemble_pdb = tempfile.mkstemp(suffix='.pdb')
            os.close(fd)

        n_atoms_ref = _count_atoms(self.structure_directory + self.df.index[0])
        with mda.Writer(ensemble_pdb, n_atoms_ref) as W:
            for structure in self.df.index:
                u = mda.Universe(self.structure_directory + structure,
                                 self.structure_directory + structure)
                u.atoms.segments.segids = 'A'
                u.atoms.chainIDs = 'A'
                W.write(u.select_atoms('all'))

        u = mda.Universe(ensemble_pdb)
        mda.analysis.align.AlignTraj(u, u, select=self.rmsd_selection, in_memory=True).run()

        # overwrite with aligned coordinates
        with mda.Writer(ensemble_pdb, u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)

        pc = PCA(u, select=self.rmsd_selection, align=True, mean=None, n_components=None).run()
        pc_projection = pc.transform(u.select_atoms(self.rmsd_selection), n_components=n_components)

        for i in range(n_components):
            self.df[f'PC{i+1}'] = pc_projection[:, i]

        if output_dir:
            for i in range(n_components):
                pc_n = pc.p_components[:, i]
                trans_n = pc_projection[:, i]
                t_values = np.linspace(trans_n.min(), trans_n.max(), n_morph_frames)
                projected = np.outer(t_values, pc_n) + pc.mean.flatten()
                coordinates = projected.reshape(n_morph_frames, -1, 3)
                proj = mda.Merge(u.select_atoms(self.rmsd_selection))
                proj.load_new(coordinates)
                with mda.Writer(os.path.join(output_dir, f'pca{i+1}.pdb'), proj.atoms.n_atoms) as W:
                    for ts in proj.trajectory:
                        W.write(proj.atoms)

        self.pca = pc
        self._universe = u
        self._ensemble_pdb = ensemble_pdb
        cumvar = {i+1: round(float(v) * 100, 1)
                  for i, v in enumerate(pc.cumulated_variance[:n_components])}
        print(f'PCA complete. Cumulative variance (%): {cumvar}')
        return self

    # ── Clustering ────────────────────────────────────────────────────────────

    def cluster(self, pc_columns=None, min_samples_frac=0.1):
        """
        Run HDBSCAN on PCA space and add a 'cluster' column to self.df.
        Stores cluster medoid filenames as the default reference structures.

        Parameters
        ----------
        pc_columns : list of str, optional
            Columns to cluster on. Defaults to all PC* columns.
        min_samples_frac : float
            Fraction of ensemble size used as HDBSCAN min_samples.

        Returns
        -------
        list of str
            Medoid structure filenames.
        """
        if pc_columns is None:
            pc_columns = sorted(c for c in self.df.columns if c.startswith('PC'))
        min_samples = max(1, int(len(self.df) * min_samples_frac))
        cl = HDBSCAN(min_samples=min_samples, store_centers='medoid').fit(self.df[pc_columns])
        self.df['cluster'] = cl.labels_

        n_clusters = len(set(cl.labels_)) - (1 if -1 in cl.labels_ else 0)
        representatives = []
        for medoid in cl.medoids_:
            mask = (self.df[pc_columns] == medoid).all(axis=1)
            representatives.append(self.df[mask].index[0])

        self._ref_structures = representatives
        print(f'{n_clusters} clusters. Medoids: {representatives}')
        return representatives

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_metrics(self, ref_structures=None, distances=None,
                        compute_sasa=True, num_processes=6):
        """
        Compute per-structure metrics and add them as columns to self.df.

        Parameters
        ----------
        ref_structures : list of str, optional
            Filenames (in structure_directory) to measure RMSD against.
            Defaults to cluster medoids set by cluster().
        distances : dict, optional
            {column_name: (mda_selection_1, mda_selection_2)} pairs.
        compute_sasa : bool
        num_processes : int

        Returns
        -------
        self
        """
        structures = list(self.df.index)
        if ref_structures is None:
            ref_structures = self._ref_structures

        for ref in (ref_structures or []):
            with Pool(processes=num_processes) as pool:
                rmsds = list(tqdm(
                    pool.starmap(get_rmsd_to_ref, [
                        (self.structure_directory + s,
                         self.structure_directory + ref,
                         self.rmsd_selection)
                        for s in structures
                    ]),
                    total=len(structures), desc=f'RMSD → {ref}'
                ))
            rmsd_dict = {os.path.basename(r[0]): r[1] for r in rmsds}
            self.df[f'rmsd_to_{ref}'] = [rmsd_dict[s] for s in self.df.index]

        if compute_sasa:
            with Pool(processes=num_processes) as pool:
                sasas = list(tqdm(
                    pool.imap(get_sasa, [self.structure_directory + s for s in structures]),
                    total=len(structures), desc='SASA'
                ))
            sasa_dict = {os.path.basename(r[0]): r[1] for r in sasas}
            self.df['sasa'] = [sasa_dict[s] for s in self.df.index]

        if distances:
            dist_items = [(name, sel1, sel2) for name, (sel1, sel2) in distances.items()]
            with Pool(processes=num_processes) as pool:
                dist_results = list(tqdm(
                    pool.starmap(calc_all_distances, [
                        (self.structure_directory + s, dist_items)
                        for s in structures
                    ]),
                    total=len(structures), desc='Distances'
                ))
            dist_lookup = {os.path.basename(path): {n: v for n, v in pairs}
                           for path, pairs in dist_results}
            for name, _, _ in dist_items:
                self.df[name] = [dist_lookup[s][name] for s in self.df.index]

        return self

    def get_closest_structure(self, ref_path, ref_resid_offset=0, num_processes=6):
        """
        Find the ensemble structure with the lowest RMSD to an external reference PDB.

        Parameters
        ----------
        ref_path : str
            Absolute path to the reference PDB.
        ref_resid_offset : int
            Residue ID offset applied to the reference (use when numbering differs
            from the ensemble).
        num_processes : int

        Returns
        -------
        (structure_name, rmsd) : tuple of (str, float)
        """
        structures = list(self.df.index)
        with Pool(processes=num_processes) as pool:
            rmsds = list(pool.starmap(get_rmsd_to_ref, [
                (self.structure_directory + s, ref_path,
                 self.rmsd_selection, 0, ref_resid_offset)
                for s in structures
            ]))
        rmsd_values = {os.path.basename(r[0]): r[1] for r in rmsds}
        closest = min(rmsd_values, key=rmsd_values.get)
        return closest, rmsd_values[closest]

    def compute_reference_rmsds(self, references, ref_resid_offset=0, num_processes=6):
        """
        Compute RMSD from every ensemble structure to each user-specified external reference.

        Parameters
        ----------
        references : dict {str: str}
            Mapping of column label → absolute path of reference PDB.
            Adds column 'rmsd_to_{label}' to self.df for each entry.
        ref_resid_offset : int
            Residue ID offset applied to the reference structures (use when the
            reference numbering differs from the ensemble).
        num_processes : int

        Returns
        -------
        self
        """
        structures = list(self.df.index)
        for label, ref_path in references.items():
            col = f'rmsd_to_{label}'
            with Pool(processes=num_processes) as pool:
                rmsds = list(tqdm(
                    pool.starmap(get_rmsd_to_ref, [
                        (self.structure_directory + s, ref_path,
                         self.rmsd_selection, 0, ref_resid_offset)
                        for s in structures
                    ]),
                    total=len(structures), desc=f'RMSD → {label}'
                ))
            rmsd_dict = {os.path.basename(r[0]): r[1] for r in rmsds}
            self.df[col] = [rmsd_dict[s] for s in self.df.index]
        return self

    def add_uniprot_references(self, uniprot_id, chain=None, resolution_cutoff=None,
                               ref_resid_offset=0, num_processes=6, compute_rmsds=True):
        """
        Fetch all PDB structures for a UniProt accession, download them, and
        compute RMSDs from every ensemble structure to each one.

        Downloaded PDBs are saved to {structure_directory}/pdb_refs/ and columns
        'rmsd_to_{pdb_id}_{chain}' are added to self.df (if compute_rmsds=True).

        Parameters
        ----------
        uniprot_id : str
            UniProt accession, e.g. 'P12345'.
        chain : str or list of str, optional
            Only include entries that contain this chain ID (e.g. 'A').
            Defaults to all chains.
        resolution_cutoff : float, optional
            Exclude structures with resolution worse than this value (in Å).
            NMR entries (resolution=None) are always kept unless you set
            a cutoff and pass ``keep_nmr=False`` (see below).
        ref_resid_offset : int
            Residue ID offset applied to reference structures.
        num_processes : int
        compute_rmsds : bool
            If True (default), immediately run compute_reference_rmsds.
            Set to False to only download and return the label→path dict.

        Returns
        -------
        dict of {str: str}
            Mapping of column label → absolute path for each downloaded PDB,
            suitable for passing to compute_reference_rmsds().
        """
        entries = fetch_uniprot_pdb_ids(uniprot_id)
        print(f'{len(entries)} PDB entries found for {uniprot_id}')

        if chain is not None:
            chain_filter = {chain} if isinstance(chain, str) else set(chain)
            entries = [e for e in entries if chain_filter & set(e['chains'])]
            print(f'{len(entries)} entries after chain filter ({chain})')

        if resolution_cutoff is not None:
            entries = [e for e in entries
                       if e['resolution'] is None or e['resolution'] <= resolution_cutoff]
            print(f'{len(entries)} entries after resolution cutoff ({resolution_cutoff} Å)')

        ref_dir = os.path.join(self.structure_directory, 'pdb_refs')
        os.makedirs(ref_dir, exist_ok=True)

        references = {}
        for entry in entries:
            pdb_id = entry['pdb_id']
            entry_chains = entry['chains'] if entry['chains'] else ['']
            for ch in entry_chains:
                if chain is not None:
                    chain_filter = {chain} if isinstance(chain, str) else set(chain)
                    if ch not in chain_filter:
                        continue
                label = f'{pdb_id}_{ch}' if ch else pdb_id
                dest = os.path.join(ref_dir, f'{label}.pdb')
                if not os.path.exists(dest):
                    try:
                        download_pdb(pdb_id, dest)
                        print(f'  Downloaded {pdb_id} → {dest}')
                    except Exception as exc:
                        print(f'  Warning: could not download {pdb_id}: {exc}')
                        continue
                else:
                    print(f'  {dest} already exists, skipping download')
                references[label] = dest

        print(f'{len(references)} structures ready for RMSD calculation')

        if compute_rmsds and references:
            self.compute_reference_rmsds(references, ref_resid_offset=ref_resid_offset,
                                         num_processes=num_processes)
        return references

    def project_structure(self, pdb_path, resid_offset=0):
        """
        Project a single external PDB into the ensemble's PC space.

        Aligns the structure to the ensemble reference frame (first frame of
        ensemble.pdb), then applies (coords − mean) @ p_components.

        Parameters
        ----------
        pdb_path : str
            Absolute path to the PDB file.
        resid_offset : int
            Residue ID offset applied to the external structure.

        Returns
        -------
        np.ndarray
            PC scores, shape (n_components,).
        """
        if self.pca is None:
            raise RuntimeError('Call run_pca() before projecting structures.')
        if self._ensemble_pdb is None or not os.path.exists(self._ensemble_pdb):
            raise RuntimeError('ensemble.pdb not found — run run_pca() with output_dir set.')

        mobile = mda.Universe(pdb_path, pdb_path)
        if resid_offset:
            mobile.atoms.residues.resids += resid_offset
        ref = mda.Universe(self._ensemble_pdb)
        ref.trajectory[0]

        mda.analysis.align.alignto(mobile, ref, select=self.rmsd_selection)

        coords = mobile.select_atoms(self.rmsd_selection).positions.flatten()
        mean = np.asarray(self.pca.mean).flatten()
        p_components = np.asarray(self.pca.p_components)
        return (coords - mean) @ p_components

    def project_structures(self, references, resid_offset=0):
        """
        Project a set of external PDB structures into the ensemble's PC space.

        Parameters
        ----------
        references : dict {str: str}
            Mapping of label → absolute path, as returned by add_uniprot_references().
        resid_offset : int
            Residue ID offset applied to all reference structures.

        Returns
        -------
        pd.DataFrame
            Indexed by label, with columns PC1, PC2, … for each component.
        """
        n_components = np.asarray(self.pca.p_components).shape[1]
        pc_cols = [f'PC{i+1}' for i in range(n_components)]
        rows = {}
        for label, path in references.items():
            try:
                scores = self.project_structure(path, resid_offset=resid_offset)
                rows[label] = dict(zip(pc_cols, scores))
            except Exception as exc:
                print(f'  Warning: could not project {label}: {exc}')
        return pd.DataFrame(rows).T[pc_cols]

    # ── Matrices ──────────────────────────────────────────────────────────────

    def compute_matrices(self, cv_column='PC1', cv2_column=None, step_matrix='rmsd', sparse=False):
        """
        Compute the pairwise step-cost matrix and CV difference matrix.
        Stores results in self.cv_matrix (and optionally self.cv2_matrix), plus
        either self.rmsd_matrix or self.rama_matrix depending on step_matrix.

        Parameters
        ----------
        cv_column : str
            Column in self.df to use as the primary CV.
        cv2_column : str, optional
            Column for an optional second CV (stored as self.cv2_matrix).
        step_matrix : {'rmsd', 'rama'}
            Which pairwise step-cost matrix to compute.
        sparse : bool
            If True, only compute matrix entries for structures in adjacent bins
            (ordered by mean CV), leaving all other entries as 0. Requires that
            MCPathFinder.bin_structures() has already been called so that self.df
            contains a 'bin' column. Default False (full dense matrix).

        Returns
        -------
        self
        """
        if self._ensemble_pdb is None:
            raise RuntimeError('Call run_pca() before compute_matrices()')
        if sparse and 'bin' not in self.df.columns:
            raise RuntimeError(
                'sparse=True requires bin assignments — call MCPathFinder.bin_structures() first'
            )

        self.cv_matrix = get_diffmat(self.df, cv_column)
        if cv2_column:
            self.cv2_matrix = get_diffmat(self.df, cv2_column)

        if step_matrix == 'rama':
            self.rama_matrix = get_ramachandran_matrix(
                self.df, self.structure_directory,
                sparse=sparse, cv_column=cv_column,
            )
            print(f'Matrices computed — Rama: {self.rama_matrix.shape}, '
                  f'CV ({cv_column}): {self.cv_matrix.shape}')
        else:
            coords = pdb_to_coordinates(self._ensemble_pdb)
            self.rmsd_matrix = get_rmsdmat_jax(
                coords, self.df,
                sparse=sparse, cv_column=cv_column,
            )
            print(f'Matrices computed — RMSD: {self.rmsd_matrix.shape}, '
                  f'CV ({cv_column}): {self.cv_matrix.shape}')
        return self
