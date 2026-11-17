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
from .calc_matrices import (pdb_to_coordinates, get_rmsdmat_jax, get_diffmat,
                            get_ramachandran_matrix, get_janin_matrix)
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
    janin_matrix : np.ndarray or None
        Optional pairwise sidechain (χ1/χ2) distance matrix, set by
        compute_matrices(janin=True).
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
        self.janin_matrix = None
        self.pca = None
        self._universe = None       # aligned in-memory MDA universe
        self._universe_names = None # structure name of each frame of _universe
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
            with open(os.path.join(os.path.dirname(output_file) or '.', 'rmsd_selection.txt'), 'w') as fh:
                fh.write(self.rmsd_selection + '\n')
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
        self._universe_names = list(self.df.index)
        self._ensemble_pdb = ensemble_pdb
        if output_dir:
            with open(os.path.join(output_dir, 'rmsd_selection.txt'), 'w') as fh:
                fh.write(self.rmsd_selection + '\n')
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

    # ── Saving / reopening an analysis ────────────────────────────────────────

    def save(self, run_dir):
        """
        Save the dataframe, PCA and selection to ``run_dir`` so the analysis can be reopened
        with :meth:`from_run` (the same files the template notebooks checkpoint to):
        ``ensemble_df.parquet``, ``pca_checkpoint.npz``, ``rmsd_selection.txt``.
        """
        os.makedirs(run_dir, exist_ok=True)
        self.df.to_parquet(os.path.join(run_dir, 'ensemble_df.parquet'))
        if self.rmsd_selection:
            with open(os.path.join(run_dir, 'rmsd_selection.txt'), 'w') as fh:
                fh.write(self.rmsd_selection + '\n')
        if self.pca is not None:
            pc_cols = self._pc_columns()
            np.savez(
                os.path.join(run_dir, 'pca_checkpoint.npz'),
                p_components=np.asarray(self.pca.p_components),
                mean=np.asarray(self.pca.mean).flatten(),
                cumulated_variance=np.asarray(list(self.pca.cumulated_variance)),
                variance=np.asarray(list(self.pca.variance[:len(pc_cols)])),
                **{c: self.df[c].values for c in pc_cols},
            )
        return self

    @classmethod
    def from_run(cls, run_dir, structure_directory=None, rmsd_selection=None):
        """
        Reopen an analysed ensemble from a notebook run directory (e.g.
        ``synthetic_ensemble/run_01/``), without rerunning PCA or the metrics.

        Reads ``ensemble_df.parquet`` (PCs, clusters, metrics), ``pca_checkpoint.npz`` (the PCA
        model, and the PC columns if the parquet lacks them) and ``rmsd_selection.txt``.

        Parameters
        ----------
        run_dir : str
        structure_directory : str, optional
            Where the PDBs are. Default: ``<run_dir>/../structures/`` (the generator layout).
        rmsd_selection : str, optional
            Selection to superpose on when writing structures. Default: ``rmsd_selection.txt``
            in run_dir, else the generator's ``pca_selection.txt``, else all Cα.
        """
        import types
        run_dir = os.path.abspath(run_dir)
        if structure_directory is None:
            structure_directory = os.path.join(os.path.dirname(run_dir), 'structures')
        ens = cls(structure_directory, thresh_pLDDT=0)

        df_path = os.path.join(run_dir, 'ensemble_df.parquet')
        npz_path = os.path.join(run_dir, 'pca_checkpoint.npz')
        if not os.path.exists(df_path):
            raise FileNotFoundError(f'{df_path} not found (save an analysis with Ensemble.save)')
        ens.df = pd.read_parquet(df_path)
        ens.df.index.name = 'structure'

        if os.path.exists(npz_path):
            z = np.load(npz_path)
            for c in sorted((k for k in z.files if k.startswith('PC')), key=lambda k: int(k[2:])):
                if c not in ens.df.columns and len(z[c]) == len(ens.df):
                    ens.df[c] = z[c]
            cum = z['cumulated_variance']
            ens.pca = types.SimpleNamespace(
                p_components=z['p_components'], mean=z['mean'], cumulated_variance=cum,
                variance=(z['variance'] if 'variance' in z.files
                          else np.diff(np.concatenate([[0], cum]))),
            )

        for path in (os.path.join(run_dir, 'rmsd_selection.txt'),
                     os.path.join(os.path.dirname(run_dir), 'pca_selection.txt')):
            if rmsd_selection is None and os.path.exists(path):
                with open(path) as fh:
                    rmsd_selection = fh.read().strip() or None
        ens.rmsd_selection = rmsd_selection

        pdb = os.path.join(run_dir, 'ensemble.pdb')
        ens._ensemble_pdb = pdb if os.path.exists(pdb) else None

        missing = [s for s in ens.df.index[:50] if not os.path.exists(ens.structure_directory + s)]
        if missing:
            print(f'Warning: e.g. {missing[0]} is not in {ens.structure_directory}')
        print(f'Opened {len(ens.df)} structures from {run_dir} '
              f'(columns: {", ".join(ens.df.columns)})')
        return ens

    # ── Accessing structures ──────────────────────────────────────────────────
    #
    # ens.df is the table of everything (one row per structure, indexed by file name).
    # Anything that picks rows of it (a pandas query, nlargest, a boolean mask, select(),
    # nearest(), representatives()) can be passed to paths() or write().

    def _pc_columns(self):
        return sorted((c for c in self.df.columns if c.startswith('PC') and c[2:].isdigit()),
                      key=lambda c: int(c[2:]))

    def _names(self, structures=None):
        """Structure names from a DataFrame/Series/mask/Index/list/str (None = all)."""
        if structures is None:
            return list(self.df.index)
        if isinstance(structures, str):
            names = [structures]
        elif isinstance(structures, pd.DataFrame):
            names = list(structures.index)
        elif isinstance(structures, pd.Series):
            if structures.dtype == bool:
                names = list(structures[structures].index)
            elif structures.index.equals(pd.RangeIndex(len(structures))):
                names = list(structures.values)     # a plain list of names
            else:
                names = list(structures.index)      # e.g. ens.df['PC1'].nlargest(10)
        elif isinstance(structures, np.ndarray) and structures.dtype == bool:
            names = list(self.df.index[structures])
        else:
            names = [os.path.basename(str(s)) for s in structures]
        unknown = [s for s in names if s not in self.df.index]
        if unknown:
            raise KeyError(f'{len(unknown)} structure(s) not in ens.df, e.g. {unknown[0]!r}')
        return names

    def paths(self, structures=None):
        """Absolute PDB paths of ``structures`` (any selection of ens.df rows; None = all)."""
        return [os.path.abspath(self.structure_directory + s) for s in self._names(structures)]

    def select(self, query=None, **ranges):
        """
        Rows of ens.df within ranges of any columns.

        ``ens.select(PC1=(2, None), PC2=(-1, 1))``    PC1 ≥ 2 and −1 ≤ PC2 ≤ 1 (None = open end)
        ``ens.select(cluster=3)``                       equality
        ``ens.select(cluster=[1, 2])``                  membership
        ``ens.select('PC1 > 2 and pLDDT > 85')``        a pandas query string
        ``ens.select({'rmsd_to_IF': (None, 2)})``       a dict, for column names that aren't
                                                        valid Python keywords
        Returns a DataFrame (a view of ens.df rows), which paths() and write() accept.
        """
        if isinstance(query, dict):
            ranges = {**query, **ranges}
            query = None
        mask = pd.Series(True, index=self.df.index)
        for col, r in ranges.items():
            if col not in self.df.columns:
                raise KeyError(f'no column {col!r} in ens.df (have: {", ".join(self.df.columns)})')
            v = self.df[col]
            if isinstance(r, tuple) and len(r) == 2:
                lo, hi = r
                if lo is not None:
                    mask &= v >= lo
                if hi is not None:
                    mask &= v <= hi
            elif isinstance(r, (list, set, frozenset)):
                mask &= v.isin(list(r))
            else:
                mask &= v == r
        out = self.df[mask]
        if query:
            out = out.query(query)
        return out

    def nearest(self, n=1, scale=False, **point):
        """
        The ``n`` structures closest to a point, e.g. ``ens.nearest(PC1=3.0, PC2=-1.5, n=5)``.

        Distance is Euclidean over the named columns; ``scale=True`` divides each column by its
        standard deviation first (use it when mixing units, e.g. a PC and a distance).
        Returns the rows sorted by distance, with a ``distance`` column added.
        """
        if not point:
            raise ValueError('give the point as column=value, e.g. nearest(PC1=3.0, PC2=-1.5)')
        cols = list(point)
        X = self.df[cols].to_numpy(float)
        target = np.array([point[c] for c in cols], float)
        if scale:
            sd = X.std(axis=0)
            sd[sd == 0] = 1.0
            X, target = X / sd, target / sd
        d = np.linalg.norm(X - target, axis=1)
        order = np.argsort(d)[:n]
        out = self.df.iloc[order].copy()
        out['distance'] = d[order]
        return out

    def representatives(self, by='cluster', columns=None, n_bins=None, skip=(-1,)):
        """
        One representative structure per group.

        by : a column of labels (default 'cluster', from cluster() or your own clustering):
            the member closest to the group centroid in ``columns`` (default: all PC columns).
            Labels in ``skip`` are left out (HDBSCAN noise is −1).
        n_bins : int, optional
            Instead treat ``by`` as continuous (e.g. 'PC1'), cut its range into n_bins equal
            bins, and take the structure closest to each bin centre (empty bins are left out).

        Returns the representative rows, with ``group`` and ``n_members`` columns added.
        """
        df = self.df
        rows, groups, sizes = [], [], []
        if n_bins is not None:
            v = df[by].to_numpy(float)
            edges = np.linspace(v.min(), v.max(), n_bins + 1)
            idx = np.clip(np.digitize(v, edges[1:-1]), 0, n_bins - 1)
            for b in range(n_bins):
                members = np.flatnonzero(idx == b)
                if len(members) == 0:
                    continue
                centre = 0.5 * (edges[b] + edges[b + 1])
                rows.append(members[np.argmin(np.abs(v[members] - centre))])
                groups.append(f'{edges[b]:.3g}..{edges[b + 1]:.3g}')
                sizes.append(len(members))
        else:
            if by not in df.columns:
                raise KeyError(f'no column {by!r} in ens.df; run cluster() or add your own labels')
            if columns is None:
                columns = self._pc_columns()
            X = df[columns].to_numpy(float)
            labels = df[by].to_numpy()
            for label in pd.unique(labels):
                if label in skip:
                    continue
                members = np.flatnonzero(labels == label)
                centroid = X[members].mean(axis=0)
                rows.append(members[np.argmin(np.linalg.norm(X[members] - centroid, axis=1))])
                groups.append(label)
                sizes.append(len(members))
        out = df.iloc[rows].copy()
        out['group'] = groups
        out['n_members'] = sizes
        if n_bins is None:
            out = out.sort_values('group')
        return out

    @property
    def universe(self):
        """
        All structures of ens.df as one aligned in-memory MDAnalysis Universe, one frame per row
        (frame i = ens.df.index[i] at the time it was built). Built by run_pca(), otherwise on
        first use from the run's structures.xtc (or, without one, by reading every PDB).
        """
        if self._universe is None or self._universe_names != list(self.df.index):
            from MDAnalysis.coordinates.memory import MemoryReader
            names = list(self.df.index)
            coords = np.empty((len(names), self._reference().atoms.n_atoms, 3), dtype=np.float32)
            for i, ag in enumerate(self._iter_aligned(names, selection='all', align=True)):
                coords[i] = ag.positions
            self._universe = mda.Universe(self.structure_directory + names[0], coords,
                                          format=MemoryReader)
            self._universe_names = names
        return self._universe

    def _reference(self):
        """Superposition reference: the first structure (frame 0 of ensemble.pdb)."""
        if getattr(self, '_ref_universe', None) is None:
            self._ref_universe = mda.Universe(self.structure_directory + self.df.index[0])
        return self._ref_universe

    def _run_trajectory(self):
        """
        The generator's ``structures.xtc`` (next to the structures directory) opened as one
        Universe, with {file name: frame}, or None if there isn't one. Reading frames from it
        is much faster than opening thousands of symlinked PDBs on NFS.
        """
        if getattr(self, '_run_traj', None) is None:
            import warnings
            base = os.path.dirname(os.path.abspath(self.structure_directory.rstrip('/')))
            paths = [os.path.join(base, f) for f in
                     ('structures.pdb', 'structures.xtc', 'structures_index.csv')]
            if not all(os.path.exists(p) for p in paths):
                self._run_traj = False
            else:
                idx = pd.read_csv(paths[2])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    u = mda.Universe(paths[0], paths[1])
                if len(u.trajectory) != len(idx):
                    self._run_traj = False      # stale index; fall back to the PDBs
                else:
                    self._run_traj = (u, dict(zip(idx['file'], idx['frame'])))
        return self._run_traj or None

    def _iter_aligned(self, names, selection='all', align=True):
        """
        Yield an AtomGroup per name, superposed on the ensemble reference (the first structure).
        Source, fastest first: the in-memory universe, the run's structures.xtc, the PDB files.
        align=False skips the superposition (raw AF2 coordinates from PDBs; the generator's
        alignment from structures.xtc).
        """
        import warnings
        if align and self._universe is not None and self._universe_names is not None:
            frame_of = {s: i for i, s in enumerate(self._universe_names)}
            if all(s in frame_of for s in names):
                u = self._universe
                ag = u.select_atoms(selection)
                for s in names:
                    u.trajectory[frame_of[s]]
                    yield ag
                return
        ref = self._reference()
        align_sel = self.rmsd_selection or 'name CA'
        if len(ref.select_atoms(align_sel)) == 0:
            align_sel = 'name CA'
        traj = self._run_trajectory()
        if traj is not None:
            u, frame_of = traj
            if u.atoms.n_atoms == ref.atoms.n_atoms and all(s in frame_of for s in names):
                ag = u.select_atoms(selection)
                for s in names:
                    u.trajectory[frame_of[s]]
                    if align:
                        mda.analysis.align.alignto(u, ref, select=align_sel, match_atoms=False)
                    yield ag
                return
        for s in names:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                u = mda.Universe(self.structure_directory + s)
            if align:
                mda.analysis.align.alignto(u, ref, select=align_sel, match_atoms=False)
            yield u.select_atoms(selection)

    def write(self, structures, output, selection='all', align=True, index=True):
        """
        Write a set of structures, superposed on the ensemble reference frame.

        structures : anything that picks rows of ens.df (DataFrame from select/nearest/
            representatives/query, a boolean mask, a Series from nlargest, a list of names).
        output :
            ``name.xtc`` (or .dcd/.trr)  →  name.xtc + name.pdb (topology) + name_index.csv
            ``name.pdb``                 →  one multi-model PDB (+ name_index.csv)
            ``some_dir/`` (no extension) →  one PDB per structure, original file names
                                            (+ some_dir/index.csv)
        selection : MDAnalysis selection of atoms to write (default all; e.g. 'protein and name CA').
        align : False writes the raw AF2 coordinates.
        index : write the CSV mapping frame → structure, with all ens.df columns.

        Returns the path written (the trajectory, the PDB or the directory).
        """
        import warnings
        names = self._names(structures)
        if not names:
            raise ValueError('no structures selected')
        root, ext = os.path.splitext(output.rstrip('/'))
        ext = ext.lower()
        rows = self.df.loc[names].copy()
        rows.insert(0, 'frame', range(len(names)))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if ext == '':
                os.makedirs(output, exist_ok=True)
                for s, ag in zip(names, self._iter_aligned(names, selection, align)):
                    ag.write(os.path.join(output, s))
                written, index_path = output, os.path.join(output, 'index.csv')
            else:
                frames = self._iter_aligned(names, selection, align)
                first = next(frames)
                n_atoms = first.n_atoms
                if ext != '.pdb':
                    first.write(root + '.pdb')
                kw = {'multiframe': True} if ext == '.pdb' else {}
                with mda.Writer(output, n_atoms, **kw) as w:
                    w.write(first)
                    for ag in frames:
                        w.write(ag)
                written, index_path = output, root + '_index.csv'
        if index:
            rows.to_csv(index_path)
        print(f'Wrote {len(names)} structures → {os.path.abspath(written)}'
              + (f' (topology {root}.pdb)' if ext not in ('', '.pdb') else ''))
        return written

    # ── Matrices ──────────────────────────────────────────────────────────────

    def compute_matrices(self, cv_column='PC1', cv2_column=None, step_matrix='rmsd', sparse=False,
                         janin=False):
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
        janin : bool
            If True, also compute self.janin_matrix (sidechain χ1/χ2 distances)
            over the residues in self.rmsd_selection, for the optional Janin
            term in the MC objective. Default False.

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

        if janin:
            self.janin_matrix = get_janin_matrix(
                self.df, self.structure_directory, selection=self.rmsd_selection,
                sparse=sparse, cv_column=cv_column,
            )
            print(f'Janin matrix computed: {self.janin_matrix.shape}')
        return self
