"""
Retroactive importance-sampling reweighting for adaptive AF2 ensembles.

The adaptive pipeline over-samples sparse regions of PC space by using
template-directed ColabFold runs.  compute_adaptive_weights() reconstructs
the unbiased AF2 sampling distribution by computing per-structure importance
weights: w(C) = p0(C) / p_total(C), where C is the final-PCA grid cell each
structure lands in.
"""

import json
import os
import re

import MDAnalysis as mda
import MDAnalysis.analysis.align
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ── Filename parsing ──────────────────────────────────────────────────────────

def _parse_round_info(filename):
    """
    Return (round_idx, nominal_cell) from a structures/ symlink filename.

    Filename patterns (from collect_structures):
      r0_{msa}_{rank}.pdb                     → round 0, no cell
      r0_{NN}_{msa}_{rank}.pdb                → round 0 (anneal_round0 mode)
      r{N}_cell_{ci}_{cj}_{msa}_{rank}.pdb    → adaptive round N, cell (ci, cj)
    """
    if re.match(r'^r0', filename):
        return 0, None
    m = re.match(r'^r(\d+)_cell_(\d+)_(\d+)_', filename)
    if m:
        return int(m.group(1)), (int(m.group(2)), int(m.group(3)))
    return None, None


# ── PCA helpers (mirrors generate_af2_ensemble_v2) ───────────────────────────

def _load_pca_npz(path):
    data = np.load(path, allow_pickle=False)
    n_comp = len(data['explained_variance_ratio'])
    pca = PCA(n_components=n_comp)
    pca.components_ = data['components']
    pca.mean_ = data['mean']
    pca.explained_variance_ratio_ = data['explained_variance_ratio']
    pca.n_features_in_ = int(data['n_features_in'])
    pca.n_components_ = n_comp
    p1_edges    = data['p1_edges']    if 'p1_edges'    in data else None
    p2_edges    = data['p2_edges']    if 'p2_edges'    in data else None
    ref_basename = str(data['ref_basename']) if 'ref_basename' in data else None
    return pca, p1_edges, p2_edges, ref_basename


def _pca_apply(pca, flat):
    return (flat - pca.mean_) @ pca.components_.T


def _resolve_ref(ref_basename, structures_dir):
    """Return the path to use as the PCA alignment reference."""
    if ref_basename:
        candidate = os.path.join(structures_dir, ref_basename)
        if os.path.lexists(candidate):
            return candidate
    return None


def _project_all(structure_paths, ref_path, pca, pca_selection):
    """
    Align every structure in structure_paths to ref_path, then project onto pca.
    Returns (valid_paths, pc_coords).
    """
    ref_u = mda.Universe(os.path.realpath(ref_path))
    n_sel = len(ref_u.select_atoms(pca_selection))

    valid_paths, projected = [], []
    for path in structure_paths:
        try:
            u = mda.Universe(os.path.realpath(path))
            ag = u.select_atoms(pca_selection)
            if len(ag) != n_sel:
                continue
            if os.path.realpath(path) != os.path.realpath(ref_path):
                mda.analysis.align.alignto(u, ref_u,
                                           select=pca_selection,
                                           match_atoms=False)
            flat = u.select_atoms(pca_selection).positions.flatten()
            projected.append(_pca_apply(pca, flat.reshape(1, -1))[0])
            valid_paths.append(path)
        except Exception:
            continue

    return valid_paths, np.array(projected) if projected else np.empty((0, pca.n_components_))


# ── Main public function ──────────────────────────────────────────────────────

def compute_adaptive_weights(output_dir, pca_selection=None, min_p0=0.0):
    """
    Reconstruct the unbiased AF2 sampling distribution from an adaptive run.

    Projects all accepted structures onto the final PCA axes, assigns them to
    final-grid cells, and computes importance weights that correct for the
    over-sampling of sparse cells introduced by template-directed rounds.

    Parameters
    ----------
    output_dir : str
        Root output directory from run_adaptive_ensemble_v2().
    pca_selection : str, optional
        MDAnalysis selection string used during the pipeline PCA.  If None,
        reads from <output_dir>/pca_selection.txt (written automatically by
        recent pipeline versions).
    min_p0 : float, optional
        Floor applied to p0(cell) before computing weights.  Default 0 means
        cells with no round-0 structures get weight=0.  A small positive value
        (e.g. 1e-3) avoids hard zeroing.

    Returns
    -------
    pd.DataFrame
        One row per accepted structure.  Columns:

          path          absolute path (symlink in structures/)
          filename      basename only
          plddt         mean pLDDT score
          round_idx     0 = unbiased, N = adaptive round N
          is_adaptive   bool
          nominal_cell  (ci, cj) string of the cell targeted at run time
                        (under that round's PCA axes); None for round-0
          PC1, PC2      coordinates under the *final* PCA
          cell_final    (ci, cj) string under the final PCA grid
          p0            p0(cell_final): fraction of round-0 structures in cell
          p_total       p_total(cell_final): fraction of all structures in cell
          weight        importance weight = p0 / p_total, normalised to mean 1
                        over non-zero-weight structures; 0 for unvisited cells
    """
    output_dir    = output_dir.rstrip('/') + '/'
    structures_dir = os.path.join(output_dir, 'structures')

    # ── pca_selection ─────────────────────────────────────────────────────────
    if pca_selection is None:
        sel_path = os.path.join(output_dir, 'pca_selection.txt')
        if not os.path.exists(sel_path):
            raise FileNotFoundError(
                f'pca_selection.txt not found in {output_dir}. '
                'Pass pca_selection= explicitly, or re-run the pipeline '
                '(recent versions save this file automatically).'
            )
        with open(sel_path) as f:
            pca_selection = f.read().strip()

    # ── Final PCA state ───────────────────────────────────────────────────────
    pca_state_path = os.path.join(output_dir, 'pca_state.npz')
    if not os.path.exists(pca_state_path):
        raise FileNotFoundError(f'pca_state.npz not found in {output_dir}')
    pca, p1_edges, p2_edges, ref_basename = _load_pca_npz(pca_state_path)

    if p1_edges is None or p2_edges is None:
        raise ValueError(
            'pca_state.npz does not contain grid edges. '
            'Ensure the pipeline completed at least one adaptive round.'
        )
    n_grid = len(p1_edges) - 1

    # ── plddt_dict ────────────────────────────────────────────────────────────
    plddt_path = os.path.join(output_dir, 'plddt_dict.json')
    if not os.path.exists(plddt_path):
        raise FileNotFoundError(f'plddt_dict.json not found in {output_dir}')
    with open(plddt_path) as f:
        stored = json.load(f)
    plddt_dict = {
        os.path.join(structures_dir, k): v
        for k, v in stored.items()
        if os.path.lexists(os.path.join(structures_dir, k))
    }
    if not plddt_dict:
        raise ValueError(f'No structures found under {structures_dir}')

    structure_paths = sorted(plddt_dict.keys())

    # ── Alignment reference ───────────────────────────────────────────────────
    ref_path = _resolve_ref(ref_basename, structures_dir)
    if ref_path is None:
        ref_path = structure_paths[0]

    # ── Project all structures onto final PCA ─────────────────────────────────
    print(f'  Projecting {len(structure_paths)} structures onto final PCA...',
          end=' ', flush=True)
    valid_paths, pc_coords = _project_all(structure_paths, ref_path, pca, pca_selection)
    print(f'done  ({len(valid_paths)} projected)')

    if not valid_paths:
        raise RuntimeError('No structures could be projected — check pca_selection.')

    # ── Build DataFrame with metadata ─────────────────────────────────────────
    records = []
    for path, pc in zip(valid_paths, pc_coords):
        filename = os.path.basename(path)
        round_idx, nominal_cell = _parse_round_info(filename)
        records.append({
            'path':         path,
            'filename':     filename,
            'plddt':        plddt_dict.get(path, float('nan')),
            'round_idx':    round_idx,
            'is_adaptive':  round_idx is not None and round_idx > 0,
            'nominal_cell': str(nominal_cell) if nominal_cell else None,
            'PC1':          float(pc[0]),
            'PC2':          float(pc[1]),
        })
        if pca.n_components_ >= 3:
            records[-1]['PC3'] = float(pc[2])

    df = pd.DataFrame(records)

    # ── Assign to final-grid cells ────────────────────────────────────────────
    ci = np.clip(np.digitize(df['PC1'].values, p1_edges) - 1, 0, n_grid - 1)
    cj = np.clip(np.digitize(df['PC2'].values, p2_edges) - 1, 0, n_grid - 1)
    df['cell_final'] = [f'({i},{j})' for i, j in zip(ci, cj)]

    # ── p0 and p_total ────────────────────────────────────────────────────────
    r0_mask = df['round_idx'] == 0
    n_r0 = int(r0_mask.sum())
    if n_r0 == 0:
        raise ValueError(
            'No round-0 (unbiased) structures found. '
            'Cannot estimate p0 — check filename prefixes.'
        )

    r0_counts   = df.loc[r0_mask, 'cell_final'].value_counts()
    all_counts  = df['cell_final'].value_counts()
    n_total     = len(df)

    p0_map      = (r0_counts / n_r0).to_dict()
    p_total_map = (all_counts / n_total).to_dict()

    df['p0']      = df['cell_final'].map(lambda c: max(p0_map.get(c, 0.0), min_p0))
    df['p_total'] = df['cell_final'].map(p_total_map)

    # ── Importance weights ────────────────────────────────────────────────────
    df['weight'] = df['p0'] / df['p_total'].clip(lower=1e-10)

    nonzero = df['weight'] > 0
    if nonzero.any():
        df.loc[nonzero, 'weight'] /= df.loc[nonzero, 'weight'].mean()

    n_zeroed = int((df['weight'] == 0).sum())
    n_cells  = df['cell_final'].nunique()
    print(f'  {n_cells} occupied cells  |  '
          f'{n_r0} round-0 structures  |  '
          f'{n_zeroed} structures zeroed (no round-0 support)')

    return df
