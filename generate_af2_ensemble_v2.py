#!/usr/bin/env python3
"""
Template-directed Adaptive AF2 Ensemble Generator  v2.

Two control levers
------------------
  1. MSA depth / temperature
       Low depth  → high stochasticity → broad conformational exploration.
       High depth → low stochasticity  → refinement near confident basins.
       The probe identifies optimal_depth = argmax U(d), where
           U(d) = yield(d) × intra_rmsd(d)
       rewards depth that produces diverse AND high-quality structures.

  2. Template targeting
       Each adaptive round identifies under-sampled cells in PC1-PC2 space
       and runs one ColabFold job per cell, passing the cell's highest-pLDDT
       member as --custom-template-path.  This provides a weak geometric prior
       toward the sparse region; dropout keeps predictions stochastic.

Probe (two-phase)
-----------------
  Phase 1 — coarse sweep at powers-of-2 depths, n_probe seeds × 1 model each.
  Phase 2 — fine sweep at 2^(1/fine_steps) steps around the utility peak.
  Output  — optimal_depth, viable_window (all depths with U ≥ viable_fraction
             × U_peak), probe_stats.

Round 0
-------
  Default: budget distributed evenly across viable_window (pure MSA, no templates).

  With --anneal-round0: schedule is viable_window depths followed by a geometric
  extension from viable_window[-1] up to the MSA ceiling.  Seeds the ensemble
  with both diverse low-depth structures and high-coevolution-constrained
  end-state conformations.  Extension step count is controlled by
  --round0-step-density; seeds per step is recalculated so the full
  --budget-initial is used.

Adaptive rounds (1+)
--------------------
  1.  PCA on SS-element Cα atoms (DSSP on the best-probe structure; falls back
      to all Cα if DSSP unavailable).  Re-run every --pca-interval rounds;
      between re-runs structures are projected onto cached eigenvectors.
      PCA model, grid edges, and round counters are written to pca_state.npz
      and reloaded automatically on restart.
  2.  Partition PC1-PC2 into an N×N grid (auto or --n-grid).  Identify sparse
      cells (count < --min-count-per-cell).  With --pca-interval > 1, grid
      edges are held fixed for the duration of each interval (--no-stable-grid
      disables this).
  3.  For each sparse cell: select the member with the highest pLDDT.
  4.  One ColabFold job per cell at the current depth, passing the selected
      structure as --custom-template-path.
  5.  n_cells_per_round = budget_per_round // (seeds_per_cell × n_models),
      capped at the number of sparse cells.  Most-sparse cells targeted first.
  6.  Dynamic temperature: MSA depth adjusts each round to maintain a target
      template hit-rate (--hit-rate-target).  Set to 0 to use fixed-rate
      annealing (--anneal-rate) instead.

  Stops when coverage ≥ --coverage-threshold or --max-rounds is reached.

Fill mode (optional, post-hoc)
-------------------------------
  --fill skips the probe/round-0/adaptive pipeline entirely and instead runs
  a one-off pass against an already-completed (or paused) run: it projects
  the existing accepted structures onto one cached PCA axis (--fill-axis),
  finds gaps along it, and for each gap builds a synthetic template by
  linearly blending the two bracketing real structures' full-atom
  coordinates with weights solved so the blend's own PCA projection lands
  exactly on the target point (exact because the cached PCA projection is
  linear in the input coordinates).  Every atom is blended by the same
  weights — not just the PCA-selection atoms — so there is no separate
  question of what to do with atoms outside the selection; they ride along.

  --fill-interpolate targets interior gaps (target strictly between the two
  donors).  --fill-extrapolate targets one step beyond each sampled extreme
  (donor weights outside [0, 1]) — e.g. to push further along a one-way
  motion (pore opening) than anything sampled.  At least one of the two
  must be given.  Both were spot-checked against ColabFold's actual
  mk_hhsearch_db/mk_template code path (not just bond-length heuristics):
  blended structures are accepted as normal templates, get a clean
  100%-identity hhsearch hit, and pass AlphaFold's template feature
  extraction without error.

  The synthetic blend is a weak geometric prior only — it is never added to
  the output ensemble.  Blended templates live under rounds/fill/*/template/
  and are never scanned by collect_structures(); only real ColabFold
  predictions that pass the normal quality filters are accepted, exactly as
  in adaptive rounds.  Overly aggressive interpolation/extrapolation should
  simply show up as a low per-target hit rate, the same signal used for
  dynamic-temperature adjustment in adaptive rounds.

Output structure
----------------
  <output_dir>/structures/    — symlinks to all accepted PDBs
  <output_dir>/rounds/        — raw ColabFold output per round/cell
  <output_dir>/rounds/fill/   — raw ColabFold output per --fill target
  <output_dir>/pca_state.npz  — PCA model + grid edges (restart state)
  <output_dir>/stats.json     — per-round statistics
  <output_dir>/plddt_dict.json

Restart behaviour
-----------------
  Runs are fully restartable.  Existing structures are reloaded from
  plddt_dict.json; PCA state and round counters from pca_state.npz.
  ColabFold directories that already contain a log file are skipped.
  The rounds_since_pca counter is preserved so --pca-interval scheduling
  continues correctly across restarts.

Handoff to the analysis pipeline
---------------------------------
    from ensemblr import Ensemble
    e = Ensemble('output_dir/structures/')
    # plddt_dict is returned by run_adaptive_ensemble_v2(), or reload:
    import json
    plddt_dict = json.load(open('output_dir/plddt_dict.json'))
"""

import os
import json
import math
import shutil
import random
import subprocess
import argparse

import numpy as np
from tqdm import tqdm
import MDAnalysis as mda
import MDAnalysis.analysis.align
from sklearn.decomposition import PCA

from ensemblr.smart_selector import generate_selection_token


# ── Environment ───────────────────────────────────────────────────────────────

_COLABFOLD_CONDA = (
    '/biggin/b212/bioc1781/software/colabfold_local/localcolabfold/'
    'colabfold-conda'
)
_COLABFOLD_BIN = os.path.join(_COLABFOLD_CONDA, 'bin')
_COLABFOLD_LIB = os.path.join(_COLABFOLD_CONDA, 'lib')


def _colabfold_env():
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    ld = env.get('LD_LIBRARY_PATH', '')
    if _COLABFOLD_LIB not in ld:
        env['LD_LIBRARY_PATH'] = _COLABFOLD_LIB + (':' + ld if ld else '')
    path = env.get('PATH', '')
    if _COLABFOLD_BIN not in path:
        env['PATH'] = _COLABFOLD_BIN + (':' + path if path else '')
    return env


# ── Naming helpers ────────────────────────────────────────────────────────────

def _msa_label(msa_depth):
    """'16:32' → '16-32' (safe for filenames and directory names)."""
    return msa_depth.replace(':', '-')


def _paired_int(depth_str):
    """'16:32' → 16."""
    return int(depth_str.split(':')[0])


# ── MSA pre-fetch ─────────────────────────────────────────────────────────────

def _count_msa_sequences(a3m_path):
    with open(a3m_path) as f:
        content = f.read()
    entries    = content.split('>')
    n_paired   = 0
    n_unpaired = 0
    for entry in entries[2:]:
        seq = ''.join(entry.split('\n')[1:])
        if '\x00' in seq:
            n_paired += 1
        else:
            n_unpaired += 1
    return n_paired, n_unpaired


def fetch_msa(fasta, msa_dir, dry_run=False):
    """
    Pre-fetch the MSA from the server using --msa-only.
    Returns (msa_dir, n_total, n_paired).
    """
    os.makedirs(msa_dir, exist_ok=True)
    a3m_files = [f for f in os.listdir(msa_dir) if f.endswith('.a3m')]

    if dry_run:
        print(f'  [dry-run] would fetch MSA → {msa_dir}')
        return msa_dir, None, None

    if a3m_files:
        print(f'  MSA already cached → {msa_dir}')
    else:
        print(f'  Fetching MSA from server → {msa_dir}')
        cmd      = ['colabfold_batch', fasta, msa_dir, '--msa-only']
        log_path = os.path.join(msa_dir, 'msa_fetch.log')
        with open(log_path, 'w') as log_f:
            subprocess.run(cmd, env=_colabfold_env(), check=True,
                           stdout=log_f, stderr=subprocess.STDOUT)
        a3m_files = [f for f in os.listdir(msa_dir) if f.endswith('.a3m')]

    if not a3m_files:
        print('  Warning: no .a3m file found after MSA fetch.')
        return msa_dir, None, None

    n_paired, n_unpaired = _count_msa_sequences(os.path.join(msa_dir, a3m_files[0]))
    n_total = n_paired + n_unpaired
    print(f'  {"─"*40}')
    print(f'  MSA summary')
    print(f'  {"─"*40}')
    print(f'  Total sequences  : {n_total}')
    print(f'  Paired sequences : {n_paired}'
          + ('  (multimer)' if n_paired > 0 else '  (monomer)'))
    print(f'  Unpaired         : {n_unpaired}')
    print(f'  Max useful depth : {n_total}:{n_total*2}  '
          f'(--max-msa above this has no effect)')
    print(f'  {"─"*40}')
    return msa_dir, n_total, n_paired


# ── Depth generation ──────────────────────────────────────────────────────────

def _auto_depths(msa_start, n_total):
    """
    Doubling MSA depths from 2^msa_start up to n_total, plus natural anchor.
    """
    max_exp = math.ceil(math.log2(max(n_total, 2)))
    depths  = [f'{2**i}:{2**(i+1)}' for i in range(msa_start, max_exp + 1)
               if 2**i <= n_total]
    anchor  = f'{n_total}:{n_total * 2}'
    if anchor not in depths:
        depths.append(anchor)
    return depths


def _interleave_depths(d_lo_paired, d_hi_paired, n_steps):
    """
    Generate n_steps-1 intermediate 'paired:unpaired' depth strings between
    d_lo_paired and d_hi_paired using geometric spacing (2^(1/n_steps) steps).
    Returns an empty list when d_lo_paired >= d_hi_paired or n_steps < 2.
    """
    if d_lo_paired >= d_hi_paired or n_steps < 2:
        return []
    ratio  = (d_hi_paired / d_lo_paired) ** (1.0 / n_steps)
    result = []
    d = d_lo_paired * ratio
    for _ in range(n_steps - 1):
        p = max(1, round(d))
        s = f'{p}:{p * 2}'
        if s not in result:
            result.append(s)
        d *= ratio
    return result


# ── Round-0 annealing schedule ────────────────────────────────────────────────

def _build_round0_annealing_schedule(viable_window, ceiling_paired,
                                      seeds_per_step, n_models, budget_initial):
    """
    Option-C round-0 schedule: viable_window + geometric extension to ceiling.

    total_steps  = max(n_surv, budget // (seeds_per_step × n_models))
    n_ext        = total_steps - n_surv  (steps added above viable window)
    seeds / step = budget // (actual_steps × n_models)  [recalculated to use full budget]

    Returns (schedule, n_seeds_per_step, n_ext_actual)
    """
    n_surv            = len(viable_window)
    viable_max_paired = _paired_int(viable_window[-1])

    total_steps = max(n_surv, budget_initial // max(1, seeds_per_step * n_models))
    n_ext       = total_steps - n_surv

    extension = []
    if n_ext > 0 and viable_max_paired < ceiling_paired:
        ratio  = (ceiling_paired / viable_max_paired) ** (1.0 / n_ext)
        seen   = set(viable_window)
        raw_ds = [round(viable_max_paired * ratio ** i) for i in range(1, n_ext + 1)]
        raw_ds[-1] = ceiling_paired  # guarantee ceiling as final point
        for d in raw_ds:
            d = max(viable_max_paired + 1, min(d, ceiling_paired))
            s = f'{d}:{d * 2}'
            if s not in seen:
                extension.append(s)
                seen.add(s)

    schedule       = list(viable_window) + extension
    n_steps_actual = len(schedule)
    n_seeds        = max(1, budget_initial // (n_steps_actual * n_models))
    return schedule, n_seeds, len(extension)


# ── ColabFold wrapper ─────────────────────────────────────────────────────────

def run_colabfold(msa_dir, output_dir, msa_depth, n_seeds, n_models,
                  num_recycle=0, seed=None, extra_flags='',
                  template_dir=None, use_pdb_templates=False, dry_run=False):
    """
    Call colabfold_batch using a pre-fetched MSA directory as input.
    By default (use_pdb_templates=False) no --templates flag is passed and
    ColabFold does not query the PDB — this is already the binary's default.
    If template_dir is provided, --custom-template-path is added instead.
    If use_pdb_templates=True and no template_dir, --templates is added.
    Skips silently if the output directory already contains a log file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not dry_run:
        done = [f for f in os.listdir(output_dir) if f.endswith('.done.txt')]
        if done:
            tqdm.write(f'    skipping {os.path.basename(output_dir)} (already complete)')
            return output_dir

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    cmd = [
        'colabfold_batch', msa_dir, output_dir,
        '--random-seed', str(seed),
        '--num-seeds',   str(n_seeds),
        '--num-models',  str(n_models),
        '--num-recycle', str(num_recycle),
        '--use-dropout',
        '--rank', 'plddt',
        '--max-msa', msa_depth,
    ]
    if template_dir is not None:
        cmd += ['--templates', '--custom-template-path', template_dir]
    elif use_pdb_templates:
        cmd += ['--templates']
    if extra_flags:
        cmd += extra_flags.split()

    if dry_run:
        tqdm.write(f'    [dry-run] {" ".join(cmd)}')
        return output_dir

    log_path = os.path.join(output_dir, 'colabfold.log')
    with open(log_path, 'w') as log_f:
        subprocess.run(cmd, env=_colabfold_env(), check=True,
                       stdout=log_f, stderr=subprocess.STDOUT)
    return output_dir


# ── Score-metric helpers ──────────────────────────────────────────────────────

SCORE_METRICS = ('plddt', 'ptm', 'iptm', 'iptm_composite')

_METRIC_LABELS = {
    'plddt':          'pLDDT',
    'ptm':            'pTM×100',
    'iptm':           'ipTM×100',
    'iptm_composite': 'ipTM_comp×100',
}

_METRIC_DEFAULT_THRESH = {
    'plddt':          80.0,
    'ptm':            50.0,
    'iptm':           50.0,
    'iptm_composite': 50.0,
}

_METRIC_DEFAULT_MIN_MEAN = {
    'plddt':          50.0,
    'ptm':            30.0,
    'iptm':           30.0,
    'iptm_composite': 30.0,
}


def _selection_mean_plddt(pdb_path, selection):
    """Mean per-residue pLDDT over `selection` atoms in one PDB.

    ColabFold writes per-residue pLDDT into the B-factor column of every
    output PDB (relaxed and unrelaxed alike), so this is read straight from
    the atoms rather than the whole-chain scalar in log.txt.  Returns None if
    the selection matches no atoms or the file can't be parsed.
    """
    try:
        u  = mda.Universe(pdb_path)
        ag = u.select_atoms(selection)
    except Exception:
        return None
    if len(ag) == 0:
        return None
    return float(ag.tempfactors.mean())


def _quality_from_metrics(metrics, score_metric):
    """Normalise a ColabFold metrics dict to a 0–100 quality score.

    pLDDT is used as-is; pTM/ipTM (native 0–1) are multiplied by 100 so all
    metrics share the same scale and ``--quality-thresh`` means the same thing
    regardless of which metric is selected.
    """
    if score_metric == 'plddt':
        return metrics.get('plddt')
    if score_metric == 'ptm':
        v = metrics.get('ptm')
        return v * 100.0 if v is not None else None
    if score_metric == 'iptm':
        v = metrics.get('iptm')
        return v * 100.0 if v is not None else None
    if score_metric == 'iptm_composite':
        ptm  = metrics.get('ptm')
        iptm = metrics.get('iptm')
        if ptm is not None and iptm is not None:
            return (0.8 * iptm + 0.2 * ptm) * 100.0
    return None


# ── ColabFold log parsing ─────────────────────────────────────────────────────

def _find_log(colabfold_dir):
    logs = [f for f in os.listdir(colabfold_dir) if f.endswith('log.txt')]
    if not logs:
        raise FileNotFoundError(f'No log.txt found in {colabfold_dir}')
    return os.path.join(colabfold_dir, logs[0])


def parse_log(log_path):
    """Return {rank_str: {'plddt': float, 'ptm': float|None, 'iptm': float|None}}."""
    result = {}
    with open(log_path) as f:
        for line in f:
            if 'rank' not in line:
                continue
            parts = line.split()
            for token in parts:
                if token.startswith('rank_'):
                    try:
                        rank    = '_'.join(token.split('_')[:2])
                        metrics = {}
                        for part in parts:
                            if part.startswith('pLDDT='):
                                metrics['plddt'] = float(part[6:])
                            elif part.startswith('pTM='):
                                metrics['ptm']   = float(part[4:])
                            elif part.startswith('ipTM='):
                                metrics['iptm']  = float(part[5:])
                        if 'plddt' in metrics:
                            result[rank] = metrics
                    except (IndexError, ValueError):
                        pass
                    break
    return result


# ── Structure collection (symlinks) ───────────────────────────────────────────

def collect_structures(colabfold_dir, structures_dir, msa_depth,
                       round_label, quality_thresh, relaxed=False, dry_run=False,
                       score_metric='plddt', pca_selection=None):
    """
    Symlink accepted PDB files from colabfold_dir into structures_dir.

    If `pca_selection` is given and score_metric == 'plddt', quality is the
    mean per-residue pLDDT over that selection (read from PDB B-factors)
    rather than ColabFold's whole-chain mean — this lets accept/reject and
    representative selection judge only the region of interest (e.g. the
    ordered core), ignoring disordered tails/loops outside the selection.

    Returns
    -------
    accepted : dict {symlink_path: quality}   quality is normalised to 0–100
    n_total  : int   total PDB files seen (accepted + rejected)
    """
    if dry_run:
        return {}, 0

    tag   = '_relaxed_' if relaxed else '_unrelaxed_'
    label = _msa_label(msa_depth)
    os.makedirs(structures_dir, exist_ok=True)

    rank_metrics = parse_log(_find_log(colabfold_dir))

    accepted, n_total = {}, 0
    for fname in sorted(os.listdir(colabfold_dir)):
        if not (fname.endswith('.pdb') and tag in fname):
            continue
        parts = fname.split('_')
        if 'rank' not in parts:
            continue
        n_total += 1
        rank    = 'rank_' + parts[parts.index('rank') + 1].split('.')[0]
        metrics = rank_metrics.get(rank)
        quality = _quality_from_metrics(metrics, score_metric) if metrics else None
        if quality is None:
            continue

        src = os.path.abspath(os.path.join(colabfold_dir, fname))
        if pca_selection is not None and score_metric == 'plddt':
            sel_quality = _selection_mean_plddt(src, pca_selection)
            if sel_quality is not None:
                quality = sel_quality

        if quality <= quality_thresh:
            continue

        dest = os.path.join(structures_dir,
                            f'{round_label}_{label}_{rank}.pdb')
        if not os.path.lexists(dest):
            os.symlink(os.path.relpath(src, os.path.dirname(dest)), dest)
        accepted[dest] = quality

    return accepted, n_total


# ── Probe helpers ─────────────────────────────────────────────────────────────

def _utility(yield_val, intra_rmsd):
    """U(d) = yield × intra_rmsd  — rewards quality AND structural diversity."""
    return float(yield_val) * float(intra_rmsd or 0.0)


def _load_probe_coords(out_dir, relaxed=False):
    tag  = '_relaxed_' if relaxed else '_unrelaxed_'
    pdbs = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.endswith('.pdb') and tag in f
    )
    coords = []
    for path in pdbs:
        u  = mda.Universe(path)
        ca = u.select_atoms('name CA')
        if len(ca) > 0:
            coords.append(ca.positions.copy())
    return coords


def _mean_pairwise_rmsd(coords):
    from MDAnalysis.analysis.rms import rmsd as mda_rmsd
    if len(coords) < 2:
        return 0.0
    rmsds = [
        mda_rmsd(coords[i], coords[j], superposition=True)
        for i in range(len(coords))
        for j in range(i + 1, len(coords))
        if len(coords[i]) == len(coords[j])
    ]
    return float(np.mean(rmsds)) if rmsds else 0.0


def _score_probe_dir(out_dir, quality_thresh, min_mean_quality, relaxed,
                     score_metric='plddt', pca_selection=None):
    """
    Compute {mean_quality, yield, intra_rmsd, utility} for one already-
    populated probe directory (no ColabFold invocation — just scoring).

    If `pca_selection` is given and score_metric == 'plddt', per-structure
    quality is the mean pLDDT over that selection (read from PDB B-factors)
    rather than ColabFold's whole-chain mean from the log.

    Returns None if no scoreable predictions are found in out_dir.
    """
    if pca_selection is not None and score_metric == 'plddt':
        tag  = '_relaxed_' if relaxed else '_unrelaxed_'
        pdbs = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                if f.endswith('.pdb') and tag in f]
        all_q = [_selection_mean_plddt(p, pca_selection) for p in pdbs]
        all_q = [q for q in all_q if q is not None]
    else:
        rank_metrics = parse_log(_find_log(out_dir))
        all_q = [_quality_from_metrics(m, score_metric)
                 for m in rank_metrics.values()]
        all_q = [q for q in all_q if q is not None]

    if not all_q:
        return None

    mean_q = float(np.mean(all_q))
    yld    = sum(1 for q in all_q if q > quality_thresh) / len(all_q)
    if mean_q >= min_mean_quality:
        coords = _load_probe_coords(out_dir, relaxed)
        intra  = round(_mean_pairwise_rmsd(coords), 3)
    else:
        intra = 0.0

    return {
        'mean_quality': round(mean_q, 2),
        'yield':        round(yld,    3),
        'intra_rmsd':   intra,
        'utility':      round(_utility(yld, intra), 4),
    }


def _run_probe_batch(msa_dir, probe_dir, depths, n_probe, quality_thresh,
                     min_mean_quality, num_recycle, extra_flags,
                     dry_run, relaxed, no_pdb_templates=True, pbar=None,
                     score_metric='plddt', pca_selection=None):
    """
    Run n_probe predictions at each depth, compute per-depth stats.
    Returns list of dicts: {depth, mean_quality, yield, intra_rmsd, utility}.
    """
    stats = []

    for i_d, depth in enumerate(depths):
        label   = _msa_label(depth)
        out_dir = os.path.join(probe_dir, f'msa_{label}')
        run_colabfold(msa_dir, out_dir, depth,
                      n_seeds=n_probe, n_models=1,
                      num_recycle=num_recycle, extra_flags=extra_flags,
                      use_pdb_templates=not no_pdb_templates, dry_run=dry_run)
        if pbar:
            pbar.update(n_probe)

        if dry_run:
            # synthetic values: diversity peaks mid-range, quality is stable
            mid    = len(depths) // 2
            scored = {
                'mean_quality': 85.0,
                'yield':        0.9,
                'intra_rmsd':   round(max(0.5, 4.5 - abs(i_d - mid) * 0.6), 2),
            }
            scored['utility'] = round(_utility(scored['yield'], scored['intra_rmsd']), 4)
        else:
            scored = _score_probe_dir(out_dir, quality_thresh, min_mean_quality,
                                      relaxed, score_metric, pca_selection)
            if scored is None:
                tqdm.write(f'    {depth:>14s}  no predictions — skipping')
                continue

        stats.append({'depth': depth, **scored})

    return stats


def _rescore_probe_stats(stats, probe_dir, quality_thresh, min_mean_quality,
                         relaxed, score_metric, pca_selection):
    """
    Re-derive stats for already-probed depths using `pca_selection` instead
    of whatever scoring was used the first time round (retroactive rescore —
    no new ColabFold predictions, just re-reading the PDBs already on disk).
    Falls back to the original entry if a depth's directory can't be rescored.
    """
    rescored = []
    for s in stats:
        out_dir = os.path.join(probe_dir, f'msa_{_msa_label(s["depth"])}')
        scored  = _score_probe_dir(out_dir, quality_thresh, min_mean_quality,
                                   relaxed, score_metric, pca_selection)
        rescored.append({'depth': s['depth'], **scored} if scored is not None else s)
    return rescored


# ── Two-phase probe ───────────────────────────────────────────────────────────

def probe_and_optimize(
    msa_dir, rounds_dir, coarse_depths, n_probe,
    quality_thresh, min_mean_quality,
    fine_steps=4, viable_fraction=0.5,
    num_recycle=0, extra_flags='',
    no_pdb_templates=True, dry_run=False, relaxed=False,
    score_metric='plddt', selection_quality=False, quality_thresh_pct=None,
    pca_exclude=None, pca_include=None,
):
    """
    Phase 1: coarse sweep at coarse_depths (powers-of-2).
    Phase 2: fine sweep at 2^(1/fine_steps) spacing around the utility peak.

    The anchor depth (coarse_depths[-1] = natural MSA ceiling) is included in
    the sweep for landscape context but excluded from the viable window.

    If `selection_quality` (score_metric == 'plddt' only), the PCA selection
    is built from the coarse-sweep structures right after phase 1 — the
    earliest point real structures exist to run DSSP on — the coarse stats
    are retroactively rescored with it, and the fine sweep (phase 2) scores
    directly against it.  Without this, quality throughout probing stays
    whole-chain pLDDT and the selection is left for the caller to build
    afterwards (as before).

    If `quality_thresh_pct` is given, `quality_thresh` is replaced with
    `quality_thresh_pct × (best coarse depth's mean_quality)` right after the
    coarse sweep (and any selection rescore above), making acceptance
    protein-relative instead of an absolute cutoff.  The coarse stats are
    rescored again with the resolved threshold, and it is used for the fine
    sweep and returned for round 0 / adaptive rounds to reuse.

    Returns
    -------
    optimal_depth  : str    depth with highest U = yield × intra_rmsd
    viable_window  : list   depths with U ≥ viable_fraction × U_peak,
                            anchor excluded, sorted shallow → deep
    all_stats      : list   one dict per probed depth (coarse + fine)
    pca_selection  : str|None  selection built from coarse structures if
                               selection_quality was applied, else None
    quality_thresh : float  resolved threshold (dynamic if quality_thresh_pct
                            was given, otherwise the input value unchanged)

    `pca_exclude` / `pca_include` (MDA selection tokens) bound the PCA
    selection built here, e.g. to drop a soluble domain the transporter's
    loop-of-interest analysis shouldn't be judged against.
    """
    probe_dir    = os.path.join(rounds_dir, 'probe')
    anchor_depth = coarse_depths[-1]
    n_coarse     = len(coarse_depths)

    # ── Phase 1: coarse sweep ─────────────────────────────────────────────────
    print(f'\n{"─"*70}')
    print(f'  Probe  phase 1 / 2  —  coarse sweep'
          f'  ({n_probe} seeds × 1 model, {n_coarse} depths)')
    print(f'{"─"*70}')

    with tqdm(total=n_probe * n_coarse, desc='  coarse probe',
              unit='pred', leave=False) as pbar:
        coarse_stats = _run_probe_batch(
            msa_dir, probe_dir, coarse_depths, n_probe, quality_thresh,
            min_mean_quality, num_recycle, extra_flags, dry_run, relaxed,
            no_pdb_templates=no_pdb_templates, pbar=pbar,
            score_metric=score_metric)

    # Build the PCA selection from the coarse-sweep structures as early as
    # possible, then rescore the coarse stats against it, so the fine sweep
    # (and everything downstream) can judge quality on the region of
    # interest rather than the whole chain.
    pca_selection = None
    if selection_quality and score_metric == 'plddt' and not dry_run:
        pca_selection = _build_pca_selection(probe_dir, relaxed, dry_run=dry_run,
                                             score_metric=score_metric,
                                             pca_exclude=pca_exclude,
                                             pca_include=pca_include)
        coarse_stats = _rescore_probe_stats(
            coarse_stats, probe_dir, quality_thresh, min_mean_quality,
            relaxed, score_metric, pca_selection)

    # Resolve a protein-relative threshold from the best coarse depth's mean
    # quality (using whichever metric was just settled above), then rescore
    # the coarse stats' yield/utility against it.
    if quality_thresh_pct is not None and coarse_stats and not dry_run:
        best_coarse_quality = max(s['mean_quality'] for s in coarse_stats)
        quality_thresh = round(quality_thresh_pct * best_coarse_quality, 2)
        print(f'  Dynamic quality threshold: {quality_thresh_pct:.0%} × '
              f'{best_coarse_quality:.1f} (best coarse depth) = {quality_thresh:.1f}')
        coarse_stats = _rescore_probe_stats(
            coarse_stats, probe_dir, quality_thresh, min_mean_quality,
            relaxed, score_metric, pca_selection)

    already_probed = {s['depth'] for s in coarse_stats}

    # Find coarse utility peak among non-anchor, non-noisy depths
    valid_coarse = [s for s in coarse_stats
                    if s['mean_quality'] >= min_mean_quality
                    and s['depth'] != anchor_depth
                    and s['utility'] > 0]

    if not valid_coarse:
        print('  Warning: no valid coarse depths — skipping fine probe.')
        return None, [], coarse_stats, pca_selection, quality_thresh

    peak_coarse  = max(valid_coarse, key=lambda s: s['utility'])
    peak_idx     = coarse_depths.index(peak_coarse['depth'])

    # ── Phase 2: fine sweep around coarse peak ────────────────────────────────
    # Range: one step below and above peak, but never into the anchor
    lo_idx = max(0, peak_idx - 1)
    hi_idx = min(len(coarse_depths) - 2, peak_idx + 1)  # -2 keeps anchor out

    fine_depths = [d for d in _interleave_depths(
                       _paired_int(coarse_depths[lo_idx]),
                       _paired_int(coarse_depths[hi_idx]),
                       fine_steps)
                   if d not in already_probed]

    fine_stats = []
    if fine_depths:
        print(f'\n  Probe  phase 2 / 2  —  fine sweep around {peak_coarse["depth"]}'
              f'  ({len(fine_depths)} new depths,  step ≈ {2**(1/fine_steps):.2f}×)')
        with tqdm(total=n_probe * len(fine_depths), desc='  fine probe',
                  unit='pred', leave=False) as pbar:
            fine_stats = _run_probe_batch(
                msa_dir, probe_dir, fine_depths, n_probe, quality_thresh,
                min_mean_quality, num_recycle, extra_flags, dry_run, relaxed,
                no_pdb_templates=no_pdb_templates, pbar=pbar,
                score_metric=score_metric, pca_selection=pca_selection)
    else:
        print(f'\n  Probe  phase 2 / 2  —  coarse peak at boundary; no fine depths needed.')

    # ── Combine and sort by paired depth ──────────────────────────────────────
    all_stats = sorted(coarse_stats + fine_stats, key=lambda s: _paired_int(s['depth']))

    # Overall optimal (exclude anchor)
    valid_all = [s for s in all_stats
                 if s['mean_quality'] >= min_mean_quality
                 and s['depth'] != anchor_depth
                 and s['utility'] > 0]

    if not valid_all:
        _print_probe_table(all_stats, None, [], anchor_depth, viable_fraction, score_metric)
        return None, [], all_stats, pca_selection, quality_thresh

    optimal = max(valid_all, key=lambda s: s['utility'])
    U_max   = optimal['utility']

    viable_window = sorted(
        [s['depth'] for s in valid_all if s['utility'] >= viable_fraction * U_max],
        key=_paired_int,
    )

    _print_probe_table(all_stats, optimal['depth'], viable_window,
                       anchor_depth, viable_fraction, score_metric)

    print(f'\n  Optimal depth  : {optimal["depth"]}  '
          f'(U = {U_max:.3f}  =  yield {optimal["yield"]:.0%}  ×  '
          f'{optimal["intra_rmsd"]:.2f} Å intra-RMSD)')
    print(f'  Viable window  : {viable_window[0]} → {viable_window[-1]}  '
          f'({len(viable_window)} depths,  U ≥ {viable_fraction:.0%} × U_peak)')

    return optimal['depth'], viable_window, all_stats, pca_selection, quality_thresh


def _print_probe_table(all_stats, optimal_depth, viable_window,
                       anchor_depth, viable_fraction, score_metric='plddt'):
    W         = 72
    qlbl      = _METRIC_LABELS.get(score_metric, score_metric)
    print(f'\n  {"─"*W}')
    print(f'  {"depth":>14s}  {qlbl:>8s}  {"yield":>5s}  '
          f'{"intra(Å)":>8s}  {"utility":>7s}  status')
    print(f'  {"─"*W}')
    for s in all_stats:
        d      = s['depth']
        is_opt = (d == optimal_depth)
        in_win = d in viable_window
        is_anc = (d == anchor_depth)
        marker  = ('★' if is_opt else ('▶' if in_win else ' '))
        d_label = (d + ' *') if is_anc else d
        intra_s = (f'{s["intra_rmsd"]:8.2f}'
                   if s['intra_rmsd'] is not None else '       —')
        util_s  = (f'{s["utility"]:7.3f}'
                   if s['utility'] is not None else '      —')
        status  = ('optimal' if is_opt else
                   ('viable'  if in_win else
                    ('anchor'  if is_anc else
                     ('noisy'   if s['mean_quality'] < 50 else 'below U threshold'))))
        print(f'  {marker} {d_label:>13s}  {s["mean_quality"]:>8.1f}  '
              f'{s["yield"]:>4.0%}  {intra_s}  {util_s}  {status}')
    print(f'  {"─"*W}')
    print(f'  U(d) = yield × intra_rmsd  ★ = optimal  '
          f'▶ = viable (U ≥ {viable_fraction:.0%} × U_peak)  * = anchor')


# ── PCA helpers ───────────────────────────────────────────────────────────────

def _find_best_probe_structure(probe_dir, relaxed=False, score_metric='plddt'):
    """Return (path, quality) of the highest-quality PDB across all probe dirs."""
    tag = '_relaxed_' if relaxed else '_unrelaxed_'
    best_path, best_quality = None, -1.0

    for depth_dir in sorted(os.listdir(probe_dir)):
        depth_path = os.path.join(probe_dir, depth_dir)
        if not os.path.isdir(depth_path):
            continue
        try:
            rank_metrics = parse_log(_find_log(depth_path))
        except FileNotFoundError:
            continue
        for fname in os.listdir(depth_path):
            if not (fname.endswith('.pdb') and tag in fname):
                continue
            parts = fname.split('_')
            if 'rank' not in parts:
                continue
            rank    = 'rank_' + parts[parts.index('rank') + 1].split('.')[0]
            metrics = rank_metrics.get(rank) or {}
            q       = _quality_from_metrics(metrics, score_metric) if metrics else None
            if q is not None and q > best_quality:
                best_quality = q
                best_path    = os.path.join(depth_path, fname)

    return best_path, best_quality


def _build_pca_selection(probe_dir, relaxed=False, dry_run=False, score_metric='plddt',
                         pca_exclude=None, pca_include=None):
    """
    Run DSSP on the highest-quality probe structure to get a SS-element CA
    selection.  Falls back to 'name CA' if DSSP fails.

    `pca_exclude` / `pca_include` are MDAnalysis selection tokens (e.g.
    'resid 1:120') forwarded to generate_selection_token as excluded_residues
    / explicitly_include, letting the PCA (and --selection-quality gating)
    be bounded to a region of interest — e.g. dropping a soluble domain or
    always including a functional loop regardless of its DSSP assignment.
    """
    if dry_run:
        print('  PCA selection: all CA atoms  [dry-run, skipping DSSP]')
        return 'name CA'

    best_path, best_quality = _find_best_probe_structure(probe_dir, relaxed, score_metric)
    if best_path is None:
        print('  PCA selection: all CA atoms (no probe structures found for DSSP)')
        return 'name CA'

    qlbl = _METRIC_LABELS.get(score_metric, score_metric)
    try:
        sel = generate_selection_token(best_path, excluded_residues=pca_exclude,
                                       explicitly_include=pca_include)
        bounds = []
        if pca_exclude:
            bounds.append(f'exclude=({pca_exclude})')
        if pca_include:
            bounds.append(f'include=({pca_include})')
        bounds_str = f'  [{", ".join(bounds)}]' if bounds else ''
        print(f'  PCA selection: SS-element CA atoms  '
              f'(DSSP on {os.path.basename(best_path)}, {qlbl}={best_quality:.1f})'
              f'{bounds_str}')
        return sel
    except Exception as e:
        print(f'  PCA selection: all CA atoms (DSSP failed: {e})')
        return 'name CA'


def lightweight_pca(structure_paths, selection='name CA', n_components=3):
    """
    Align structures to the first one on `selection`, then PCA on those coords.
    Structures whose atom count differs from the reference are skipped.

    Returns
    -------
    pc_coords   : np.ndarray  shape (N_valid, n_components)
    valid_paths : list of str
    eig_ratios  : list of float  explained variance ratio per component
    """
    ref   = mda.Universe(structure_paths[0])
    n_sel = len(ref.select_atoms(selection))

    coords      = []
    valid_paths = []
    for path in structure_paths:
        u  = mda.Universe(path)
        ag = u.select_atoms(selection)
        if len(ag) != n_sel:
            continue
        if path != structure_paths[0]:
            mda.analysis.align.alignto(u, ref, select=selection,
                                       match_atoms=False)
        coords.append(u.select_atoms(selection).positions.flatten())
        valid_paths.append(path)

    coords = np.array(coords)
    pca    = PCA(n_components=min(n_components, coords.shape[0],
                                  coords.shape[1]))
    pc_coords = pca.fit_transform(coords)
    return pc_coords, valid_paths, pca.explained_variance_ratio_.tolist(), pca.components_, pca


# ── PCA state persistence ─────────────────────────────────────────────────────

def _save_pca_state(path, pca_model, rounds_since_pca, total_rounds,
                    p1_edges=None, p2_edges=None, ref_basename=None):
    """Persist PCA model + counters + grid edges so restarts pick up correctly."""
    arrays = dict(
        components=pca_model.components_,
        mean=pca_model.mean_,
        explained_variance_ratio=pca_model.explained_variance_ratio_,
        n_features_in=np.array(pca_model.n_features_in_),
        rounds_since_pca=np.array(rounds_since_pca),
        total_rounds=np.array(total_rounds),
    )
    if p1_edges is not None:
        arrays['p1_edges'] = p1_edges
        arrays['p2_edges'] = p2_edges
    if ref_basename is not None:
        arrays['ref_basename'] = np.array(ref_basename)
    np.savez(path, **arrays)


def _load_pca_state(path):
    data = np.load(path)
    n_comp = len(data['explained_variance_ratio'])
    pca = PCA(n_components=n_comp)
    pca.components_ = data['components']
    pca.mean_ = data['mean']
    pca.explained_variance_ratio_ = data['explained_variance_ratio']
    pca.n_features_in_ = int(data['n_features_in'])
    pca.n_components_ = n_comp
    rounds_since_pca = int(data['rounds_since_pca']) if 'rounds_since_pca' in data else 0
    total_rounds     = int(data['total_rounds'])     if 'total_rounds'     in data else 0
    p1_edges         = data['p1_edges']              if 'p1_edges'         in data else None
    p2_edges         = data['p2_edges']              if 'p2_edges'         in data else None
    return (pca, data['components'], data['explained_variance_ratio'].tolist(),
            rounds_since_pca, total_rounds, p1_edges, p2_edges)


def _pca_apply(pca_model, flat):
    """Apply a fitted PCA to a (1, n_features) array without sklearn validation."""
    return (flat - pca_model.mean_) @ pca_model.components_.T


def _project_onto_pca(structure_paths, ref_universe, n_ref_sel,
                      pca_selection, pca_model):
    """Project structures onto existing (fitted) PCA axes without refitting."""
    projected   = []
    valid_paths = []
    for path in structure_paths:
        try:
            u  = mda.Universe(os.path.realpath(path))
            ag = u.select_atoms(pca_selection)
            if len(ag) != n_ref_sel:
                continue
            mda.analysis.align.alignto(u, ref_universe,
                                       select=pca_selection, match_atoms=False)
            flat = u.select_atoms(pca_selection).positions.flatten()
            projected.append(_pca_apply(pca_model, flat))
            valid_paths.append(path)
        except Exception:
            pass
    if not projected:
        return [], np.empty((0, pca_model.n_components_))
    return valid_paths, np.array(projected)


# ── Fill mode: PC-axis gap templates ──────────────────────────────────────────

def _build_blend_template(lo_path, hi_path, target_val, lo_val, hi_val,
                          ref_universe, pca_selection, out_pdb_path):
    """
    Full-atom linear blend of two real structures such that the blend's
    projection onto the fixed PCA axis lands exactly at target_val.

    Every candidate is an AF2 prediction of the identical sequence (same
    atom composition/ordering), so every atom -- not just the PCA-selection
    atoms -- can be blended by the same weights.  Because the cached PCA
    projection is linear in the coordinates, y = (flat - mean) @ components.T,
    a linear combination of two structures' coordinates projects to exactly
    the same combination of their individual projections.  w outside [0, 1]
    extrapolates the pair's displacement past hi_val (or below lo_val).
    """
    w_hi = (target_val - lo_val) / (hi_val - lo_val)
    w_lo = 1.0 - w_hi

    u_lo = mda.Universe(os.path.realpath(lo_path))
    u_hi = mda.Universe(os.path.realpath(hi_path))
    mda.analysis.align.alignto(u_lo, ref_universe, select=pca_selection, match_atoms=False)
    mda.analysis.align.alignto(u_hi, ref_universe, select=pca_selection, match_atoms=False)

    blended = mda.Merge(u_lo.atoms)
    blended.atoms.positions = (w_lo * u_lo.atoms.positions
                               + w_hi * u_hi.atoms.positions)
    blended.atoms.write(out_pdb_path)
    return w_lo, w_hi


def _count_axis_hits(accepted_paths, ref_universe, n_ref_sel, pca_model,
                     pca_selection, axis_idx, target_val, tol):
    """
    Project each accepted PDB back onto the fill axis and count how many
    land within `tol` of `target_val`.  Same role as _count_cell_hits() for
    adaptive rounds' dynamic temperature -- a low rate here signals the
    template (interpolated or extrapolated) was too weak/aggressive a prior.
    """
    hits, projected = 0, 0
    for path in accepted_paths:
        try:
            u  = mda.Universe(os.path.realpath(path))
            ag = u.select_atoms(pca_selection)
            if len(ag) != n_ref_sel:
                continue
            mda.analysis.align.alignto(u, ref_universe, select=pca_selection, match_atoms=False)
            flat = u.select_atoms(pca_selection).positions.flatten()
            pc   = _pca_apply(pca_model, flat)
            if abs(pc[axis_idx] - target_val) <= tol:
                hits += 1
            projected += 1
        except Exception:
            pass
    return hits, projected


# ── Sparse-cell detection with representative selection ───────────────────────

def find_sparse_cells(pc_coords, valid_paths, plddt_dict,
                      n_grid=None, min_count=3,
                      p1_edges=None, p2_edges=None):
    """
    Partition PC1-PC2 into an n × n grid.  For each sparse cell (member count
    < min_count), select the representative: the member with the highest pLDDT.

    If p1_edges / p2_edges are supplied the grid boundaries are fixed (stable-
    grid mode); otherwise they are computed from the current data range.

    Returns
    -------
    sparse      : list[dict]  sorted by deficit (most sparse first)
                  keys: cell (ci,cj), centre [pc1,pc2], representative str,
                        count int, deficit int
    coverage    : float  fraction of occupied cells that are well-sampled
    n_grid      : int    grid resolution used
    n_occupied  : int    number of occupied cells
    """
    n = len(pc_coords)
    pc2d = pc_coords[:, :2]
    p1, p2 = pc2d[:, 0], pc2d[:, 1]

    if p1_edges is not None and p2_edges is not None:
        n_grid = len(p1_edges) - 1
    else:
        if n_grid is None:
            n_grid = max(5, min(30, int(np.sqrt(n / min_count))))
        p1_edges = np.linspace(p1.min(), p1.max(), n_grid + 1)
        p2_edges = np.linspace(p2.min(), p2.max(), n_grid + 1)

    p1_idx = np.clip(np.digitize(p1, p1_edges) - 1, 0, n_grid - 1)
    p2_idx = np.clip(np.digitize(p2, p2_edges) - 1, 0, n_grid - 1)

    cell_members = {}
    for k, (ci, cj) in enumerate(zip(p1_idx, p2_idx)):
        cell_members.setdefault((ci, cj), []).append(k)

    n_occupied     = len(cell_members)
    n_well_sampled = sum(1 for m in cell_members.values() if len(m) >= min_count)
    coverage       = n_well_sampled / n_occupied if n_occupied else 1.0

    sparse = []
    for (ci, cj), member_idxs in cell_members.items():
        if len(member_idxs) >= min_count:
            continue
        centre = np.array([
            (p1_edges[ci] + p1_edges[ci + 1]) / 2,
            (p2_edges[cj] + p2_edges[cj + 1]) / 2,
        ])
        # Highest-quality member becomes the template for this cell
        best_path, best_quality = None, -1.0
        for idx in member_idxs:
            path    = valid_paths[idx]
            quality = plddt_dict.get(path, -1.0)
            if quality > best_quality:
                best_quality = quality
                best_path    = path

        neighbour_occ = sum(
            len(cell_members.get((ci + di, cj + dj), []))
            for di in (-1, 0, 1) for dj in (-1, 0, 1)
            if (di, dj) != (0, 0)
        )

        sparse.append({
            'cell':           (ci, cj),
            'centre':         centre.tolist(),
            'representative': best_path,
            'count':          len(member_idxs),
            'deficit':        min_count - len(member_idxs),
            'neighbour_occ':  neighbour_occ,
        })

    cell_counts = {k: len(v) for k, v in cell_members.items()}

    # Prioritise cells embedded in well-sampled regions (high neighbour
    # occupancy) over isolated edge outliers; break ties by deficit.
    sparse.sort(key=lambda x: (x['neighbour_occ'], x['deficit']), reverse=True)
    return sparse, coverage, n_grid, n_occupied, p1_edges, p2_edges, cell_counts


# ── Coverage grid visualisation ───────────────────────────────────────────────

def _print_coverage_grid(cell_counts, n_grid, min_count, target_cells=None):
    """
    Print an ASCII n×n grid of PC1-PC2 coverage.
      PC1 → columns  (left = min, right = max)
      PC2 ↑ rows     (top  = max, bottom = min)
    Symbols: '.' empty  '░' sparse  '█' well-sampled  '*' targeted this round
    """
    target_cells = target_cells or set()
    rows = []
    for cj in range(n_grid - 1, -1, -1):
        row = ''
        for ci in range(n_grid):
            if (ci, cj) in target_cells:
                row += '*'
            else:
                cnt = cell_counts.get((ci, cj), 0)
                if cnt == 0:
                    row += '.'
                elif cnt < min_count:
                    row += '░'
                else:
                    row += '█'
        rows.append(row)

    w = n_grid
    print(f'  PC2')
    print(f'   ↑  ┌{"─" * w}┐')
    for row in rows:
        print(f'      │{row}│')
    print(f'      └{"─" * w}┘')
    print(f'{"PC1 →":>{9 + w}}')
    print(f'  .=empty  ░=sparse(<{min_count})  █=well-sampled(≥{min_count})  *=targeted')


# ── Cell hit-rate checking ────────────────────────────────────────────────────

def _count_cell_hits(accepted_paths, ref_universe, n_ref_sel,
                     pca_model, pca_selection,
                     ci, cj, p1_edges, p2_edges):
    """
    Project each accepted PDB into the current round's PC1-PC2 space and count
    how many land in the target cell (ci, cj).

    Returns (n_hits, n_projected).
    """
    n_grid   = len(p1_edges) - 1
    hits     = 0
    projected = 0
    for path in accepted_paths:
        try:
            u  = mda.Universe(os.path.realpath(path))
            ag = u.select_atoms(pca_selection)
            if len(ag) != n_ref_sel:
                continue
            mda.analysis.align.alignto(u, ref_universe,
                                       select=pca_selection, match_atoms=False)
            flat = u.select_atoms(pca_selection).positions.flatten()
            pc   = _pca_apply(pca_model, flat)
            ci_new = int(np.clip(np.digitize([pc[0]], p1_edges) - 1, 0, n_grid - 1)[0])
            cj_new = int(np.clip(np.digitize([pc[1]], p2_edges) - 1, 0, n_grid - 1)[0])
            if ci_new == ci and cj_new == cj:
                hits += 1
            projected += 1
        except Exception:
            pass
    return hits, projected


# ── Annealing schedule ────────────────────────────────────────────────────────

def _annealed_depth(optimal_paired, viable_window, round_idx, anneal_rate):
    """
    Depth for adaptive round round_idx:
        paired = optimal_paired × anneal_rate^(round_idx - 1)
    Clamped to the deepest depth in viable_window.
    """
    target = round(optimal_paired * (anneal_rate ** (round_idx - 1)))
    if viable_window:
        cap    = max(_paired_int(d) for d in viable_window)
        target = min(target, cap)
    return f'{target}:{target * 2}'


# ── Template directory ────────────────────────────────────────────────────────

def _prepare_template_dir(template_pdb, template_dir):
    """
    Copy the representative PDB into a fresh template directory for ColabFold.
    ColabFold scans --custom-template-path for PDB/mmCIF files and uses them
    as structural priors instead of searching the PDB database.
    """
    if os.path.exists(template_dir):
        shutil.rmtree(template_dir)
    os.makedirs(template_dir)
    real_pdb = os.path.realpath(template_pdb)
    shutil.copy2(real_pdb, os.path.join(template_dir, 'tmpl.pdb'))


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def check_dependencies(dry_run=False):
    import shutil as _shutil

    print('  Checking dependencies...')
    errors = []

    cf_path = _shutil.which('colabfold_batch',
                            path=_COLABFOLD_BIN + ':' + os.environ.get('PATH', ''))
    if cf_path:
        print(f'    colabfold_batch  ✓  ({cf_path})')
    else:
        print(f'    colabfold_batch  ✗  not found in {_COLABFOLD_BIN}')
        errors.append('colabfold_batch not found')

    dssp_path = _shutil.which('mkdssp')
    if dssp_path:
        try:
            subprocess.run(['mkdssp', '--version'], capture_output=True, timeout=10)
            print(f'    mkdssp           ✓  ({dssp_path})')
        except Exception as e:
            print(f'    mkdssp           ⚠  found but failed to run ({e})')
            print(f'                        PCA will use all CA atoms instead of SS elements')
    else:
        print(f'    mkdssp           ⚠  not found — '
              f'PCA will use all CA atoms instead of SS elements')

    if errors and not dry_run:
        raise RuntimeError('Missing hard dependencies: ' + ', '.join(errors))


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _write_stats(path, stats):
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)


def _write_plddt_dict(path, plddt_dict):
    basename_plddt = {os.path.basename(k): v for k, v in plddt_dict.items()}
    with open(path, 'w') as f:
        json.dump(basename_plddt, f, indent=2)


def _update_yield(depth, n_accepted, n_total,
                  depth_accepted, depth_total, yield_per_depth):
    depth_accepted[depth]  = depth_accepted.get(depth, 0) + n_accepted
    depth_total[depth]     = depth_total.get(depth, 0)    + n_total
    yield_per_depth[depth] = (depth_accepted[depth] / depth_total[depth]
                              if depth_total[depth] else 0.0)


# ── Settings summary ──────────────────────────────────────────────────────────

def _print_settings(fasta, output_dir, coarse_depths, n_probe, fine_steps,
                    viable_fraction, budget_initial, budget_per_round,
                    n_models, seeds_per_cell, num_recycle, max_rounds,
                    quality_thresh, min_mean_quality, min_count_per_cell,
                    n_grid, coverage_threshold, anneal_rate,
                    hit_rate_target, temp_step, relaxed,
                    score_metric='plddt', quality_thresh_pct=None,
                    anneal_round0=False, round0_step_density=10):
    W            = 66
    n_d          = len(coarse_depths)
    coarse_preds = n_probe * n_d
    fine_est     = n_probe * max(0, fine_steps - 1)
    cells_pr     = budget_per_round // (seeds_per_cell * n_models)

    print('━' * W)
    print('  Template-directed Adaptive AF2 Ensemble Generator  v2')
    print('━' * W)
    print(f'  Input            : {fasta}')
    print(f'  Output           : {output_dir}')
    print(f'  Structures       : {"relaxed" if relaxed else "unrelaxed"}')
    print(f'  Num recycles     : {num_recycle}')
    print()
    print(f'  Coarse depths    : {coarse_depths[0]} → {coarse_depths[-1]}'
          f'  ({n_d} depths)')
    print(f'  Probe (coarse)   : {n_probe} seeds × 1 model × {n_d} depths'
          f'  = {coarse_preds} predictions')
    print(f'  Probe (fine)     : {n_probe} seeds × ~{fine_steps-1} depths'
          f'  ≈ {fine_est} predictions  (step ≈ {2**(1/fine_steps):.2f}×)')
    print(f'  Viable fraction  : U ≥ {viable_fraction:.0%} × U_peak → round-0 window')
    print()
    if anneal_round0:
        print(f'  Round 0          : {budget_initial} predictions  '
              f'[viable window + annealing → ceiling,  ~{round0_step_density} seeds/step]')
    else:
        print(f'  Round 0          : {budget_initial} predictions across viable window')
    print(f'  Adaptive rounds  : {max_rounds} × {budget_per_round} preds'
          f'  → ~{cells_pr} cells/round × {seeds_per_cell} seeds × {n_models} models')
    if hit_rate_target > 0:
        print(f'  Temperature      : dynamic  (target hit rate {hit_rate_target:.0%}, '
              f'step {temp_step:.2f}×)')
    elif anneal_rate != 1.0:
        print(f'  Temperature      : annealing  {anneal_rate:.2f}× depth/round from optimal')
    else:
        print(f'  Temperature      : fixed at optimal depth')
    print()
    qlbl = _METRIC_LABELS.get(score_metric, score_metric)
    print(f'  Score metric     : {score_metric}  ({qlbl}, 0–100 scale)')
    if quality_thresh_pct is not None:
        print(f'  Quality accept   : > {quality_thresh_pct:.0%} × best coarse depth  '
              f'(dynamic — resolved after probing)')
    else:
        print(f'  Quality accept   : > {quality_thresh}')
    print(f'  Min mean quality : {min_mean_quality}  (probe filter)')
    _gs = (f'{n_grid}×{n_grid}' if n_grid is not None
           else 'auto (clamp(√(N/min_count), 5, 30))')
    print(f'  Coverage grid    : {_gs},  min/cell: {min_count_per_cell}')
    print(f'  Stop when        : coverage ≥ {coverage_threshold:.0%}')
    print('━' * W)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_adaptive_ensemble_v2(
    fasta,
    output_dir,
    msa_start=1,
    msa_ceiling=None,
    depths=None,
    n_probe=25,
    fine_steps=4,
    viable_fraction=0.5,
    budget_initial=2000,
    n_models=5,
    num_recycle=0,
    max_rounds=50,
    budget_per_round=200,
    seeds_per_cell=3,
    score_metric='plddt',
    quality_thresh=None,
    quality_thresh_pct=None,
    min_mean_quality=None,
    selection_quality=True,
    pca_exclude=None,
    pca_include=None,
    min_count_per_cell=3,
    n_grid=None,
    coverage_threshold=1.0,
    n_pca_components=3,
    anneal_rate=1.0,
    hit_rate_target=0.0,
    temp_step=1.5,
    pca_interval=5,
    stable_grid=True,
    anneal_round0=False,
    round0_step_density=10,
    no_pdb_templates=True,
    extra_flags='',
    relaxed=False,
    dry_run=False,
    verbose=True,
):
    """
    Run the full template-directed adaptive pipeline.

    Returns
    -------
    structures_dir : str
    plddt_dict     : dict {symlink_path: quality}  quality normalised 0–100
    """
    # Auto-scale quality thresholds if not provided
    if quality_thresh is None:
        quality_thresh   = _METRIC_DEFAULT_THRESH[score_metric]
    if min_mean_quality is None:
        min_mean_quality = _METRIC_DEFAULT_MIN_MEAN[score_metric]

    output_dir      = output_dir.rstrip('/') + '/'
    rounds_dir      = os.path.join(output_dir, 'rounds')
    structures_dir  = os.path.join(output_dir, 'structures')
    msa_dir         = os.path.join(output_dir, 'msa')
    stats_path      = os.path.join(output_dir, 'stats.json')
    plddt_json_path = os.path.join(output_dir, 'plddt_dict.json')
    os.makedirs(rounds_dir,     exist_ok=True)
    os.makedirs(structures_dir, exist_ok=True)

    check_dependencies(dry_run=dry_run)

    # ── Pre-fetch MSA ─────────────────────────────────────────────────────────
    msa_dir, n_total_seqs, n_paired = fetch_msa(fasta, msa_dir, dry_run=dry_run)

    # ── Build coarse depth list ───────────────────────────────────────────────
    if depths is None:
        if n_total_seqs is not None and n_total_seqs > 0:
            effective_ceiling = n_total_seqs
            if msa_ceiling is not None:
                if msa_ceiling > n_total_seqs:
                    print(f'  --msa-ceiling {msa_ceiling} exceeds natural MSA depth '
                          f'({n_total_seqs}); using {n_total_seqs}')
                else:
                    effective_ceiling = msa_ceiling
            coarse_depths = _auto_depths(msa_start, effective_ceiling)
        else:
            coarse_depths = [f'{2**i}:{2**(i+1)}' for i in range(msa_start, 14)]
    else:
        coarse_depths = list(depths)
        if n_total_seqs is not None and n_total_seqs > 0:
            coarse_depths = [d for d in coarse_depths
                             if _paired_int(d) <= n_total_seqs]
            if not coarse_depths:
                raise RuntimeError(
                    f'No depths ≤ natural MSA depth ({n_total_seqs}). '
                    f'Lower --msa-start.')
            anchor = f'{n_total_seqs}:{n_total_seqs * 2}'
            if anchor not in coarse_depths:
                coarse_depths.append(anchor)

    _print_settings(fasta, output_dir, coarse_depths, n_probe, fine_steps,
                    viable_fraction, budget_initial, budget_per_round,
                    n_models, seeds_per_cell, num_recycle, max_rounds,
                    quality_thresh, min_mean_quality, min_count_per_cell,
                    n_grid, coverage_threshold, anneal_rate,
                    hit_rate_target, temp_step, relaxed,
                    score_metric=score_metric,
                    quality_thresh_pct=quality_thresh_pct,
                    anneal_round0=anneal_round0,
                    round0_step_density=round0_step_density)

    # Restore plddt_dict from a previous run if present
    plddt_dict      = {}
    yield_per_depth = {}
    depth_accepted  = {}
    depth_total     = {}

    if os.path.exists(plddt_json_path) and not dry_run:
        with open(plddt_json_path) as f:
            stored = json.load(f)
        plddt_dict = {
            os.path.join(structures_dir, k): v
            for k, v in stored.items()
            if os.path.lexists(os.path.join(structures_dir, k))
        }
        if plddt_dict:
            print(f'  Restart: found {len(plddt_dict)} existing structures.')

    all_stats = {}
    if os.path.exists(stats_path) and not dry_run:
        with open(stats_path) as f:
            all_stats = json.load(f)

    # ── Two-phase probe ───────────────────────────────────────────────────────
    optimal_depth, viable_window, probe_stats, pca_selection, quality_thresh = probe_and_optimize(
        msa_dir, rounds_dir, coarse_depths, n_probe,
        quality_thresh, min_mean_quality,
        fine_steps=fine_steps, viable_fraction=viable_fraction,
        num_recycle=num_recycle, extra_flags=extra_flags,
        no_pdb_templates=no_pdb_templates, dry_run=dry_run, relaxed=relaxed,
        score_metric=score_metric, selection_quality=selection_quality,
        quality_thresh_pct=quality_thresh_pct,
        pca_exclude=pca_exclude, pca_include=pca_include,
    )
    all_stats['probe'] = probe_stats

    if optimal_depth is None or not viable_window:
        raise RuntimeError(
            'No viable depths found by probe.  '
            'Try lowering --min-mean-plddt or --viable-fraction.')

    optimal_paired = _paired_int(optimal_depth)

    for s in probe_stats:
        d = s['depth']
        depth_total[d]     = n_probe
        depth_accepted[d]  = round(s['yield'] * n_probe)
        yield_per_depth[d] = s['yield']

    # ── PCA atom selection ────────────────────────────────────────────────────
    print()
    probe_dir = os.path.join(rounds_dir, 'probe')
    if pca_selection is None:
        # selection_quality was off (or dry-run) — build it the old way, from
        # the best structure across the whole probe (coarse + fine).
        pca_selection = _build_pca_selection(probe_dir, relaxed, dry_run=dry_run,
                                             score_metric=score_metric,
                                             pca_exclude=pca_exclude,
                                             pca_include=pca_include)
    if not dry_run:
        with open(os.path.join(output_dir, 'pca_selection.txt'), 'w') as _f:
            _f.write(pca_selection)

    # Quality gating can optionally be restricted to this same selection.
    # When enabled, probe_and_optimize already applied it from the coarse
    # sweep onward; this carries it through to round 0 and adaptive rounds.
    quality_selection = pca_selection if (selection_quality and score_metric == 'plddt') else None
    if selection_quality and score_metric != 'plddt':
        print(f'  Quality gating   : whole-model {score_metric} '
              '(selection gating needs per-residue pLDDT)')
    elif selection_quality:
        print(f'  Quality gating   : mean pLDDT over selection only  ({pca_selection})')

    # ── Round 0: distribute budget across viable window ───────────────────────
    # ── Build round-0 depth schedule ─────────────────────────────────────────
    n_surv            = len(viable_window)
    r0_ceiling_paired = _paired_int(coarse_depths[-1])

    if anneal_round0:
        r0_depths, n_seeds_r0_ann, n_ext = _build_round0_annealing_schedule(
            viable_window, r0_ceiling_paired,
            round0_step_density, n_models, budget_initial)
    else:
        r0_depths = viable_window

    n_r0_steps = len(r0_depths)
    if anneal_round0:
        n_seeds_r0 = n_seeds_r0_ann  # already computed by _build_round0_annealing_schedule
    else:
        min_budget = n_r0_steps * n_models
        if budget_initial < min_budget:
            print(f'  Note: --budget-initial {budget_initial} is below the minimum '
                  f'({n_r0_steps} depths × {n_models} models = {min_budget}).  '
                  f'Raising to {min_budget}.')
            budget_initial = min_budget
        n_seeds_r0 = (budget_initial // n_r0_steps) // n_models

    actual_r0_total = n_seeds_r0 * n_models * n_r0_steps
    if actual_r0_total != budget_initial:
        print(f'  Note: --budget-initial {budget_initial} does not divide evenly '
              f'({n_r0_steps} depths × {n_models} models). '
              f'Running {actual_r0_total} predictions '
              f'({n_seeds_r0} seeds × {n_models} models × {n_r0_steps} depths).')

    print(f'\n{"─"*66}')
    if anneal_round0:
        print(f'  Round 0  ({n_surv} viable + {n_ext} annealing = {n_r0_steps} depths,  '
              f'{r0_depths[0]} → {r0_depths[-1]},  '
              f'{n_seeds_r0} seeds × {n_models} models/depth,  {actual_r0_total} total)')
    else:
        print(f'  Round 0  ({n_r0_steps} viable depths, '
              f'{n_seeds_r0} seeds × {n_models} models = '
              f'{n_seeds_r0 * n_models} preds/depth, {actual_r0_total} total)')
    print(f'{"─"*66}')
    print(f'  {"depth":>12s}  {"accepted":>8s}  {"produced":>8s}  {"yield":>6s}')

    round0_dir   = os.path.join(rounds_dir, 'round_0')
    round0_stats = []

    with tqdm(total=actual_r0_total, desc='  round 0', unit='pred', leave=False) as pbar:
        for i, depth in enumerate(r0_depths):
            label       = _msa_label(depth)
            dir_pfx     = f'step_{i:02d}_' if anneal_round0 else ''
            struct_lbl  = f'r0_{i:02d}'    if anneal_round0 else 'r0'
            out_dir     = os.path.join(round0_dir, f'{dir_pfx}msa_{label}')
            run_colabfold(msa_dir, out_dir, depth,
                          n_seeds=n_seeds_r0, n_models=n_models,
                          num_recycle=num_recycle, extra_flags=extra_flags,
                          use_pdb_templates=not no_pdb_templates, dry_run=dry_run)

            accepted, n_total = collect_structures(
                out_dir, structures_dir, depth, struct_lbl,
                quality_thresh, relaxed, dry_run=dry_run,
                score_metric=score_metric, pca_selection=quality_selection)
            plddt_dict.update(accepted)

            n_acc = len(accepted)
            _update_yield(depth, n_acc, n_total,
                          depth_accepted, depth_total, yield_per_depth)
            yld = yield_per_depth[depth]

            round0_stats.append({
                'depth': depth, 'produced': n_total,
                'accepted': n_acc, 'yield': round(yld, 3),
            })
            pbar.update(n_seeds_r0 * n_models)
            tqdm.write(f'  {depth:>12s}  {n_acc:>8d}  {n_total:>8d}  {yld:>5.0%}')

    total_r0 = sum(s['accepted'] for s in round0_stats)
    print(f'  {"total":>12s}  {total_r0:>8d}')

    all_stats['probe_optimal'] = optimal_depth
    all_stats['viable_window'] = viable_window
    all_stats['round_0']       = round0_stats
    _write_stats(stats_path, all_stats)
    _write_plddt_dict(plddt_json_path, plddt_dict)

    # ── Adaptive rounds ───────────────────────────────────────────────────────
    n_cells_per_round  = max(1, budget_per_round // (seeds_per_cell * n_models))
    prev_eig_ratios    = None
    prev_components    = None
    prev_cell_labels   = set()

    # PCA state persists across rounds (supports pca_interval > 1 and restarts)
    pca_state_path   = os.path.join(output_dir, 'pca_state.npz')
    pca_model        = None
    pca_ref          = None
    n_ref_sel        = None
    rounds_since_pca = 0   # rounds elapsed since last PCA recompute
    total_rounds     = 0   # total adaptive rounds completed across all runs
    cached_p1_edges  = None
    cached_p2_edges  = None

    if os.path.exists(pca_state_path) and not dry_run and plddt_dict:
        try:
            (pca_model, prev_components, prev_eig_ratios,
             rounds_since_pca, total_rounds,
             cached_p1_edges, cached_p2_edges) = _load_pca_state(pca_state_path)
            grid_msg = (f', grid edges cached'
                        if cached_p1_edges is not None else '')
            print(f'  Restart: loaded PCA state '
                  f'({total_rounds} round(s) completed, '
                  f'{rounds_since_pca} since last PCA recompute, '
                  f'{pca_model.n_components_} components{grid_msg}).')
        except Exception as e:
            print(f'  Warning: could not load PCA state ({e}) — will recompute.')

    # Dynamic temperature: current_paired adjusts each round based on hit rate
    current_paired     = optimal_paired
    round_hit_rates    = []
    _vp = [_paired_int(d) for d in viable_window] if viable_window else [optimal_paired]
    viable_paired_min  = min(_vp)
    viable_paired_max  = max(_vp)

    if dry_run:
        print(f'\n  [dry-run] Would run up to {max_rounds} adaptive round(s).')
        print(f'  [dry-run] Each round: PCA → sparse cells → '
              f'select representative → run ColabFold with --custom-template-path.')
        print(f'  [dry-run] Targeting up to {n_cells_per_round} cells/round '
              f'({seeds_per_cell} seeds × {n_models} models each).')
        if hit_rate_target > 0:
            print(f'  [dry-run] Dynamic temp: target hit rate {hit_rate_target:.0%}, '
                  f'step {temp_step:.2f}×, start depth {current_paired}:{current_paired*2}.')
        elif anneal_rate != 1.0:
            print(f'  [dry-run] Annealing: {anneal_rate:.2f}× depth/round '
                  f'from {optimal_depth}.')

    for round_idx in range(1, max_rounds + 1):
        if dry_run:
            break

        structure_paths = sorted(plddt_dict)
        n_structures    = len(structure_paths)
        min_needed      = max(10, n_pca_components + 1)

        if n_structures < min_needed:
            print(f'\n  Round {round_idx}: only {n_structures} structures '
                  f'(need ≥ {min_needed}) — stopping.')
            break

        if hit_rate_target > 0:
            round_depth = f'{current_paired}:{current_paired * 2}'
        else:
            round_depth = _annealed_depth(optimal_paired, viable_window,
                                          round_idx, anneal_rate)

        total_rounds += 1
        round_label   = f'{total_rounds}' + (f'  [run round {round_idx}]'
                                              if total_rounds != round_idx else '')
        print(f'\n{"─"*66}')
        print(f'  Round {round_label}  ({n_structures} structures,  '
              f'depth = {round_depth},  budget = {budget_per_round} preds)')
        print(f'{"─"*66}')

        run_pca = (pca_model is None) or (rounds_since_pca >= pca_interval)

        if run_pca:
            print(f'  PCA ({n_pca_components} components)...', end=' ', flush=True)
            pc_coords, valid_paths, eig_ratios, components, pca_model = lightweight_pca(
                structure_paths, selection=pca_selection,
                n_components=n_pca_components)
            pca_ref          = mda.Universe(structure_paths[0])
            n_ref_sel        = len(pca_ref.select_atoms(pca_selection))
            rounds_since_pca = 1
            print(f'done  ({len(valid_paths)} structures)')
        else:
            rounds_until_pca = pca_interval - rounds_since_pca
            print(f'  Projecting onto cached PCA axes  '
                  f'(PCA re-run in {rounds_until_pca} round(s))...',
                  end=' ', flush=True)
            if pca_ref is None:
                pca_ref   = mda.Universe(structure_paths[0])
                n_ref_sel = len(pca_ref.select_atoms(pca_selection))
            valid_paths, pc_coords = _project_onto_pca(
                structure_paths, pca_ref, n_ref_sel, pca_selection, pca_model)
            eig_ratios = pca_model.explained_variance_ratio_.tolist()
            components = pca_model.components_
            rounds_since_pca += 1
            print(f'done  ({len(valid_paths)} structures)')

        eig_str  = '  '.join(f'PC{i+1}={v:.1%}' for i, v in enumerate(eig_ratios))
        overlaps = None
        if run_pca:
            if prev_components is not None:
                n_shared = min(len(components), len(prev_components))
                overlaps = [float(abs(np.dot(components[k], prev_components[k])))
                            for k in range(n_shared)]
                ov_str   = '  '.join(f'PC{k+1}={v:.3f}' for k, v in enumerate(overlaps))
                print(f'  Variance explained  : {eig_str}')
                print(f'  Eigenvector overlap : {ov_str}'
                      f'  (1.0=unchanged  0.0=orthogonal)')
            else:
                print(f'  Variance explained  : {eig_str}')
            prev_eig_ratios = eig_ratios
            prev_components = components
        else:
            print(f'  Variance explained  : {eig_str}  [cached]')

        use_cached_grid = (stable_grid and not run_pca
                           and cached_p1_edges is not None)
        (sparse_cells, coverage, n_grid_used, n_occupied,
         p1_edges, p2_edges, cell_counts) = find_sparse_cells(
            pc_coords, valid_paths, plddt_dict,
            n_grid=n_grid, min_count=min_count_per_cell,
            p1_edges=cached_p1_edges if use_cached_grid else None,
            p2_edges=cached_p2_edges if use_cached_grid else None,
        )
        # Cache / update grid edges and persist with PCA state
        cached_p1_edges = p1_edges
        cached_p2_edges = p2_edges
        _ref_basename = (os.path.basename(pca_ref.filename)
                         if pca_ref is not None else None)
        _save_pca_state(pca_state_path, pca_model, rounds_since_pca, total_rounds,
                        p1_edges, p2_edges, ref_basename=_ref_basename)
        _save_pca_state(os.path.join(output_dir, f'pca_state_round_{total_rounds}.npz'),
                        pca_model, rounds_since_pca, total_rounds, p1_edges, p2_edges,
                        ref_basename=_ref_basename)

        n_sparse     = len(sparse_cells)
        n_well       = n_occupied - n_sparse
        mean_occ     = len(valid_paths) / n_occupied if n_occupied else 0.0
        mean_deficit = (sum(c['deficit'] for c in sparse_cells) / n_sparse
                        if n_sparse else 0.0)
        grid_tag = '  [stable]' if use_cached_grid else ''
        print(f'  Coverage: {n_well}/{n_occupied} occupied cells well-sampled '
              f'({coverage:.0%})  [grid {n_grid_used}×{n_grid_used}, '
              f'min/cell={min_count_per_cell}{grid_tag}]')
        print(f'  Occupancy: mean {mean_occ:.1f} structures/cell'
              + (f'  |  sparse cells: mean deficit {mean_deficit:.1f}'
                 if n_sparse else ''))
        if coverage >= coverage_threshold:
            if verbose:
                _print_coverage_grid(cell_counts, n_grid_used, min_count_per_cell)
            print(f'  Coverage target reached ({coverage:.0%} ≥ '
                  f'{coverage_threshold:.0%}) — stopping.')
            break

        if not sparse_cells:
            if verbose:
                _print_coverage_grid(cell_counts, n_grid_used, min_count_per_cell)
            print('  No sparse cells found — stopping.')
            break

        cells_this_round  = sparse_cells[:min(n_cells_per_round, n_sparse)]
        this_cell_labels  = {f'cell_{ci}_{cj}' for ci, cj in
                             (c['cell'] for c in cells_this_round)}
        n_new = sum(1 for lbl in this_cell_labels if lbl not in prev_cell_labels)
        if verbose:
            _print_coverage_grid(cell_counts, n_grid_used, min_count_per_cell,
                                 target_cells={c['cell'] for c in cells_this_round})
        print(f'  Targeting {len(cells_this_round)} of {n_sparse} sparse cells  '
              f'({seeds_per_cell} seeds × {n_models} models each,  '
              f'{n_new} new vs prev round)\n')

        # PC lookup for representatives (for table display)
        path_to_pc2d = {path: pc_coords[i, :2].tolist()
                        for i, path in enumerate(valid_paths)}

        round_key   = f'round_{round_idx}'
        round_dir   = os.path.join(rounds_dir, round_key)
        round_stats = []

        total_preds = len(cells_this_round) * seeds_per_cell * n_models
        rep_w = 36
        print(f'  {"":1s}  {"cell":>10s}  {"deficit":>7s}  {"n_occ":>5s}  {"ctr_PC1":>8s}  '
              f'{"ctr_PC2":>8s}  {"representative":<{rep_w}s}  {"acc":>5s}  {"prod":>5s}  {"hits":>9s}')

        with tqdm(total=total_preds, desc=f'  round {round_idx}',
                  unit='pred', leave=False) as pbar:
            for cell_info in cells_this_round:
                ci, cj      = cell_info['cell']
                rep_path    = cell_info['representative']
                ctr_pc1, ctr_pc2 = cell_info['centre']
                cell_label  = f'cell_{ci}_{cj}'
                is_new      = cell_label not in prev_cell_labels
                marker      = '▶' if is_new else ' '
                cell_dir    = os.path.join(round_dir, cell_label)
                tmpl_dir    = os.path.join(cell_dir, 'template')
                cf_out_dir  = os.path.join(cell_dir,
                                           f'msa_{_msa_label(round_depth)}')

                _prepare_template_dir(rep_path, tmpl_dir)
                run_colabfold(
                    msa_dir, cf_out_dir, round_depth,
                    n_seeds=seeds_per_cell, n_models=n_models,
                    num_recycle=num_recycle, extra_flags=extra_flags,
                    template_dir=tmpl_dir,
                    dry_run=dry_run,
                )

                accepted, n_total_cell = collect_structures(
                    cf_out_dir, structures_dir, round_depth,
                    f'r{round_idx}_{cell_label}', quality_thresh, relaxed,
                    dry_run=dry_run, score_metric=score_metric,
                    pca_selection=quality_selection,
                )
                plddt_dict.update(accepted)

                n_acc    = len(accepted)
                rep_name = os.path.basename(rep_path)
                _update_yield(round_depth, n_acc, n_total_cell,
                              depth_accepted, depth_total, yield_per_depth)

                n_hits, n_proj = _count_cell_hits(
                    list(accepted.keys()), pca_ref, n_ref_sel,
                    pca_model, pca_selection,
                    ci, cj, p1_edges, p2_edges)
                hit_str = (f'{n_hits}/{n_proj} ({n_hits/n_proj:.0%})'
                           if n_proj > 0 else '—')

                pbar.update(seeds_per_cell * n_models)
                tqdm.write(
                    f'  {marker}  {cell_label:>10s}  {cell_info["deficit"]:>7d}  '
                    f'{cell_info["neighbour_occ"]:>5d}  '
                    f'{ctr_pc1:>8.3f}  {ctr_pc2:>8.3f}  '
                    f'{rep_name:<{rep_w}s}  {n_acc:>5d}  {n_total_cell:>5d}  {hit_str:>9s}')

                rep_pc = path_to_pc2d.get(rep_path, [None, None])
                round_stats.append({
                    'cell':           cell_label,
                    'new_this_round': is_new,
                    'cell_pc1':       round(ctr_pc1, 3),
                    'cell_pc2':       round(ctr_pc2, 3),
                    'representative': rep_name,
                    'rep_pc1':        round(rep_pc[0], 3) if rep_pc[0] is not None else None,
                    'rep_pc2':        round(rep_pc[1], 3) if rep_pc[1] is not None else None,
                    'count_before':   cell_info['count'],
                    'deficit':        cell_info['deficit'],
                    'neighbour_occ':  cell_info['neighbour_occ'],
                    'depth':          round_depth,
                    'produced':       n_total_cell,
                    'accepted':       n_acc,
                    'cell_hits':      n_hits,
                    'cell_projected': n_proj,
                })

        prev_cell_labels = this_cell_labels

        total_round  = sum(s['accepted'] for s in round_stats)
        rnd_hits     = sum(s.get('cell_hits', 0) for s in round_stats)
        rnd_proj     = sum(s.get('cell_projected', 0) for s in round_stats)
        rnd_hit_rate = rnd_hits / rnd_proj if rnd_proj > 0 else None

        hit_str = (f'  hit rate {rnd_hit_rate:.0%} ({rnd_hits}/{rnd_proj})'
                   if rnd_hit_rate is not None else '')
        print(f'  Round {round_idx} total: {total_round} new structures{hit_str}  '
              f'({len(plddt_dict)} accumulated)')

        # Dynamic temperature: adjust current_paired for the next round
        next_depth_str = ''
        if hit_rate_target > 0 and rnd_hit_rate is not None:
            round_hit_rates.append(rnd_hit_rate)
            old_paired = current_paired
            if rnd_hit_rate < hit_rate_target:
                current_paired = min(round(current_paired * temp_step),
                                     viable_paired_max)
            elif rnd_hit_rate > hit_rate_target * 2.0:
                current_paired = max(round(current_paired / temp_step),
                                     viable_paired_min)
            if current_paired != old_paired:
                arrow = '↑' if current_paired > old_paired else '↓'
                cmp   = '<' if current_paired > old_paired else '>'
                print(f'  Dynamic temp: hit rate {rnd_hit_rate:.0%} {cmp} target '
                      f'{hit_rate_target:.0%}  → depth {arrow} '
                      f'{old_paired}:{old_paired*2} → {current_paired}:{current_paired*2}')
            else:
                print(f'  Dynamic temp: hit rate {rnd_hit_rate:.0%}  '
                      f'depth unchanged ({current_paired}:{current_paired*2})')

        all_stats[round_key] = {
            'depth':           round_depth,
            'pca_recomputed':  run_pca,
            'coverage_before': round(coverage, 3),
            'eig_ratios':      [round(v, 4) for v in eig_ratios],
            'eig_overlaps':    [round(v, 4) for v in overlaps] if overlaps is not None else None,
            'hit_rate':        round(rnd_hit_rate, 3) if rnd_hit_rate is not None else None,
            'cells':           round_stats,
        }
        _write_stats(stats_path, all_stats)
        _write_plddt_dict(plddt_json_path, plddt_dict)

    # ── Final output ──────────────────────────────────────────────────────────
    _write_plddt_dict(plddt_json_path, plddt_dict)
    total = len(plddt_dict)
    print(f'\n{"━"*66}')
    print(f'  Done: {total} accepted structures')
    print(f'  structures/     →  {structures_dir}')
    print(f'  stats.json      →  {stats_path}')
    print(f'  plddt_dict.json →  {plddt_json_path}')
    print(f'{"━"*66}')
    return structures_dir, plddt_dict


# ── Fill mode runner ──────────────────────────────────────────────────────────

def _find_fill_targets(vals_sorted, paths_sorted, interpolate, extrapolate,
                       extrap_directions, min_gap, max_points_per_gap, extrap_step):
    """
    One pass of gap/extrapolation-target discovery against the current
    (sorted) axis values.  Pulled out of run_fill_mode() so multi-round fill
    can re-derive targets from scratch each round against updated data
    without duplicating the logic.
    """
    targets = []

    if interpolate:
        for i in range(len(vals_sorted) - 1):
            gap = vals_sorted[i + 1] - vals_sorted[i]
            if gap <= min_gap:
                continue
            n_pts = min(max_points_per_gap, max(1, int(round(gap / min_gap)) - 1))
            for frac in np.linspace(0, 1, n_pts + 2)[1:-1]:
                t = vals_sorted[i] + frac * gap
                targets.append({
                    'kind':    'interp',
                    'lo_path': paths_sorted[i],     'lo_val': float(vals_sorted[i]),
                    'hi_path': paths_sorted[i + 1],  'hi_val': float(vals_sorted[i + 1]),
                    'target':  float(t),
                })

    if extrapolate:
        directions = (['low', 'high'] if extrap_directions == 'both'
                      else [extrap_directions])
        if 'high' in directions:
            x_edge, edge_path = vals_sorted[-1], paths_sorted[-1]
            idx = np.searchsorted(vals_sorted, x_edge - min_gap, side='right') - 1
            idx = int(np.clip(idx, 0, len(vals_sorted) - 2))
            x_ref, ref_path = vals_sorted[idx], paths_sorted[idx]
            t = x_edge + extrap_step * (x_edge - x_ref)
            targets.append({
                'kind':    'extrap_high',
                'lo_path': ref_path,  'lo_val': float(x_ref),
                'hi_path': edge_path, 'hi_val': float(x_edge),
                'target':  float(t),
            })
        if 'low' in directions:
            x_edge, edge_path = vals_sorted[0], paths_sorted[0]
            idx = np.searchsorted(vals_sorted, x_edge + min_gap, side='left')
            idx = int(np.clip(idx, 1, len(vals_sorted) - 1))
            x_ref, ref_path = vals_sorted[idx], paths_sorted[idx]
            t = x_edge - extrap_step * (x_ref - x_edge)
            targets.append({
                'kind':    'extrap_low',
                'lo_path': edge_path, 'lo_val': float(x_edge),
                'hi_path': ref_path,  'hi_val': float(x_ref),
                'target':  float(t),
            })

    # Largest gaps / extrapolations first
    targets.sort(
        key=lambda t: abs(t['target'] - t['hi_val']) if t['kind'] != 'interp'
                      else abs(t['target'] - (t['lo_val'] + t['hi_val']) / 2),
        reverse=True)
    return targets


def run_fill_mode(
    fasta,
    output_dir,
    fill_axis=1,
    interpolate=False,
    extrapolate=False,
    extrap_directions='both',
    min_gap=None,
    min_gap_percentile=99.0,
    max_points_per_gap=1,
    extrap_step=1.0,
    max_targets=None,
    fill_rounds=1,
    depth=None,
    seeds=5,
    n_models=5,
    num_recycle=0,
    score_metric='plddt',
    quality_thresh=None,
    min_mean_quality=None,
    selection_quality=False,
    relaxed=False,
    extra_flags='',
    dry_run=False,
):
    """
    Post-hoc gap-filling pass against an already-completed (or paused) run.
    See the module docstring's "Fill mode" section for the method.

    With --fill-rounds > 1, targets are re-derived from scratch each round
    against whatever got newly accepted in the previous round (e.g. for
    --fill-extrapolate, an accepted structure beyond the old edge becomes
    the new edge to extrapolate further from next round).  A round that
    accepts zero new structures stops the loop early -- further rounds would
    just regenerate and retry the same target.  Only the newly accepted
    structures are re-projected between rounds (not the whole ensemble), so
    the cost of extra rounds scales with what fill itself produced, not with
    the size of the input ensemble.

    Returns
    -------
    structures_dir : str
    plddt_dict     : dict {symlink_path: quality}
    """
    if not interpolate and not extrapolate:
        raise ValueError(
            '--fill requires at least one of --fill-interpolate / --fill-extrapolate.')

    if quality_thresh is None:
        quality_thresh = _METRIC_DEFAULT_THRESH[score_metric]
    if min_mean_quality is None:
        min_mean_quality = _METRIC_DEFAULT_MIN_MEAN[score_metric]

    output_dir      = output_dir.rstrip('/') + '/'
    rounds_dir      = os.path.join(output_dir, 'rounds')
    structures_dir  = os.path.join(output_dir, 'structures')
    msa_dir         = os.path.join(output_dir, 'msa')
    stats_path      = os.path.join(output_dir, 'stats.json')
    plddt_json_path = os.path.join(output_dir, 'plddt_dict.json')
    pca_state_path  = os.path.join(output_dir, 'pca_state.npz')
    pca_sel_path    = os.path.join(output_dir, 'pca_selection.txt')

    for p, desc in [(plddt_json_path, 'plddt_dict.json'),
                    (pca_state_path, 'pca_state.npz'),
                    (pca_sel_path, 'pca_selection.txt'),
                    (msa_dir, 'msa/')]:
        if not os.path.exists(p):
            raise RuntimeError(
                f'--fill requires a completed run in {output_dir} '
                f'(missing {desc}).  Run without --fill first.')

    check_dependencies(dry_run=dry_run)
    msa_dir, _, _ = fetch_msa(fasta, msa_dir, dry_run=dry_run)

    with open(plddt_json_path) as f:
        stored = json.load(f)
    plddt_dict = {
        os.path.join(structures_dir, k): v for k, v in stored.items()
        if os.path.lexists(os.path.join(structures_dir, k))
    }
    if not plddt_dict:
        raise RuntimeError(f'No structures found in {structures_dir}.')

    with open(pca_sel_path) as f:
        pca_selection = f.read().strip()

    (pca_model, _components, _eig_ratios, _rspca, _tr,
     _p1, _p2) = _load_pca_state(pca_state_path)
    if fill_axis < 1 or fill_axis > pca_model.n_components_:
        raise ValueError(f'--fill-axis {fill_axis} out of range '
                          f'(PCA has {pca_model.n_components_} components).')
    axis_idx = fill_axis - 1

    all_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            all_stats = json.load(f)

    if depth is None:
        depth = all_stats.get('probe_optimal')
        if depth is None:
            raise RuntimeError(
                '--fill-depth not given and no probe_optimal found in '
                f'{stats_path}; specify --fill-depth explicitly.')

    structure_paths = sorted(plddt_dict)
    pca_ref   = mda.Universe(structure_paths[0])
    n_ref_sel = len(pca_ref.select_atoms(pca_selection))

    print(f'\n{"─"*66}')
    print(f'  Fill mode  —  PC{fill_axis}  '
          f'(interpolate={interpolate}  extrapolate={extrapolate})')
    print(f'{"─"*66}')
    print(f'  Projecting {len(structure_paths)} existing structures onto PC{fill_axis}...',
          end=' ', flush=True)
    valid_paths, pc_coords = _project_onto_pca(
        structure_paths, pca_ref, n_ref_sel, pca_selection, pca_model)
    print(f'done  ({len(valid_paths)} projected)')

    axis_vals    = pc_coords[:, axis_idx]
    order        = np.argsort(axis_vals)
    vals_sorted  = axis_vals[order]
    paths_sorted = [valid_paths[i] for i in order]

    deltas = np.diff(vals_sorted)
    deltas = deltas[deltas > 0]
    if len(deltas):
        pcts = np.percentile(deltas, [50, 90, 99, 100])
        print(f'  PC{fill_axis} gap distribution (nearest-neighbour spacing): '
              f'median={pcts[0]:.3f}  p90={pcts[1]:.3f}  p99={pcts[2]:.3f}  '
              f'max={pcts[3]:.3f}')

    if min_gap is None:
        min_gap = (float(np.percentile(deltas, min_gap_percentile))
                   if len(deltas) else 1.0)
        print(f'  Auto min-gap: p{min_gap_percentile:.1f} of gap distribution '
              f'= {min_gap:.3f}  (PC{fill_axis} units) -- if this over/under-shoots, '
              f'tune --fill-min-gap-percentile or set --fill-min-gap directly '
              f'(see the distribution above)')

    if dry_run:
        targets = _find_fill_targets(vals_sorted, paths_sorted, interpolate, extrapolate,
                                     extrap_directions, min_gap, max_points_per_gap, extrap_step)
        if max_targets is not None:
            targets = targets[:max_targets]
        if not targets:
            print('  No fill targets found (no gaps exceeded --fill-min-gap; '
                  'try lowering it, or enable --fill-extrapolate).')
            return structures_dir, plddt_dict
        print(f'\n  [dry-run] Round 1 would generate {len(targets)} fill template(s) '
              f'and run ColabFold at depth {depth}.')
        for t in targets[:10]:
            print(f'    {t["kind"]:<12s}  target PC{fill_axis}={t["target"]:7.2f}  '
                  f'donors=({os.path.basename(t["lo_path"])}, '
                  f'{os.path.basename(t["hi_path"])})')
        if len(targets) > 10:
            print(f'    ... and {len(targets) - 10} more')
        if fill_rounds > 1:
            print(f'  [dry-run] --fill-rounds={fill_rounds} requested, but only round 1 '
                  'can be previewed -- later rounds depend on which structures actually '
                  'get accepted, which dry-run never runs.')
        return structures_dir, plddt_dict

    fill_dir   = os.path.join(rounds_dir, 'fill')
    tol        = min_gap / 2
    quality_selection = pca_selection if (selection_quality and score_metric == 'plddt') else None
    all_round_stats   = []
    run_entry_idx     = len(all_stats.setdefault('fill_runs', []))

    for round_idx in range(1, fill_rounds + 1):
        targets = _find_fill_targets(vals_sorted, paths_sorted, interpolate, extrapolate,
                                     extrap_directions, min_gap, max_points_per_gap, extrap_step)
        if max_targets is not None:
            targets = targets[:max_targets]

        if not targets:
            print(f'\n  Fill round {round_idx}: no targets found -- stopping.')
            break

        print(f'\n  Fill round {round_idx}/{fill_rounds}: {len(targets)} target(s)  '
              f'({seeds} seeds x {n_models} models each, depth={depth})')

        round_stats    = []
        newly_accepted = {}

        for i, t in enumerate(targets):
            label    = f'fill_r{round_idx}_{i:03d}_{t["kind"]}'
            cell_dir = os.path.join(fill_dir, label)
            tmpl_dir = os.path.join(cell_dir, 'template')
            cf_out   = os.path.join(cell_dir, f'msa_{_msa_label(depth)}')

            if os.path.exists(tmpl_dir):
                shutil.rmtree(tmpl_dir)
            os.makedirs(tmpl_dir)
            tmpl_path = os.path.join(tmpl_dir, 'tmpl.pdb')

            w_lo, w_hi = _build_blend_template(
                t['lo_path'], t['hi_path'], t['target'], t['lo_val'], t['hi_val'],
                pca_ref, pca_selection, tmpl_path)

            run_colabfold(
                msa_dir, cf_out, depth,
                n_seeds=seeds, n_models=n_models, num_recycle=num_recycle,
                extra_flags=extra_flags, template_dir=tmpl_dir, dry_run=dry_run)

            accepted, n_total = collect_structures(
                cf_out, structures_dir, depth, label, quality_thresh, relaxed,
                dry_run=dry_run, score_metric=score_metric, pca_selection=quality_selection)
            plddt_dict.update(accepted)
            newly_accepted.update(accepted)

            n_hits, n_proj = _count_axis_hits(
                list(accepted.keys()), pca_ref, n_ref_sel, pca_model, pca_selection,
                axis_idx, t['target'], tol)
            hit_str = f'{n_hits}/{n_proj} ({n_hits/n_proj:.0%})' if n_proj else '—'

            print(f'  {label:<28s}  target PC{fill_axis}={t["target"]:7.2f}  '
                  f'donors=({t["lo_val"]:.2f}, {t["hi_val"]:.2f})  '
                  f'w=({w_lo:.2f},{w_hi:.2f})  '
                  f'accepted={len(accepted)}/{n_total}  hit_rate={hit_str}')

            round_stats.append({
                'label':     label,      'kind':   t['kind'],
                'target':    round(t['target'], 3),
                'lo_path':   os.path.basename(t['lo_path']), 'lo_val': round(t['lo_val'], 3),
                'hi_path':   os.path.basename(t['hi_path']), 'hi_val': round(t['hi_val'], 3),
                'w_lo':      round(w_lo, 3), 'w_hi': round(w_hi, 3),
                'produced':  n_total,     'accepted': len(accepted),
                'hit_rate':  round(n_hits / n_proj, 3) if n_proj else None,
            })

        all_round_stats.append({'round': round_idx, 'targets': round_stats})
        this_run_entry = {
            'fill_axis':    fill_axis,
            'min_gap':      round(min_gap, 4),
            'interpolate':  interpolate,
            'extrapolate':  extrapolate,
            'depth':        depth,
            'rounds':       all_round_stats,
        }
        if run_entry_idx < len(all_stats['fill_runs']):
            all_stats['fill_runs'][run_entry_idx] = this_run_entry
        else:
            all_stats['fill_runs'].append(this_run_entry)
        _write_stats(stats_path, all_stats)
        _write_plddt_dict(plddt_json_path, plddt_dict)

        total_new_round = sum(s['accepted'] for s in round_stats)
        print(f'  Fill round {round_idx} total: {total_new_round} new accepted structures  '
              f'({len(plddt_dict)} accumulated total)')

        if total_new_round == 0:
            print(f'  Fill round {round_idx}: 0 new accepted structures -- stopping early '
                  '(further rounds would just regenerate and retry the same target(s)).')
            break

        if round_idx < fill_rounds:
            new_paths = sorted(newly_accepted)
            new_valid, new_pc = _project_onto_pca(
                new_paths, pca_ref, n_ref_sel, pca_selection, pca_model)
            if new_valid:
                vals_sorted  = np.concatenate([vals_sorted, new_pc[:, axis_idx]])
                paths_sorted = paths_sorted + new_valid
                order        = np.argsort(vals_sorted)
                vals_sorted  = vals_sorted[order]
                paths_sorted = [paths_sorted[i] for i in order]

    total_new = sum(s['accepted'] for round_s in all_round_stats for s in round_s['targets'])
    print(f'\n{"━"*66}')
    print(f'  Fill done: {len(all_round_stats)} round(s), '
          f'{total_new} new accepted structures  '
          f'({len(plddt_dict)} accumulated total)')
    print(f'{"━"*66}')
    return structures_dir, plddt_dict


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Template-directed adaptive AF2 ensemble generation (v2).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Typical use, from the target directory (reads input.fasta, '
               'writes synthetic_ensemble/):  generate_af2_ensemble_v2.py --dry-run, '
               'then generate_af2_ensemble_v2.py.  Rerun the same command to resume; '
               'raise --max-rounds to extend.  Multimers: --score-metric iptm.  '
               'Gap filling along PC1 afterwards: --fill --fill-interpolate.',
    )
    p.add_argument('fasta',      nargs='?', default='input.fasta',
                   help='Input FASTA file.')
    p.add_argument('output_dir', nargs='?', default='synthetic_ensemble',
                   help='Root output directory.')

    g = p.add_argument_group('MSA depth range')
    g.add_argument('--msa-start',   type=int, default=1, metavar='EXP',
                   help='log2 of smallest paired MSA depth  (1 → 2:4).')
    g.add_argument('--msa-ceiling', type=int, default=None, metavar='N',
                   help='Override auto-computed depth ceiling (absolute count).')

    g = p.add_argument_group('Probe')
    g.add_argument('--n-probe',         type=int,   default=25,
                   help='Seeds per depth in probe rounds.')
    g.add_argument('--fine-steps',      type=int,   default=4,
                   help='Geometric steps per octave in the fine probe.  '
                        'fine_steps=4 → step ≈ 1.19× (3 new depths per octave).')
    g.add_argument('--viable-fraction', type=float, default=0.5,
                   help='Depths with U ≥ this fraction of U_peak enter the '
                        'viable window (round-0 sampling range).')
    g.add_argument('--min-mean-quality', type=float, default=None,
                   help='Probe depths whose mean quality score (0–100) is below '
                        'this are marked noisy and excluded from the viable window.  '
                        'Default: 50 for plddt, 30 for pTM/ipTM-based metrics.')

    g = p.add_argument_group('Sampling budget')
    g.add_argument('--budget-initial',   type=int, default=2000,
                   help='Predictions for round 0, spread across the viable window.')
    g.add_argument('--budget-per-round', type=int, default=200,
                   help='Predictions per adaptive round.')
    g.add_argument('--n-models',         type=int, default=5,
                   help='AF2 models per seed.')
    g.add_argument('--seeds-per-cell',   type=int, default=3,
                   help='Seeds per targeted sparse cell in adaptive rounds.  '
                        'n_cells/round = budget_per_round // (seeds_per_cell × n_models).')
    g.add_argument('--num-recycle',      type=int, default=0,
                   help='AF2 recycling iterations (0 = single forward pass).')
    g.add_argument('--max-rounds',       type=int, default=50,
                   help='Maximum adaptive rounds after round 0.')

    g = p.add_argument_group('Quality filters')
    g.add_argument('--score-metric', choices=SCORE_METRICS, default='plddt',
                   help='Quality metric used for accept/reject filtering and '
                        'representative selection.  All metrics are normalised to '
                        '0–100: pLDDT as-is; pTM and ipTM multiplied by 100.  '
                        'iptm_composite = 0.8×ipTM + 0.2×pTM (ColabFold ranking '
                        'convention).  Default: plddt (suitable for monomers).  '
                        'For multimer runs, iptm or iptm_composite is recommended.')
    g.add_argument('--quality-thresh', type=float, default=None,
                   help='Minimum quality score (0–100) to accept a structure.  '
                        'Default: 80 for plddt, 50 for pTM/ipTM-based metrics.  '
                        'Ignored if --quality-thresh-pct is set.')
    g.add_argument('--quality-thresh-pct', type=float, default=None, metavar='PCT',
                   help='Make the accept threshold protein-relative instead of '
                        'absolute: right after the coarse probe sweep, '
                        'quality_thresh is set to PCT × the best coarse '
                        "depth's mean quality (e.g. 0.9 for 90%%), overriding "
                        '--quality-thresh / its metric default.  The coarse '
                        'stats are rescored with the resolved threshold, and '
                        'it carries through the fine sweep, round 0, and all '
                        'adaptive rounds.  Useful for targets that never '
                        'reach the fixed default in their ordered core '
                        '(e.g. small or IDR-rich proteins).')
    g.add_argument('--relaxed', action='store_true',
                   help='Collect relaxed PDBs (requires --amber in --extra-flags).')
    g.add_argument('--selection-quality', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Gate quality on mean pLDDT over the PCA selection '
                        '(SS-element core) instead of the whole-chain mean, '
                        'read from PDB B-factors.  Ignores intrinsically '
                        'disordered / loop regions outside the selection when '
                        'accepting structures and picking sparse-cell '
                        'representatives.  Only applies with --score-metric '
                        'plddt.  The selection is built from the coarse probe '
                        'sweep (the earliest point real structures exist for '
                        'DSSP); coarse-sweep stats are then retroactively '
                        'rescored with it, and the fine sweep, round 0, and '
                        'all adaptive rounds apply it directly.')
    g.add_argument('--pca-exclude', type=str, default=None, metavar='MDA_SEL',
                   help='MDAnalysis selection token for residues to exclude from '
                        'the PCA selection (and --selection-quality gating), e.g. '
                        '"resid 1:120" to drop a soluble domain the adaptive '
                        'seeding should ignore.  Applied on top of the DSSP '
                        'secondary-structure selection.')
    g.add_argument('--pca-include', type=str, default=None, metavar='MDA_SEL',
                   help='MDAnalysis selection token for residues to always '
                        'include (by CA) in the PCA selection even if DSSP does '
                        'not assign them to a secondary-structure element, e.g. '
                        'a functional loop of interest.')

    g = p.add_argument_group('Coverage analysis')
    g.add_argument('--min-count-per-cell',  type=int,   default=3,
                   help='Grid cells with fewer structures are considered sparse.')
    g.add_argument('--coverage-threshold',  type=float, default=1.0,
                   help='Stop when this fraction of occupied cells are well-sampled.  '
                        'Set to 1.0 to effectively disable this early stop — coverage '
                        'can only reach exactly 1.0 when there are no sparse cells '
                        'left, which already ends the run via that separate check — '
                        'so --max-rounds becomes the only limit.')
    g.add_argument('--n-grid',              type=int,   default=None,
                   help='Override automatic grid resolution (n × n).')
    g.add_argument('--n-pca-components',    type=int,   default=3,
                   help='PCA components (grid uses first 2).')

    g = p.add_argument_group('Round-0 annealing')
    g.add_argument('--anneal-round0', action='store_true', default=False,
                   help='Extend round-0 sampling above the viable window up to the '
                        'MSA ceiling.  The schedule is: viable_window depths (as '
                        'normal) followed by a geometric extension to ceiling.  '
                        'Seeds the ensemble with high-coevolution-constrained '
                        'end-state conformations alongside the diverse viable-window '
                        'structures.')
    g.add_argument('--round0-step-density', type=int, default=10, metavar='N',
                   help='Target seeds per depth step in the round-0 annealing '
                        'extension.  Determines how many extension steps fit in '
                        'the budget: total_steps = budget // (N × n_models), '
                        'extension_steps = total_steps − len(viable_window).  '
                        'Seeds per step is recalculated from actual total steps '
                        'so the full budget is used.  Default: 10.')

    g = p.add_argument_group('PCA schedule')
    g.add_argument('--pca-interval', type=int, default=5, metavar='N',
                   help='Re-run PCA every N adaptive rounds.  Between re-runs, '
                        'structures are projected onto the cached eigenvectors.  '
                        'Eigenvectors and grid edges are written to pca_state.npz '
                        'and reloaded automatically on restart.  '
                        '1 = re-run every round.')
    g.add_argument('--no-stable-grid', action='store_true', default=False,
                   help='Recompute grid edges every round even when PCA axes are '
                        'cached (--pca-interval > 1).  Default is to hold grid '
                        'edges fixed for the duration of each PCA interval so cell '
                        'indices are stable.')

    g = p.add_argument_group('Annealing / dynamic temperature')
    g.add_argument('--hit-rate-target', type=float, default=0.0,
                   help='Target fraction (0–1) of accepted structures that land in '
                        'the template target cell.  E.g. 0.25 = 25%%.  '
                        'If below this, depth is increased by --temp-step to make '
                        'templates more directive.  '
                        'Default 0 disables dynamic temperature (fixed depth at the '
                        'probe optimum, or --anneal-rate).  E.g. 0.5 turns it on.')
    g.add_argument('--temp-step', type=float, default=1.5,
                   help='Multiplicative factor for depth adjustment when hit rate '
                        'deviates from --hit-rate-target.  1.5 = 50%% change per round.')
    g.add_argument('--anneal-rate', type=float, default=1.0,
                   help='Fallback depth multiplier per round when --hit-rate-target=0.  '
                        '1.0 = fixed, 1.2 = 20%% deeper each round.')

    g = p.add_argument_group('Fill mode (post-hoc PC-axis gap filling)')
    g.add_argument('--fill', action='store_true', default=False,
                   help='Skip the probe/round-0/adaptive pipeline and instead run '
                        'a one-off post-hoc pass against an existing completed '
                        '(or paused) run in output_dir: builds synthetic '
                        'templates by linearly blending real accepted structures '
                        "so their PCA projection lands at a chosen point along "
                        'one PC axis, then runs ColabFold with that template '
                        'exactly like an adaptive round.  Requires at least one '
                        'of --fill-interpolate / --fill-extrapolate.  The '
                        'synthetic blend is never itself added to the output '
                        'ensemble -- only real ColabFold predictions that pass '
                        'the normal quality filters are.')
    g.add_argument('--fill-axis', type=int, default=1, metavar='N',
                   help='Which PCA component (1-indexed) to fill gaps along.')
    g.add_argument('--fill-interpolate', action='store_true', default=False,
                   help='Fill interior gaps: for consecutive sampled structures '
                        'along --fill-axis whose separation exceeds '
                        '--fill-min-gap, generate template(s) at evenly spaced '
                        'point(s) between them.')
    g.add_argument('--fill-extrapolate', action='store_true', default=False,
                   help='Generate one template beyond each sampled extreme '
                        'along --fill-axis (e.g. to push further into a '
                        'pore-opening motion than anything sampled).  Donors '
                        'are the extreme structure and the nearest structure at '
                        'least --fill-min-gap back from it, extrapolated '
                        '--fill-extrap-step x their separation beyond the edge.')
    g.add_argument('--fill-extrap-direction', choices=['low', 'high', 'both'],
                   default='both',
                   help='Which end(s) of --fill-axis to extrapolate beyond.')
    g.add_argument('--fill-min-gap', type=float, default=None, metavar='PC_UNITS',
                   help='Minimum separation (in PC-axis units) between '
                        'consecutive structures to count as a gap worth filling '
                        '/ a valid extrapolation step-back reference.  Default: '
                        'auto, the --fill-min-gap-percentile percentile of the '
                        'observed nearest-neighbour spacing along the axis.  '
                        'Median-based scaling is deliberately not used here: '
                        'for a bimodal/discontinuous PC (e.g. a helix that '
                        'packs two different ways) the handful of genuinely '
                        'large gaps get swamped by many small ones from local '
                        'sampling density, so a high percentile is used '
                        'instead.  Run with --dry-run first to see the printed '
                        'gap-distribution percentiles and tune this manually if '
                        'the auto value over/under-shoots.')
    g.add_argument('--fill-min-gap-percentile', type=float, default=99.0,
                   help='Percentile of the nearest-neighbour gap distribution '
                        'used to auto-compute --fill-min-gap when not given.')
    g.add_argument('--fill-max-points-per-gap', type=int, default=1, metavar='N',
                   help='Maximum evenly spaced fill targets to place inside a '
                        'single large gap.')
    g.add_argument('--fill-extrap-step', type=float, default=1.0,
                   help='How far beyond the sampled edge to extrapolate, in '
                        'units of the donor pair separation.  1.0 = one more '
                        'step the same size as the last observed gap.')
    g.add_argument('--fill-max-targets', type=int, default=None, metavar='N',
                   help='Cap on the number of fill targets processed per round '
                        '(largest gaps / extrapolations first).  Default: '
                        'process all found.')
    g.add_argument('--fill-rounds', type=int, default=1, metavar='N',
                   help='Number of fill rounds to run in one invocation.  Each '
                        'round re-derives targets from scratch against '
                        'whatever was newly accepted in the previous round -- '
                        'e.g. for --fill-extrapolate, a structure accepted '
                        'beyond the old edge becomes the new edge to '
                        'extrapolate further from.  A round that accepts zero '
                        'new structures stops the loop early.  Only the newly '
                        'accepted structures are re-projected between rounds, '
                        'not the whole ensemble.  Default 1 = single pass '
                        '(the previous manual "call --fill again" workflow).')
    g.add_argument('--fill-depth', type=str, default=None, metavar='PAIRED:UNPAIRED',
                   help='MSA depth for fill ColabFold runs.  Default: reuse the '
                        "run's probe-optimal depth from stats.json.")
    g.add_argument('--fill-seeds', type=int, default=5,
                   help='Seeds per fill target.')

    p.add_argument('--pdb-templates', action='store_true', default=False,
                   help='Allow ColabFold to use PDB database templates in probe '
                        'and round-0 runs.  Default is off (--no-templates is '
                        'passed) so all non-adaptive runs are pure MSA predictions.  '
                        'Adaptive rounds always use --custom-template-path regardless.')
    p.add_argument('--extra-flags', type=str, default='',
                   help='Extra flags passed verbatim to colabfold_batch '
                        '(e.g. "--amber --relax-max-iterations 200").')
    p.add_argument('--dry-run', action='store_true',
                   help='Print settings and commands without running ColabFold.  '
                        'Probe uses synthetic values so the logic is exercised.')
    p.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True,
                   help='Print an ASCII N×N grid visualising PC1-PC2 coverage '
                        'after each adaptive round.  '
                        'Symbols: . empty  ░ sparse  █ well-sampled.')

    args = p.parse_args()

    if args.fill:
        run_fill_mode(
            fasta              = args.fasta,
            output_dir         = args.output_dir,
            fill_axis          = args.fill_axis,
            interpolate        = args.fill_interpolate,
            extrapolate        = args.fill_extrapolate,
            extrap_directions  = args.fill_extrap_direction,
            min_gap            = args.fill_min_gap,
            min_gap_percentile = args.fill_min_gap_percentile,
            max_points_per_gap = args.fill_max_points_per_gap,
            extrap_step        = args.fill_extrap_step,
            max_targets        = args.fill_max_targets,
            fill_rounds        = args.fill_rounds,
            depth              = args.fill_depth,
            seeds              = args.fill_seeds,
            n_models           = args.n_models,
            num_recycle        = args.num_recycle,
            score_metric       = args.score_metric,
            quality_thresh     = args.quality_thresh,
            min_mean_quality   = args.min_mean_quality,
            selection_quality  = args.selection_quality,
            relaxed            = args.relaxed,
            extra_flags        = args.extra_flags,
            dry_run            = args.dry_run,
        )
        return

    run_adaptive_ensemble_v2(
        fasta              = args.fasta,
        output_dir         = args.output_dir,
        msa_start          = args.msa_start,
        msa_ceiling        = args.msa_ceiling,
        n_probe            = args.n_probe,
        fine_steps         = args.fine_steps,
        viable_fraction    = args.viable_fraction,
        budget_initial     = args.budget_initial,
        budget_per_round   = args.budget_per_round,
        n_models           = args.n_models,
        seeds_per_cell     = args.seeds_per_cell,
        num_recycle        = args.num_recycle,
        max_rounds         = args.max_rounds,
        score_metric       = args.score_metric,
        quality_thresh     = args.quality_thresh,
        quality_thresh_pct = args.quality_thresh_pct,
        min_mean_quality   = args.min_mean_quality,
        selection_quality  = args.selection_quality,
        pca_exclude        = args.pca_exclude,
        pca_include        = args.pca_include,
        min_count_per_cell = args.min_count_per_cell,
        coverage_threshold = args.coverage_threshold,
        n_grid             = args.n_grid,
        n_pca_components   = args.n_pca_components,
        anneal_rate        = args.anneal_rate,
        hit_rate_target    = args.hit_rate_target,
        temp_step          = args.temp_step,
        anneal_round0         = args.anneal_round0,
        round0_step_density = args.round0_step_density,
        pca_interval       = args.pca_interval,
        stable_grid        = not args.no_stable_grid,
        no_pdb_templates   = not args.pdb_templates,
        extra_flags        = args.extra_flags,
        relaxed            = args.relaxed,
        dry_run            = args.dry_run,
        verbose            = args.verbose,
    )


if __name__ == '__main__':
    main()
