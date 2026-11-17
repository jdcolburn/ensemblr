"""
Write every accepted structure of a synthetic ensemble as one aligned trajectory.

    <output_dir>/structures.pdb         topology (the first structure)
    <output_dir>/structures.xtc         one frame per accepted structure, in (round, name) order,
                                        superposed on the first frame over the PCA selection
    <output_dir>/structures_index.csv   frame, file, round, depth, cell, quality (pLDDT/ipTM)

Load it with e.g. ``vmd structures.pdb structures.xtc`` or
``mda.Universe('structures.pdb', 'structures.xtc')``; the index maps frames back to files.

Command line (for runs made before this existed):

    ensemblr-xtc <output_dir>            # the generator's output_dir, e.g. synthetic_ensemble/
"""
import argparse
import csv
import json
import os
import re
import warnings

import MDAnalysis as mda
from MDAnalysis.analysis import align

_NAME_RE = re.compile(r'^r(?P<round>\d+)_(?:(?P<cell>cell_\d+_\d+)_)?(?P<depth>\d+-\d+)_')


def _parse_name(name):
    """Round, cell and MSA depth from an accepted-structure file name (r12_cell_3_4_181-362_rank_001.pdb)."""
    m = _NAME_RE.match(name)
    if not m:
        return None, '', ''
    return int(m.group('round')), m.group('cell') or '', m.group('depth').replace('-', ':')


def write_structures_xtc(output_dir, out_prefix='structures', align_selection=None, quiet=False):
    """
    Write all accepted structures in ``<output_dir>/structures/`` as ``<out_prefix>.pdb`` + ``.xtc``
    and a frame index CSV.  Returns the number of frames written.

    align_selection : MDAnalysis selection used to superpose every frame on the first one.
        Default: the run's PCA selection (pca_selection.txt, the DSSP core Cα), else all Cα.
    Structures whose atom count differs from the first are skipped (reported at the end).
    """
    structures_dir = os.path.join(output_dir, 'structures')
    plddt_path = os.path.join(output_dir, 'plddt_dict.json')
    quality = {}
    if os.path.exists(plddt_path):
        with open(plddt_path) as fh:
            quality = {os.path.basename(k): v for k, v in json.load(fh).items()}

    names = [n for n in os.listdir(structures_dir) if n.endswith('.pdb')]
    if not names:
        raise FileNotFoundError(f'no .pdb files in {structures_dir}')

    def sort_key(n):
        rnd, _, _ = _parse_name(n)
        return (rnd if rnd is not None else -1, n)
    names.sort(key=sort_key)

    if align_selection is None:
        sel_path = os.path.join(output_dir, 'pca_selection.txt')
        if os.path.exists(sel_path):
            with open(sel_path) as fh:
                align_selection = fh.read().strip() or None
    if align_selection is None:
        align_selection = 'name CA'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = mda.Universe(os.path.realpath(os.path.join(structures_dir, names[0])))
    n_atoms = ref.atoms.n_atoms
    if len(ref.select_atoms(align_selection)) == 0:
        align_selection = 'name CA'

    pdb_out = os.path.join(output_dir, f'{out_prefix}.pdb')
    xtc_out = os.path.join(output_dir, f'{out_prefix}.xtc')
    csv_out = os.path.join(output_dir, f'{out_prefix}_index.csv')
    ref.atoms.write(pdb_out)

    skipped = []
    n_frames = 0
    iterator = names
    if not quiet:
        try:
            from tqdm import tqdm
            iterator = tqdm(names, desc='  structures.xtc', unit='struct')
        except ImportError:
            pass
    with warnings.catch_warnings(), mda.Writer(xtc_out, n_atoms) as w, open(csv_out, 'w', newline='') as fh:
        warnings.simplefilter('ignore')
        out = csv.writer(fh)
        out.writerow(['frame', 'file', 'round', 'depth', 'cell', 'quality'])
        for name in iterator:
            path = os.path.realpath(os.path.join(structures_dir, name))
            try:
                u = mda.Universe(path)
            except Exception:
                skipped.append(name)
                continue
            if u.atoms.n_atoms != n_atoms:
                skipped.append(name)
                continue
            align.alignto(u, ref, select=align_selection, match_atoms=False)
            w.write(u.atoms)
            rnd, cell, depth = _parse_name(name)
            q = quality.get(name)
            out.writerow([n_frames, name, '' if rnd is None else rnd, depth, cell,
                          '' if q is None else round(q, 2)])
            n_frames += 1

    if not quiet:
        print(f'  structures.xtc  →  {xtc_out}  ({n_frames} frames, topology {os.path.basename(pdb_out)})')
        if skipped:
            print(f'  skipped {len(skipped)} structures with a different atom count or unreadable '
                  f'(e.g. {skipped[0]})')
    return n_frames


def main():
    p = argparse.ArgumentParser(
        description='Write all accepted structures of an ensemblr run as one aligned trajectory '
                    '(structures.pdb + structures.xtc + structures_index.csv).')
    p.add_argument('output_dir', help="the generator's output_dir (holds structures/ and plddt_dict.json)")
    p.add_argument('--align-selection', default=None,
                   help='MDAnalysis selection to superpose on (default: pca_selection.txt, else all CA)')
    p.add_argument('--prefix', default='structures', help='output file prefix (default: structures)')
    a = p.parse_args()
    write_structures_xtc(a.output_dir, out_prefix=a.prefix, align_selection=a.align_selection)


if __name__ == '__main__':
    main()
