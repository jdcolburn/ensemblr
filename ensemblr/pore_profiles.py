"""
pore_profiles.py
----------------
HOLE pore-radius analysis across an AF2 structural ensemble.

Pipeline per structure:
  1. Align to the membrane plane with memembed (cached)
  2. Run HOLE along the Z-axis
  3. Parse output → min_pore_radius, bottleneck_z added to ensemble df
  4. Optionally save full (z, radius) profiles to a pickle file

Usage::

    from ensemblr import PoreProfiler

    profiler = PoreProfiler(
        ens,
        memembed_executable='/path/to/memembed',
        hole_executable='hole',        # must be on PATH after `conda install hole2`
    )
    profiler.run(
        out_dir = run_dir + 'pore_profiles/',
        num_processes = 6,
    )
    # profiler.profiles  → dict {struct_name: ndarray shape (N, 2) of (z, radius)}
    # ens.df now has 'min_pore_radius' and 'bottleneck_z' columns
"""

import os
import re
import pickle
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


# ── HOLE input template ──────────────────────────────────────────────────────

_HOLE_INPUT = """\
COORD   {coord}
CVECT   {cvx:.4f} {cvy:.4f} {cvz:.4f}
CPOINT  {cpx:.4f} {cpy:.4f} {cpz:.4f}
ENDRAD  {endrad:.1f}
RADIUS  {radius_file}
SPHPDB  {sphpdb}
SAMPLE  {sample:.2f}
IGNORE  HOH
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_radius_file(hole_exe: str) -> str:
    """Locate simple.rad near the HOLE binary (works for conda-installed hole2)."""
    hole_path = shutil.which(hole_exe) or hole_exe
    if hole_path is None:
        raise FileNotFoundError(f"Cannot locate HOLE executable '{hole_exe}'.")
    bin_dir = os.path.dirname(os.path.abspath(hole_path))
    candidates = [
        os.path.normpath(os.path.join(bin_dir, rel))
        for rel in [
            '../share/hole2/rad/simple.rad',
            '../../share/hole2/rad/simple.rad',
            '../lib/hole2/rad/simple.rad',
            '../dat/simple.rad',
            '../rad/simple.rad',
        ]
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    env_rad = os.environ.get('HOLE_RADIUS', '')
    if env_rad and os.path.exists(env_rad):
        return env_rad
    raise FileNotFoundError(
        "Cannot find HOLE radius file (simple.rad). "
        "Pass radius_file= explicitly or set the HOLE_RADIUS environment variable.\n"
        f"  Searched near: {bin_dir}"
    )


def _parse_hole_output(stdout: str):
    """Return (min_radius, bottleneck_z, profile) from HOLE stdout.

    profile is ndarray shape (N, 2) of (z, radius), or None if not found.
    """
    min_r = None
    bz = None
    profile = []
    in_table = False

    for line in stdout.splitlines():
        # Summary line: "Minimum radius found:  2.345 Angstrom at position  0.400"
        m = re.search(
            r'[Mm]inimum radius[^:]*:\s*([\d.]+)\s+[Aa]ngstrom.*?position\s*([-\d.]+)',
            line,
        )
        if m:
            min_r = float(m.group(1))
            bz = float(m.group(2))
            continue

        # Table header
        if 'Mid-point' in line and 'Radius' in line:
            in_table = True
            continue
        if in_table and re.match(r'\s*-{4,}', line):
            continue
        if in_table:
            m2 = re.match(r'\s*(-?[\d.]+)\s+([\d.]+)\s+\d', line)
            if m2:
                profile.append((float(m2.group(1)), float(m2.group(2))))
            elif profile and line.strip() == '':
                in_table = False

    profile_arr = np.array(profile, dtype=float) if profile else None

    # Fallback: derive scalars from profile table if summary line absent
    if (min_r is None) and (profile_arr is not None) and len(profile_arr) > 0:
        idx = int(np.argmin(profile_arr[:, 1]))
        min_r = float(profile_arr[idx, 1])
        bz = float(profile_arr[idx, 0])

    return min_r, bz, profile_arr


# ── Per-structure worker (module-level for picklability) ─────────────────────

def _worker(args):
    (struct_name, pdb_path, aligned_dir,
     hole_exe, memembed_exe, radius_file,
     endrad, sample, cvect, save_profile) = args

    # ── memembed alignment ────────────────────────────────────────────────
    aligned_pdb = os.path.join(aligned_dir, struct_name)
    if not os.path.exists(aligned_pdb):
        proc = subprocess.run(
            [memembed_exe, '-x', '-o', aligned_pdb, pdb_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not os.path.exists(aligned_pdb):
            return struct_name, None, None, None

    # ── determine HOLE starting point: protein COM ─────────────────────
    try:
        import MDAnalysis as mda
        u = mda.Universe(aligned_pdb)
        com = u.select_atoms('protein').center_of_geometry()
        cpoint = (float(com[0]), float(com[1]), float(com[2]))
    except Exception:
        return struct_name, None, None, None

    # ── run HOLE ──────────────────────────────────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sph_file = os.path.join(tmpdir, 'hole.sph')
            hole_input = _HOLE_INPUT.format(
                coord=aligned_pdb,
                cvx=cvect[0], cvy=cvect[1], cvz=cvect[2],
                cpx=cpoint[0], cpy=cpoint[1], cpz=cpoint[2],
                endrad=endrad,
                radius_file=radius_file,
                sphpdb=sph_file,
                sample=sample,
            )
            proc = subprocess.run(
                [hole_exe],
                input=hole_input,
                capture_output=True, text=True,
                cwd=tmpdir,
            )
            if proc.returncode != 0:
                return struct_name, None, None, None

            min_r, bz, profile = _parse_hole_output(proc.stdout)
    except Exception:
        return struct_name, None, None, None

    if min_r is None:
        return struct_name, None, None, None

    return struct_name, min_r, bz, (profile if save_profile else None)


# ── PoreProfiler class ───────────────────────────────────────────────────────

class PoreProfiler:
    """Run HOLE pore-radius analysis on every structure in an Ensemble.

    Parameters
    ----------
    ensemble : Ensemble
        The parent Ensemble object (needs structure_directory set).
    hole_executable : str
        Name or path of the HOLE binary.  Install with::

            conda install -c conda-forge hole2

    memembed_executable : str
        Path to the memembed binary used to align structures to the membrane
        plane before running HOLE.
    radius_file : str or None
        Path to a HOLE VdW radius file (simple.rad or amber.rad).
        If None, the file is auto-located relative to the HOLE binary.
    """

    def __init__(
        self,
        ensemble,
        hole_executable: str = 'hole',
        memembed_executable: str = '/biggin/b212/bioc1781/software/memembed/bin/memembed',
        radius_file: str | None = None,
    ):
        self.ens = ensemble
        self.hole_exe = hole_executable
        self.memembed_exe = memembed_executable
        self.radius_file = radius_file
        self.profiles: dict[str, np.ndarray] = {}

    # ── public API ────────────────────────────────────────────────────────

    def run(
        self,
        out_dir: str,
        num_processes: int = 1,
        endrad: float = 22.0,
        sample: float = 0.2,
        cvect: tuple = (0.0, 0.0, 1.0),
        save_profiles: bool = True,
        checkpoint: bool = True,
    ):
        """Align structures, run HOLE, and add pore metrics to ensemble df.

        Parameters
        ----------
        out_dir : str
            Output directory.  memembed-aligned PDBs go into ``out_dir/memembed_aligned/``.
            Full profiles are saved to ``out_dir/pore_profiles.pkl`` when save_profiles=True.
        num_processes : int
            Worker processes for parallel execution.
        endrad : float
            HOLE ENDRAD parameter (Å) — stop when pore radius exceeds this value.
        sample : float
            HOLE SAMPLE parameter (Å) — sampling interval along the pore axis.
        cvect : tuple of float
            Pore axis direction vector.  Default (0, 0, 1) assumes structures are
            pre-aligned to the membrane normal via memembed.
        save_profiles : bool
            If True, save full (z, radius) arrays to a pickle file.
        checkpoint : bool
            If True, skip structures whose results are already in the pickle file.
        """
        radius_file = self.radius_file or _find_radius_file(self.hole_exe)

        aligned_dir = os.path.join(out_dir, 'memembed_aligned')
        os.makedirs(aligned_dir, exist_ok=True)

        checkpoint_file = os.path.join(out_dir, 'pore_profiles.pkl')
        done: dict = {}
        if checkpoint and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as fh:
                done = pickle.load(fh)
            print(f"  Loaded {len(done)} cached results from {checkpoint_file}")

        structures = list(self.ens.df.index)
        todo = [s for s in structures if s not in done]

        if todo:
            args_list = [
                (
                    s,
                    self.ens.structure_directory + s,
                    aligned_dir,
                    self.hole_exe,
                    self.memembed_exe,
                    radius_file,
                    endrad,
                    sample,
                    tuple(cvect),
                    save_profiles,
                )
                for s in todo
            ]

            if num_processes > 1:
                with Pool(num_processes) as pool:
                    results = list(
                        tqdm(pool.imap(_worker, args_list),
                             total=len(args_list), desc='HOLE profiles')
                    )
            else:
                results = list(
                    tqdm((_worker(a) for a in args_list),
                         total=len(args_list), desc='HOLE profiles')
                )

            failed = []
            for name, min_r, bz, profile in results:
                if min_r is None:
                    failed.append(name)
                else:
                    done[name] = {
                        'min_pore_radius': min_r,
                        'bottleneck_z': bz,
                        'profile': profile,
                    }

            if failed:
                print(f"  Warning: {len(failed)} structure(s) failed — check memembed/HOLE output.")

            if save_profiles:
                with open(checkpoint_file, 'wb') as fh:
                    pickle.dump(done, fh)
                print(f"  Saved profiles → {checkpoint_file}")

        # ── populate ens.df ───────────────────────────────────────────────
        self.profiles = {
            k: v['profile']
            for k, v in done.items()
            if v.get('profile') is not None
        }
        for name, data in done.items():
            if name in self.ens.df.index:
                self.ens.df.loc[name, 'min_pore_radius'] = data['min_pore_radius']
                self.ens.df.loc[name, 'bottleneck_z'] = data['bottleneck_z']

        n_ok = int((~self.ens.df.get('min_pore_radius', pd_series_nan(structures)).isna()).sum())
        print(f"  Pore profiles complete: {n_ok}/{len(structures)}")
        return self

    def mean_profile(self, z_min: float = -40.0, z_max: float = 40.0,
                     n_points: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate all profiles onto a common Z-grid and return mean ± std.

        Returns
        -------
        z_grid : ndarray shape (n_points,)
        mean_r : ndarray shape (n_points,)
        std_r  : ndarray shape (n_points,)
        """
        if not self.profiles:
            raise RuntimeError("No profiles loaded — call run() first.")
        z_grid = np.linspace(z_min, z_max, n_points)
        stacked = []
        for profile in self.profiles.values():
            if profile is None or len(profile) < 2:
                continue
            r_interp = np.interp(z_grid, profile[:, 0], profile[:, 1],
                                 left=np.nan, right=np.nan)
            stacked.append(r_interp)
        arr = np.array(stacked)
        return z_grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ── small helper (avoids pandas import at module level) ─────────────────────

def pd_series_nan(index):
    import pandas as pd
    return pd.Series(np.nan, index=index)
