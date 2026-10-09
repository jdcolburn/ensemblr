"""Run a chain of grompp/mdrun stages across a set of GROMACS window
directories, against one shared (or per-window) topology.

Factored out of SolventSystemPrep.minimise_systems() so the same
window-loop/grompp/mdrun machinery can also drive UmbrellaSamplingPrep /
GridUmbrellaSamplingPrep, which build every window against a single external
topology (e.g. a CHARMM-GUI template, or a constant-pH-retitrated one) rather
than a fresh pdb2gmx topology per window.
"""

import os
import subprocess

from tqdm import tqdm


def run_gmx(args, cwd, stdin=None, log=None, env=None):
    """Run a GROMACS command, capture output to log, raise on failure."""
    result = subprocess.run(
        args, cwd=cwd,
        input=stdin, capture_output=True, text=True,
        env=env,
    )
    if log:
        with open(os.path.join(cwd, log), 'w') as f:
            f.write(result.stdout + result.stderr)
    if result.returncode != 0:
        log_hint = os.path.join(cwd, log) if log else '(no log)'
        raise RuntimeError(
            f"gmx command failed in {cwd}:\n"
            f"  {' '.join(args)}\n"
            f"See {log_hint} for details.\n"
            f"Last stderr:\n{result.stderr[-800:]}"
        )


def run_staged_mdrun(window_dirs, stages, topol_path, index_path=None,
                     initial_structures='system.gro',
                     gmx='gmx', env=None, mdrun_env=None,
                     maxwarn=5, mdrun_flags=None,
                     continue_on_error=False, desc='Running stages'):
    """
    Run a chain of grompp + mdrun stages across a set of window directories.

    Each window is processed independently through every stage in order;
    stage N's output .gro (and .cpt, if one was written) automatically
    becomes stage N+1's -c/-r/-t.  The very first stage in each window uses
    that window's entry in `initial_structures` instead.

    Parameters
    ----------
    window_dirs : list of str
        One directory per window.
    stages : list of dict
        Each dict describes one stage: {'name': str, 'mdp': str,
        'grompp_flags': list of str (optional), 'mdrun_flags': list of str
        (optional, appended after the shared mdrun_flags for this stage
        only)}.  'mdp' is passed straight through as grompp's -f, so it may
        be a bare filename (resolved relative to each window_dir) or an
        absolute path to one shared .mdp.
    topol_path : str
        Passed straight through as grompp's -p for every window (cwd is
        the window directory).  A bare filename (e.g. 'topol.top') resolves
        per-window; an absolute path points every window at one shared
        topology.
    index_path : str, optional
        Same convention as topol_path, passed as grompp's -n.
    initial_structures : str or list of str
        Coordinate file for each window's first stage.  A single string is
        used for every window (resolved relative to that window_dir); a
        list must match window_dirs one-to-one, e.g. when each window's
        starting structure has a different filename (as in
        UmbrellaSamplingPrep's window_N.pdb).
    gmx : str
        gmx executable name or full path.
    env : dict, optional
        Environment for grompp calls (and mdrun, unless mdrun_env is given).
    mdrun_env : dict, optional
        Environment for mdrun calls specifically, e.g. to sync
        OMP_NUM_THREADS with an -ntomp mdrun flag.  Defaults to `env`.
    maxwarn : int
        Passed to every grompp call.
    mdrun_flags : list of str, optional
        Flags appended to every mdrun call, e.g. ['-nb', 'gpu', '-update', 'cpu'].
    continue_on_error : bool
        If False (default), a stage failure raises immediately and stops
        the whole run — matches SolventSystemPrep.minimise_systems()'s
        historical behaviour.  If True, the failure is logged, that window
        is left at whatever stage it last completed, and the loop moves on
        to the next window — matching the older prep_us_windows.sh
        convention of running every window and letting the user inspect
        failures afterward, rather than losing all progress on the first
        one that crashes.
    desc : str
        tqdm progress bar label.

    Returns
    -------
    dict
        {window_dir: {'ok': bool, 'failed_stage': str or None,
                       'error': str or None}}
    """
    if mdrun_env is None:
        mdrun_env = env
    if mdrun_flags is None:
        mdrun_flags = []

    if isinstance(initial_structures, str):
        initial_structures = [initial_structures] * len(window_dirs)
    if len(initial_structures) != len(window_dirs):
        raise ValueError('initial_structures must be a single filename or '
                         'a list matching window_dirs in length')

    results = {}
    failed = []

    for window_dir, init_struct in zip(tqdm(window_dirs, desc=desc), initial_structures):
        c_file, r_file, t_file = init_struct, init_struct, None
        window_ok, failed_stage, error_msg = True, None, None

        for stage in stages:
            name = stage['name']
            grompp_args = [gmx, 'grompp',
                          '-f', stage['mdp'],
                          '-c', c_file,
                          '-r', r_file,
                          '-p', topol_path,
                          '-o', f'{name}.tpr',
                          '-maxwarn', str(maxwarn)]
            if index_path:
                grompp_args += ['-n', index_path]
            if t_file and os.path.exists(os.path.join(window_dir, t_file)):
                grompp_args += ['-t', t_file]
            grompp_args += stage.get('grompp_flags', [])

            try:
                run_gmx(grompp_args, cwd=window_dir, log=f'grompp_{name}.log', env=env)
                run_gmx(
                    [gmx, 'mdrun', '-deffnm', name]
                    + mdrun_flags + stage.get('mdrun_flags', []),
                    cwd=window_dir, log=f'mdrun_{name}.log', env=mdrun_env,
                )
            except RuntimeError as e:
                window_ok, failed_stage, error_msg = False, name, str(e)
                if not continue_on_error:
                    raise
                break

            c_file = r_file = f'{name}.gro'
            t_file = f'{name}.cpt'

        results[window_dir] = {
            'ok': window_ok, 'failed_stage': failed_stage, 'error': error_msg,
        }
        if not window_ok:
            failed.append((window_dir, failed_stage))

    if failed:
        print(f'\n{len(failed)}/{len(window_dirs)} window(s) failed:')
        for wd, stage in failed:
            print(f'  {wd}: failed at stage "{stage}" '
                 f'— see grompp_{stage}.log / mdrun_{stage}.log')
    else:
        print(f'\nAll {len(window_dirs)} window(s) completed all '
             f'{len(stages)} stage(s).')

    return results
