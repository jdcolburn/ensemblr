# PLUMED input file generation and utilities

import os
import re
import subprocess

import numpy as np
import MDAnalysis as mda


def selection_parser(mda_selection, u_template):

    """
    Turns an MDA selection token into a safe list of atom IDs to pass to PLUMED
    because MOLINFO may or may not work depending on what python interpreter
    is available at runtime.

    Parameters
    ----------
    mda_selection : string, Required, default: None
        MDA selection token to parse.
    u_template : MDAnalysis universe object, Required, default: None

    Returns
    -------
    selection_string : string
        List of atomIDs matching the token.
    """

    selection_string = ''
    for i in u_template.select_atoms(mda_selection).atoms.ids:
        selection_string += str(i) + ','
    selection_string = selection_string[:-1]
    return selection_string


def plumed_input_writer(cv_window_values, output_file='plumed.dat', force_constant=1000, cv_reference_pdb=None, molinfo_pdb=None, scale_factor=1, COM_1=None, COM_2=None, multidir=False):

    """
    Write a plumed input file for use in umbrella sampling, with appropriate restraint values, etc. Assumes CV is a distance if no cv_reference_pdb is provided.

    Parameters
    ----------
    cv_window_values : list, Required, default: None
        list of restraint values in the CV space.
    output_file : string, Optional, default: 'plumed.dat'
        filename of output PLUMED file.
    force_constant : float, Optional, default: 1000
        Force consant for harmonic restraints in whatever units plumed uses by default kJ / nm ^ 2
    cv_reference_pdb : string, Optional, default: None
        PDB file used by PLUMED defining the linear subspace / PC projection
    molinfo_pdb : string, Optional, default: None
        PDB file used by PLUMED defining the molecule information
    COM_1 : string, Optional, default: None
        Selection token for the first center of mass for a distance CV, if this is passed, a distance CV is assumed
    COM_2 : string, Optional, default: None
        Selection token for the second center of mass for a distance CV
    multidir : bool, Optional, default: False
        Flag to indicate if this is for a multidir simulation
    scale_factor : float, Optional, default: 1
        PLUMED divides the value of the projection by the number of atoms in the CV, so this will make your PLUMED values consistent with the PC values here


    Returns
    -------
    None

    """

    # Flag to indicate when CV is a PC/subspace projection or anything else with arbitrary units
    # in COM_1 and COM_2 are provided, assume NO pca CV
    if COM_1 != None:
        pca_cv = False
    else:
        pca_cv = True
        #cv_window_values = cv_window_values # convert window values to nm (from Angstrom)

    # sanitise cv_reference_pdb if it is an absolute path
    if cv_reference_pdb != None:
        cv_reference_pdb_filename = os.path.basename(cv_reference_pdb)
    if molinfo_pdb != None:
        molinfo_pdb_filename = os.path.basename(molinfo_pdb)

    if multidir == True:
    # write a multidir plumed input
        with open(output_file, 'w') as f:
            f.write('UNITS LENGTH=A\n\n')
            f.write(f'MOLINFO STRUCTURE=../window_0/{molinfo_pdb_filename} \n\n')

            if pca_cv:
                f.write(f'pcproj: PCAVARS REFERENCE=../{cv_reference_pdb_filename} TYPE=OPTIMAL\n\n')
                f.write(f'CV: MATHEVAL ARG=pcproj.eig-1 FUNC=x*{scale_factor} PERIODIC=NO \n\n')
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} AT=@replicas:')
                print_arg = 'CV.*,umbrella_restraint.*'
            else:
                # make universe out of reference pdb
                u_template = mda.Universe(cv_reference_pdb)
                f.write(f'com1: CENTER ATOMS={selection_parser(COM_1, u_template)} \n')
                f.write(f'com2: CENTER ATOMS={selection_parser(COM_2, u_template)} \n\n')
                f.write('CV: DISTANCE ATOMS=com1,com2 \n\n')
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} AT=@replicas:')
                print_arg = 'CV,umbrella_restraint.*'

            f.write(','.join(map(lambda x: str(x.round(4)), cv_window_values)) + '\n')
            f.write(f'\nPRINT ARG={print_arg} FILE=COLVAR_MULTI\n\n')
    else:
    # serial plumed input
        with open(output_file, 'w') as f:
            f.write('UNITS LENGTH=A\n\n')
            f.write(f'MOLINFO STRUCTURE=../window_0/{molinfo_pdb_filename} \n\n')

            if pca_cv:
                f.write(f'pcproj: PCAVARS REFERENCE=../{cv_reference_pdb_filename} TYPE=OPTIMAL\n\n')
                f.write(f'CV: MATHEVAL ARG=pcproj.eig-1 FUNC=x*{scale_factor} PERIODIC=NO \n\n')
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} AT=')
                print_arg = 'CV.*,umbrella_restraint.*'
            else:
                # make universe out of reference pdb
                u_template = mda.Universe(cv_reference_pdb)
                f.write(f'com1: CENTER ATOMS={selection_parser(COM_1, u_template)} \n')
                f.write(f'com2: CENTER ATOMS={selection_parser(COM_2, u_template)} \n\n')
                f.write('CV: DISTANCE ATOMS=com1,com2 \n\n')
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} AT=')
                print_arg = 'CV,umbrella_restraint.*'

            # write the single cv value
            f.write(str(cv_window_values) + '\n')
            f.write(f'\nPRINT ARG={print_arg} FILE=COLVAR\n\n')


def plumed_input_writer_2d(cv1_window_values, cv2_window_values,
                            cv1_reference_pdb, cv2_reference_pdb,
                            molinfo_pdb, scale_factor=1,
                            force_constant=1000, multidir=False,
                            output_file='plumed.dat'):
    """
    Write a PLUMED input file for 2D umbrella sampling with two PCA CVs.

    Writes two PCAVARS projections (PC1 and PC2) and applies one harmonic
    restraint per CV per window.

    Parameters
    ----------
    cv1_window_values : float or array-like
        PC1 target value(s).  Pass an array when multidir=True.
    cv2_window_values : float or array-like
        PC2 target value(s).  Pass an array when multidir=True.
    cv1_reference_pdb : str
        Path to the PLUMED PCA reference PDB for PC1 (eigenvector in B-factors).
    cv2_reference_pdb : str
        Path to the PLUMED PCA reference PDB for PC2.
    molinfo_pdb : str
        Path to the PDB used for MOLINFO.
    scale_factor : int
        Number of PCA atoms — PLUMED divides the projection by this value,
        so multiplying by it restores the original PC units.
    force_constant : float
        Harmonic restraint force constant (kJ mol⁻¹ nm⁻²).
    multidir : bool
        If True, write @replicas: syntax for GROMACS multidir simulations.
    output_file : str

    Returns
    -------
    None
    """
    cv1_ref_name = os.path.basename(cv1_reference_pdb)
    cv2_ref_name = os.path.basename(cv2_reference_pdb)
    molinfo_name = os.path.basename(molinfo_pdb)

    with open(output_file, 'w') as f:
        f.write('UNITS LENGTH=A\n\n')
        f.write(f'MOLINFO STRUCTURE=../window_0/{molinfo_name}\n\n')
        f.write(f'pcproj1: PCAVARS REFERENCE=../{cv1_ref_name} TYPE=OPTIMAL\n')
        f.write(f'pcproj2: PCAVARS REFERENCE=../{cv2_ref_name} TYPE=OPTIMAL\n\n')
        f.write(f'CV1: MATHEVAL ARG=pcproj1.eig-1 FUNC=x*{scale_factor} PERIODIC=NO\n')
        f.write(f'CV2: MATHEVAL ARG=pcproj2.eig-1 FUNC=x*{scale_factor} PERIODIC=NO\n\n')

        if multidir:
            cv1_str = ','.join(f'{v:.4f}' for v in cv1_window_values)
            cv2_str = ','.join(f'{v:.4f}' for v in cv2_window_values)
            f.write(f'restraint_pc1: RESTRAINT ARG=CV1 KAPPA={force_constant} '
                    f'AT=@replicas:{cv1_str}\n')
            f.write(f'restraint_pc2: RESTRAINT ARG=CV2 KAPPA={force_constant} '
                    f'AT=@replicas:{cv2_str}\n\n')
            f.write('PRINT ARG=CV1,CV2,restraint_pc1.*,restraint_pc2.* FILE=COLVAR_MULTI\n')
        else:
            f.write(f'restraint_pc1: RESTRAINT ARG=CV1 KAPPA={force_constant} '
                    f'AT={cv1_window_values:.4f}\n')
            f.write(f'restraint_pc2: RESTRAINT ARG=CV2 KAPPA={force_constant} '
                    f'AT={cv2_window_values:.4f}\n\n')
            f.write('PRINT ARG=CV1,CV2,restraint_pc1.*,restraint_pc2.* FILE=COLVAR\n')


def run_plumed_driver(window_dir, plumed_exe, gro_name='system.gro', plumed_lib=None):
    """
    Run ``plumed driver`` on a single window and return the first-frame CV values.

    Converts the GRO trajectory to PDB before calling the driver — required as
    a workaround for a broken GRO molfile reader in some PLUMED 2.8 builds that
    causes a silent crash during initialisation.  Strips restraint lines and
    rewrites the PRINT ARG so the driver does not require AT= values.  Writes
    COLVAR_driver in window_dir.

    Parameters
    ----------
    window_dir : str
        Absolute path to the window directory.  Must contain ``plumed.dat``
        and the GRO file.
    plumed_exe : str
        Path to the ``plumed`` executable.
    gro_name : str
        Filename of the trajectory/structure GRO in window_dir.
    plumed_lib : str, optional
        Path to the PLUMED lib directory (e.g. ``.../plumed/2.8.1/lib``).
        Used to build a minimal ``LD_LIBRARY_PATH`` that avoids conda
        ``libstdc++`` conflicts with PLUMED's hardcoded RPATH.  If None,
        the full parent-process environment is inherited.

    Returns
    -------
    cv_values : tuple of float
        ``(cv1, cv2)`` read from the first data line of COLVAR_driver.
        Both are ``nan`` if the driver failed or the file was not written.
    error : str or None
        The relevant PLUMED error line(s) if the run failed, else None.
    """
    pdb_path    = os.path.join(window_dir, '_system_driver.pdb')
    test_dat    = os.path.join(window_dir, '_plumed_driver_test.dat')
    colvar_path = os.path.join(window_dir, 'COLVAR_driver')

    # convert GRO → PDB (PLUMED's GRO molfile reader crashes in some builds)
    u = mda.Universe(os.path.join(window_dir, gro_name))
    u.atoms.write(pdb_path)

    # strip restraint lines, fix PRINT ARG, redirect output file
    src = open(os.path.join(window_dir, 'plumed.dat')).read()
    lines = [l for l in src.splitlines() if not l.strip().startswith('restraint')]
    test_src = re.sub(r'(PRINT ARG=)\S+', r'\1CV1,CV2', '\n'.join(lines))
    test_src = test_src.replace('FILE=COLVAR', 'FILE=COLVAR_driver')
    with open(test_dat, 'w') as f:
        f.write(test_src)

    env = (
        {'PATH': os.environ.get('PATH', ''),
         'HOME': os.environ.get('HOME', ''),
         'LD_LIBRARY_PATH': plumed_lib}
        if plumed_lib else None
    )

    result = subprocess.run(
        [plumed_exe, 'driver',
         '--plumed', '_plumed_driver_test.dat',
         '--mf_pdb', '_system_driver.pdb'],
        cwd=window_dir, capture_output=True, text=True, env=env,
    )

    if result.returncode != 0:
        error_lines = [
            l for l in (result.stdout + result.stderr).splitlines()
            if 'error' in l.lower() or 'message follows' in l.lower()
        ]
        return (float('nan'), float('nan')), '\n'.join(error_lines) or result.stderr[:300]

    cv1, cv2 = float('nan'), float('nan')
    if os.path.exists(colvar_path):
        with open(colvar_path) as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        cv1, cv2 = float(parts[1]), float(parts[2])
                        break
    return (cv1, cv2), None


def verify_2d_pca_cvs(window_dirs, structures, cv1_col, cv2_col, ensemble_df,
                       plumed_exe, gro_name='system.gro', plumed_lib=None):
    """
    Run ``plumed driver`` on a list of windows and compare CV1/CV2 against
    MDA-computed projections stored in ``ensemble_df``.

    Parameters
    ----------
    window_dirs : list of str
        Absolute paths to window directories (one per active grid cell).
    structures : list of str
        Ensemble structure filename for each window (index into ensemble_df).
    cv1_col, cv2_col : str
        Column names in ensemble_df for the two CVs (e.g. 'PC1', 'PC2').
    ensemble_df : pd.DataFrame
        Ensemble dataframe indexed by structure filename.
    plumed_exe : str
        Path to the ``plumed`` executable.
    gro_name : str
        GRO filename inside each window directory.
    plumed_lib : str, optional
        PLUMED lib directory — passed to run_plumed_driver.

    Returns
    -------
    dict with keys:
        plumed_cv1, plumed_cv2 : np.ndarray  — values read by PLUMED
        mda_cv1, mda_cv2       : np.ndarray  — values from ensemble_df
        valid                  : np.ndarray bool mask (non-nan windows)
        errors                 : list of (window_dir, error_msg) for failures
    """
    from tqdm import tqdm

    plumed_cv1, plumed_cv2 = [], []
    mda_cv1 = [ensemble_df.loc[s, cv1_col] for s in structures]
    mda_cv2 = [ensemble_df.loc[s, cv2_col] for s in structures]
    errors  = []

    for wdir in tqdm(window_dirs, desc='PLUMED driver'):
        (cv1, cv2), err = run_plumed_driver(wdir, plumed_exe, gro_name, plumed_lib)
        plumed_cv1.append(cv1)
        plumed_cv2.append(cv2)
        if err:
            errors.append((wdir, err))

    plumed_cv1 = np.array(plumed_cv1)
    plumed_cv2 = np.array(plumed_cv2)
    mda_cv1    = np.array(mda_cv1)
    mda_cv2    = np.array(mda_cv2)
    valid      = ~(np.isnan(plumed_cv1) | np.isnan(plumed_cv2))

    if errors:
        print(f'  {len(errors)} window(s) failed:')
        for wdir, msg in errors:
            print(f'  {os.path.basename(wdir)}: {msg}')

    return dict(plumed_cv1=plumed_cv1, plumed_cv2=plumed_cv2,
                mda_cv1=mda_cv1, mda_cv2=mda_cv2,
                valid=valid, errors=errors)


def run_plumed_driver_1d(window_dir, plumed_exe, gro_name='system.gro', plumed_lib=None):
    """
    Run ``plumed driver`` on a single 1D umbrella sampling window.

    Like ``run_plumed_driver`` but for single-CV plumed.dat files where the
    restraint label is ``umbrella_restraint`` and the CV label is ``CV``.
    Strips any line containing the RESTRAINT action (regardless of label),
    and rewrites PRINT ARG to ``CV``.

    Returns
    -------
    cv : float
        CV value from the first data line of COLVAR_driver, or nan on failure.
    error : str or None
    """
    pdb_path    = os.path.join(window_dir, '_system_driver.pdb')
    test_dat    = os.path.join(window_dir, '_plumed_driver_test.dat')
    colvar_path = os.path.join(window_dir, 'COLVAR_driver')

    u = mda.Universe(os.path.join(window_dir, gro_name))
    u.atoms.write(pdb_path)

    src = open(os.path.join(window_dir, 'plumed.dat')).read()
    lines = [l for l in src.splitlines()
             if not re.search(r'\bRESTRAINT\b', l)]
    test_src = re.sub(r'(PRINT ARG=)\S+', r'\1CV', '\n'.join(lines))
    test_src = test_src.replace('FILE=COLVAR', 'FILE=COLVAR_driver')
    with open(test_dat, 'w') as f:
        f.write(test_src)

    env = (
        {'PATH': os.environ.get('PATH', ''),
         'HOME': os.environ.get('HOME', ''),
         'LD_LIBRARY_PATH': plumed_lib}
        if plumed_lib else None
    )

    result = subprocess.run(
        [plumed_exe, 'driver',
         '--plumed', '_plumed_driver_test.dat',
         '--mf_pdb', '_system_driver.pdb'],
        cwd=window_dir, capture_output=True, text=True, env=env,
    )

    if result.returncode != 0:
        error_lines = [
            l for l in (result.stdout + result.stderr).splitlines()
            if 'error' in l.lower() or 'message follows' in l.lower()
        ]
        return float('nan'), '\n'.join(error_lines) or result.stderr[:300]

    cv = float('nan')
    if os.path.exists(colvar_path):
        with open(colvar_path) as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cv = float(parts[1])
                        break
    return cv, None


def verify_1d_pca_cvs(window_dirs, structures, cv_col, ensemble_df,
                       plumed_exe, gro_name='system.gro', plumed_lib=None):
    """
    Run ``plumed driver`` on a list of 1D umbrella sampling windows and compare
    the CV values against MDA projections.

    Parameters
    ----------
    window_dirs : list of str
    structures : list of str
    cv_col : str
        Column name in ensemble_df for the CV (e.g. 'PC1').
    ensemble_df : pd.DataFrame
    plumed_exe : str
    gro_name : str
    plumed_lib : str, optional

    Returns
    -------
    dict with keys:
        plumed_cv : np.ndarray
        mda_cv    : np.ndarray
        valid     : np.ndarray bool mask
        errors    : list of (window_dir, error_msg)
    """
    from tqdm import tqdm

    plumed_cv = []
    mda_cv    = [ensemble_df.loc[s, cv_col] for s in structures]
    errors    = []

    for wdir in tqdm(window_dirs, desc='PLUMED driver'):
        cv, err = run_plumed_driver_1d(wdir, plumed_exe, gro_name, plumed_lib)
        plumed_cv.append(cv)
        if err:
            errors.append((wdir, err))

    plumed_cv = np.array(plumed_cv)
    mda_cv    = np.array(mda_cv)
    valid     = ~np.isnan(plumed_cv)

    if errors:
        print(f'  {len(errors)} window(s) failed:')
        for wdir, msg in errors:
            print(f'  {os.path.basename(wdir)}: {msg}')

    return dict(plumed_cv=plumed_cv, mda_cv=mda_cv, valid=valid, errors=errors)
