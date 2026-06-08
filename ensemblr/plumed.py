# PLUMED input file generation and utilities

import os
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
