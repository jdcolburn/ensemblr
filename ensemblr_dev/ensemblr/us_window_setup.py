# set up umbrella sampling windows (for REUS)
from MDAnalysis.analysis import align, rms
import shutil
from difflib import SequenceMatcher

def write_pca_ref_pdb():

    # write a file like pca1.pdb but with "REMARK TYPE=OPTIMAL" after each ENDMDL line
    # edit this file to add "REMARK TYPE=OPTIMAL" after each ENDMDL line
    with open(umbrella_sampling_directory + 'pca_ref.pdb', 'r') as file :
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('ENDMDL', 'ENDMDL\nREMARK TYPE=OPTIMAL')
    # Write the file out again
    with open(umbrella_sampling_directory + 'pca_ref.pdb', 'w') as file:
        file.write(filedata)

# function to calculate the similarity between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# this should be a pdb from e.g. an unbiased simulation you've run before
# containing everything in the simualtion box, protein, lipids, water etc. 
# if you have a working topolgy for this system, you can re-use it here
template_pdb_for_umbrella_sampling = (base_directory + 'template.pdb')

# make a universe from this template
u_template = mda.Universe(template_pdb_for_umbrella_sampling, template_pdb_for_umbrella_sampling)

# sort out your collective variable, for later inclusion in the
# automatically generated plumed input file
print('collective variable is:', collective_variable, '\n')
if collective_variable == 'cyto_helix_bundle_separation':
    com1 = cyto_helix_bundle_1
    com2 = cyto_helix_bundle_2
    pcavar = False
elif collective_variable == 'lumen_helix_bundle_separation':
    com1 = lumen_helix_bundle_1
    com2 = lumen_helix_bundle_2
    pcavar = False
elif collective_variable == 'PC1':
    com1 = None
    com2 = None
    pcavar = True
    write_pca_ref_pdb()
    pca_ref_pdb = umbrella_sampling_directory + 'pca_ref.pdb'
else:
    print('I dont know what your collective variable is supposed to be, sort this out yourself (thinking emoji)')
    com1 = '{your atoms here}'
    com2 = '{your atoms here}'
    pcavar = False


# aligns mobile universe to first frame of reference universe
def align_universe(mobile, ref):
    alignment_selection = 'protein and name CA'
    mobile.trajectory[-1]  # set mobile trajectory to last frame
    ref.trajectory[0]  # set reference trajectory to first frame
    mobile_ca = mobile.select_atoms(alignment_selection)
    ref_ca = ref.select_atoms(alignment_selection)
    rms.rmsd(
             mobile_ca.positions, 
             ref_ca.positions, 
             superposition=False
             )
    aligner = align.AlignTraj(mobile, 
                              ref, 
                              select=alignment_selection,
                              in_memory=True).run()
    return mobile

# replace the protein coordinates in your template with those from a structure in the ensemble
def protein_coordinate_replacer(u_structure, u_template):

    # dictionary of name matches for charmm36 (and I assume amber)
    # key is the name in the template universe, value is the name in the structure universe
    name_mismatches = {
    'ALA HN'   : 'H',
    'ARG HB1'  : 'HB2',
    'ARG HB2'  : 'HB3',
    'ARG HD1'  : 'HD2',
    'ARG HD2'  : 'HD3',
    'ARG HG1'  : 'HG2',
    'ARG HG2'  : 'HG3',
    'ARG HN'   : 'H',
    'ASN HB1'  : 'HB2',
    'ASN HB2'  : 'HB3',
    'ASN HN'   : 'H',
    'ASP HB1'  : 'HB2',
    'ASP HB2'  : 'HB3',
    'ASP HN'   : 'H',
    'CYS HB1'  : 'HB2',
    'CYS HB2'  : 'HB3',
    'CYS HG1'  : 'HG',
    'CYS HN'   : 'H',
    'GLN HB1'  : 'HB2',
    'GLN HB2'  : 'HB3',
    'GLN HG1'  : 'HG2',
    'GLN HG2'  : 'HG3',
    'GLN HN'   : 'H',
    'GLU HB1'  : 'HB2',
    'GLU HB2'  : 'HB3',
    'GLU HG1'  : 'HG2',
    'GLU HG2'  : 'HG3',
    'GLU HN'   : 'H',
    'GLY HA1'  : 'HA2',
    'GLY HA2'  : 'HA3',
    'GLY HN'   : 'H',
    'HSD HB1'  : 'HB2',
    'HSD HB2'  : 'HB3',
    'HSD HN'   : 'H',
    'LEU HN'   : 'H',
    'LYS HB1'  : 'HB2',
    'LYS HB2'  : 'HB3',   
    'LYS HD1'  : 'HD2',
    'LYS HD2'  : 'HD3',
    'LYS HE1'  : 'HE2',
    'LYS HE2'  : 'HE3',
    'LYS HG1'  : 'HG2',
    'LYS HG2'  : 'HG3',
    'LYS HN'   : 'H',
    'MET HB1'  : 'HB2',
    'MET HB2'  : 'HB3',
    'MET HG1'  : 'HG2',
    'MET HG2'  : 'HG3',
    'MET HN'   : 'H',
    'PHE HB1'  : 'HB2',
    'PHE HB2'  : 'HB3',
    'PHE HN'   : 'H',
    'PRO HB1'  : 'HB2',
    'PRO HB2'  : 'HB3',
    'PRO HD1'  : 'HD2',
    'PRO HD2'  : 'HD3',
    'PRO HG1'  : 'HG2',
    'PRO HG2'  : 'HG3',
    'SER HB1'  : 'HB2',
    'SER HB2'  : 'HB3',
    'SER HG1'  : 'HG',
    'SER HN'   : 'H',
    'THR HN'   : 'H',
    'TRP HB1'  : 'HB2',
    'TRP HB2'  : 'HB3',
    'TRP HN'   : 'H',
    'TYR HB1'  : 'HB2',
    'TYR HB2'  : 'HB3',
    'TYR HN'   : 'H',
    'ILE CD'   : 'CD1',
    'ILE HD1'  : 'HD11',
    'ILE HD2'  : 'HD12',
    'ILE HD3'  : 'HD13', 
    'ILE HG11' : 'HG12',
    'ILE HG12' : 'HG13',
    'ILE HN'   : 'H',
    'LEU HB1' : 'HB2',
    'LEU HB1' : 'HB2',
    'VAL HN'  : 'H',
    }


    error_counter = 0
    error_type_list = []

    # renumber the residues in the structure universe
    u_structure.atoms.residues.resids += resid_offset

    # dict to keep track of which atoms in the structure universe have been accounted for (residue: atom)
    modified_atoms = []

    # loop though each atom in the template universe matching the atom name and residue ID to the atom in the template universe
    for atom in u_template.select_atoms('protein').atoms:

        atom_name_template = atom.name

        # if name is in the name_mismatches dictionary, search for the replacement name
        if (atom.resname + ' ' + atom.name) in name_mismatches.keys():
            #print('mismatched atom name: ', atom.name, 'in residue: ', atom.resname, 'replacing with: ', name_mismatches[(atom.resname + ' ' + atom.name)])
            atom_name_template = name_mismatches[(atom.resname + ' ' + atom.name)]

        # match atom with same name and residue ID in the structure universe (from AF ensemble)
        atom_name_structure = u_structure.atoms[(u_structure.atoms.names == atom_name_template) & (u_structure.atoms.resids == atom.resid) ].names

        # if the atom is found in the structure universe, replace the coordinates
        if len(atom_name_structure) > 0:

            # replace the coordinates of the atom in the template universe with the coordinates of the atom in the structure universe
            atom.position = u_structure.atoms[(u_structure.atoms.names == atom_name_template) & (u_structure.atoms.resids == atom.resid) ].positions

            # append the atom to a list of atoms
            modified_atoms.append((str(atom.resid) + str(atom.name)))

        else:
            print('ERROR: with: ', atom.name, atom.resname)

            # add this to a list of resname atom name pairs that are not found in the structure universe
            error_type_list.append((atom.resname, atom.name))

            error_counter += 1
            # handle mismatching atoms 

            # check that the residue itself is present
            if atom.resid in u_structure.atoms.resids:              # if the resid for this missing atom is in the structure universe

                pass
                #print('no atom in resid', atom.resid, atom.resname, 'with name: ', atom.name, 'found')
            
                #    # get the average position of the atoms in the structure universe with the same resid
                #    incomplete_resid_average_position_structure = np.mean(u_structure.atoms[u_structure.atoms.resids == atom.resid].positions, axis=0)
                #    # get the average position of the atoms in the template universe with the same resid
                #    incomplete_resid_average_position_template = np.mean(u_template.atoms[u_template.atoms.resids == atom.resid].positions, axis=0) 
                #    # get the vector between the two averages
                #    fudge_position_vector = incomplete_resid_average_position_structure - incomplete_resid_average_position_template
                #    # add the vector to the atom position in the template universe
                #    atom.position = atom.position + fudge_position_vector

            else:
                print('ERROR: resid not found in structure universe: ', atom.resid)

    print('number of atoms not found in structure universe: ', error_counter)
    # print unique error types in the list
    display(set(error_type_list))
    return u_template.atoms.positions # not important becauase the universe is modified in place, just return something

# turns your mda selection token into a safe list of atom IDs to pass to PLUMED
# because MOLINFO may or may not work depending on what python interpreter
# you have available at runtime
def selection_parser(mda_selection):
    selection_string = ''
    for i in u_template.select_atoms(mda_selection).atoms.ids:
        selection_string += str(i) + ','
    selection_string = selection_string[:-1]
    return selection_string

# write a plumed input file for REUS, if your CV is not a distance then
# you will need to modify this script appropriately
def plumed_input_writer(window_values, plumed_file, force_constant=1000):

    # convert window values to nm (from Angstrom)
    window_values = window_values / 10

    with open(plumed_file, 'w') as f:
        f.write('MOLINFO STRUCTURE=%s \n\n' % template_pdb_for_umbrella_sampling) #window_0/window_0.pdb
        if pcavar == True:
            f.write('CV: PCAVARS REFERENCE=%s TYPE=OPTIMAL\n\n' % pca_ref_pdb) #{@mda:{}}
        else:
            f.write('com1: CENTER ATOMS=%s \n' % selection_parser(com1)) #{@mda:{}}
            f.write('com2: CENTER ATOMS=%s \n' % selection_parser(com2)) #{@mda:{}}
            f.write('\n')
            f.write('CV: DISTANCE ATOMS=com1,com2 \n\n')
        f.write('umbrella_restraint: RESTRAINT ARG=CV KAPPA=%s AT=@replicas:' % force_constant)
        for i in window_values:
            # should be no comma after the last value
            if i == window_values[-1]:
                f.write(str(i.round(4)) + '\n')
            else:
                f.write(str(i.round(4)) + ',')
        f.write('\nPRINT ARG=CV,umbrella_restraint.* FILE=COLVAR_MULTI\n')
        f.write('\n')

# prepare directories for each window
umbrella_sampling_directory = base_directory + 'umbrella_sampling/'
window_directories = []
for i in range(mc_n_bins):
    window_directories.append(umbrella_sampling_directory + 'window_' + str(i))
    if not os.path.exists(window_directories[i]):
        os.makedirs(window_directories[i])
    shutil.copy(structure_directory + mc_runs_df.head(1)['path structures'].values[0][i], window_directories[i])    # copy the relevant structure in mc_runs_df to the window directory

# for each window, replace the coordinates of the protein in the template universe with the coordinates of the structure at the corresponding index in the best path
for i in range(mc_n_bins):

    # make a copy of the template universe
    u = u_template.copy()

    # make a universe out of the structure corresponding to the index in the best path
    structure = mc_runs_df.head(1)['path structures'].values[0][i]
    u_structure = mda.Universe(structure_directory + structure, structure_directory + structure)

    # align the structure to the template universe
    u_structure = align_universe(u_structure, u)
    new_coordinates = protein_coordinate_replacer(u_structure, u)

    # write out the universe to a pdb file
    u.atoms.write(window_directories[i] + '/window_' + str(i) + '.pdb')

# write out the plumed input file
plumed_input_writer(ideal_window_values, umbrella_sampling_directory + '/plumed.dat')