# Structure preparation, coordinate manipulation, and geometry fixes

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.distances import distance_array


def align_universe(mobile, ref, selection='protein and name CA'):

    """
    aligns mobile universe to the first frame of the reference universe

    Parameters
    ----------
    mobile : MDAnalysis universe object, Required, default: None
        MDA universe to superimpose onto ref.
    ref : MDAnalysis universe object, Required, default: None
        Reference MDA universe

    Returns
    -------
    mobile : MDAnalysis universe object
        Modified MDA universe. Not strictly required since the universe object is modified in-place.
    """

    #alignment_selection = 'protein and name CA'
    mobile.trajectory[-1]  # set mobile trajectory to last frame
    ref.trajectory[0]  # set reference trajectory to first frame
    mobile_ca = mobile.select_atoms(selection)
    ref_ca = ref.select_atoms(selection)
    rms.rmsd(
             mobile_ca.positions,
             ref_ca.positions,
             superposition=False,
             )
    aligner = align.AlignTraj(mobile,
                              ref,
                              select=selection,
                              in_memory=True).run()
    return mobile

def protein_coordinate_replacer(u_structure, u_template, resid_offset=0, selection_token='protein'):

    """
    Hacky way to replace protein coordinates in a template with those from another pdb structure (e.g. a structure from the AF ensemble).
    These must match, if there are mismatchign atoms some coordinates will not be updated. Also clashes will likely be introduced.

    Modifies the template universe in place.

    Parameters
    ----------
    u_structure : MDAnalysis universe object, Required, default: None
        Universe containing a structure to "insert" into the template universe.
    u_template : MDAnalysis universe object, Required, default: None
        Template universe in which to substitute the coordinates from u_structure.

    Returns
    -------
    u_template.atoms.positions : list
        Not important becauase the universe is modified in-place.
    """

    # dictionary of name matches for charmm36 (and I assume amber)
    # key is the name in the template universe, value is the name in the structure universe
    # these are used to infer which atoms are which when there are differences introduced by
    # post-processing/embedding.
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
    for atom in u_template.select_atoms(selection_token).atoms:

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
            #print('ERROR: with: ', atom.name, atom.resname)

            # add this to a list of resname atom name pairs that are not found in the structure universe
            error_type_list.append((atom.resname, atom.name))

            error_counter += 1
            # handle mismatching atoms

            # check that the residue itself is present
            if atom.resid in u_structure.atoms.resids:              # if the resid for this missing atom is in the structure universe

                pass
                #print('no atom in resid', atom.resid, atom.resname, 'with name: ', atom.name, 'found')

                # some sort of heuristic to guess mismatching atom positions goes here...

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

    #print('number of atoms not found in structure universe: ', error_counter)
    # print unique error types in the list
    #display(set(error_type_list))
    return u_template.atoms.positions # not important becauase the universe is modified in place, just return something

def write_pca_ref_pdb(u_structure, u_template, eigenvector, pca_selection_token, offset=0, output_file='pca_ref.pdb'):#pca_reference='pca1.pdb', output_file='pca_ref.pdb'):

    """
    Wries a template PDB file for use in the PLUMED script.

    Parameters
    ----------
    Some

    Returns
    -------
    None

    """

    # fix offset so the selection token works properly
    u_template.atoms.residues.resids -= offset


    list_of_atom_ids_in_template = []

    # loop though atoms matchign the selection in the template universe and get their ids (not their indices) and make these into a list
    for atom in u_template.select_atoms(pca_selection_token).atoms:
        list_of_atom_ids_in_template.append(atom.id)

    # print the length of the list
    print('Number of atoms in the template universe: ', len(list_of_atom_ids_in_template))
    #  check that this matches the length of the selection in the structure (ensemble) universe
    print('Number of atoms in the ensemble universe: ', len(u_structure.select_atoms(pca_selection_token).atoms))

    # if these dont match print an error
    if len(list_of_atom_ids_in_template) != len(u_structure.select_atoms(pca_selection_token).atoms):
        print('ERROR: Number of atoms in the template universe does not match the number of atoms in the ensemble universe\n so theres no guarantee the IDs correspond to the same atoms\n')

    # write out the selection matching pca_selection_token while PRESERVING ATOM IDS from the template (i.e. not renumbered)
    # this is important because the atom IDs in the PCA reference file must match those in the template

    pca_selection = u_structure.select_atoms(pca_selection_token)

    with open(output_file, 'w') as f:
        i = 0
        for atom in pca_selection.atoms:
            f.write('ATOM  %5d %4s %3s %5d    %8.3f%8.3f%8.3f  1.00  1.00           %2s\n' % (list_of_atom_ids_in_template[i], atom.name, atom.resname, atom.resid+offset, atom.position[0], atom.position[1], atom.position[2], atom.element))
            i += 1
        f.write('END\nREMARK TYPE=DIRECTION\n')

    # now add the eigenvector to the file
    with open(output_file, 'a') as f:
        i = 0
        for atom in pca_selection.atoms:
            f.write('ATOM  %5d %4s %3s %5d    %8.3f%8.3f%8.3f  1.00  0.00           %2s\n' % (list_of_atom_ids_in_template[i], atom.name, atom.resname, atom.resid+offset, eigenvector[i,0], eigenvector[i,1], eigenvector[i,2], atom.element))
            i += 1
        f.write('END\n')

    # write a multistate pdb file interpolating along the eigenvector to visualise the direction
    with mda.Writer(output_file.replace('.pdb', '_interpolated.pdb'), pca_selection.atoms.n_atoms) as W:
        for i in range(-10, 10):
            pca_selection.atoms.positions = pca_selection.atoms.positions + i * eigenvector
            W.write(pca_selection.atoms)


def fix_overlapping_atoms(universe, selection_token='protein or (around 10 protein)', tolerance=0.001, step=0.05):

    """
    Fixes overlapping atoms using little nudges, docs tbd

    Parameters
    ----------
    Some

    Returns
    -------
    None

    """



    # make a distance matrix for all atom pairs in the selection_token
    # if any distances are less than the tolerance, move the atoms apart

    max_iterations = 20
    num_too_close = 1
    iteration_count = 0

    too_close = 1
    while num_too_close > 0 and iteration_count < max_iterations:
        iteration_count += 1

        selection = universe.select_atoms(selection_token)
        distances = distance_array(selection.positions, selection.positions)
        np.fill_diagonal(distances, 10*tolerance) # so atoms cant "overlap themselves"

        # get the indices of the atoms that are too close
        too_close = np.where(distances < tolerance)

        num_too_close =  len(too_close[0])

        # print the number of atoms that are too close
        #print('number of atoms within', tolerance, ': ', num_too_close, 'iteration: ', iteration_count)

        # for each pair that are too close, move them apart along the vector between them
        for i in range(len(too_close[0])):
            atom1 = selection[too_close[0][i]]
            atom2 = selection[too_close[1][i]]
            # random vector for translation
            vector = np.random.rand(3)

            # move the atoms apart by step along the vector
            atom1.position += vector * step
            atom2.position -= vector * step

    if iteration_count == max_iterations:
        print('Max iterations reached, some atoms still <', tolerance, ' A apart')
    #else:
    #    print('Done fixing overlapping atoms')

    return universe.atoms.positions

def fix_long_bonds(universe, thresh=10.0, fudge_dist=1):

    # for each residue in the selection, distance matrix between all atoms (as proxy for bond lengths in absence of bond information)
    # if any are greater than thresh, move the atoms apart

    # loop through each residue in the selection
    selection = universe.select_atoms('protein')

    # loop through residues
    for residue in selection.residues:
        # get the distance matrix for the atoms in the residue
        #distances = distance_array(residue.atoms.positions, residue.atoms.positions)
        # get the distance matrix for all atoms and the CA atom
        distances = distance_array(residue.atoms.positions, residue.atoms.select_atoms('name C').positions)
        # get the indices of the atoms that are too close
        too_far = np.where(distances > thresh)

        # if there are any atoms that are too close
        if len(too_far[0]) > 0:
            #print('number of atoms greater than', thresh, ': ', len(too_far[0]))

            # for each atom that is too far from the CA atom, move it closer
            for i in range(len(too_far[0])):
                atom1 = residue.atoms[too_far[0][i]]
                atom2 = residue.atoms[too_far[1][i]]
                # get the vector between the two atoms
                vector = atom1.position - atom2.position
                # normalise the vector
                vector = vector / np.linalg.norm(vector)
                # translate atom 1 such that it is now at a distance of thresh from atom 2
                atom1.position = atom2.position + vector * fudge_dist

            # fix overlapping atoms
            #fix_overlapping_atoms(universe, selection_token='resid %s' % residue.resid, tolerance=0.4, step=0.1)

            distances = distance_array(residue.atoms.positions, residue.atoms.select_atoms('name C').positions)
            too_far = np.where(distances > thresh)
            #print('NEW number of atoms greater than', thresh, ': ', len(too_far[0]))

    #print('Done fixing long bonds')
    return universe.atoms.positions

def fix_overlapping_with_protein(universe, dist_interval=1.0):
    """
    find lipids within the protein "cylinder" and translate them out.

    Parameters
    ----------
    universe : MDAnalysis universe object, Required, default: None
        the universe to modify in memory.
    dist_interval : float, Optional, default: 1.0
        distance by which to translate the lipids.

    Returns
    -------
    None

    """

    # find lipids within the protein "cylinder" and translate them out
    # select cylindrical zone around protein

    # get vector between COM of protein adn COM of lipids in the cylinder
    # translate lipids in cylinder by this vector

    # get cellextent in z
    #cell_extent_z = universe.dimensions[2]

    # get the min and max z values of everything in the universe
    min_z = np.min(universe.atoms.positions[:,2])
    max_z = np.max(universe.atoms.positions[:,2])
    # work out the dimensions of the box from this
    box_z = max_z - min_z

    # get protein COM in z
    cell_extent_z = box_z

    protein = universe.select_atoms('protein')

    protein_com_z = protein.atoms.center_of_geometry(pbc=False)[2]
    # get the z distance between the protein COM and the top of the box
    protein_top_z = cell_extent_z - protein_com_z
    # get the z distance between the protein COM and the bottom of the box
    protein_bottom_z = -protein_com_z
    # store temporary protein coordinates
    protein_coords = protein.atoms.positions
    # set hte z coordinate of the protein to zero
    protein_coords[:,2] = 0
    # make a distance matrix of the protein
    from MDAnalysis.analysis import distances
    protein_distmat = distances.distance_array(protein_coords, protein_coords, box=universe.dimensions)
    # get the mean distance between protein atoms
    protein_radius = np.mean(protein_distmat[np.triu_indices_from(protein_distmat, k=1)])/2

    # get the largest distance between protein atoms
    protein_largest_diameter = np.max(protein_distmat)/2

    # cylinder
    #print(protein_radius*2, protein_top_z, protein_bottom_z)

    lipids_in_cylinder_token = "((byres name P and (cyzone %s %s %s protein)))" %(protein_radius*2.01, protein_top_z, protein_bottom_z)

    # while there are still lipids in the cylinder
    selection = universe.select_atoms(lipids_in_cylinder_token)
    while len(selection) > 0:

        # print out how many still in the cylinder
        #print(f"lipids in cylinder: {len(selection)}")

        # update the selection
        selection = universe.select_atoms(lipids_in_cylinder_token)

        for lipid in selection.residues:

                # get the vector between the COM of the protein and the COM of the lipid
                vector = lipid.atoms.center_of_geometry(pbc=False) - protein.atoms.center_of_geometry(pbc=False)
                # set the third component to zero (z axis)
                vector[2] = 0

                # normalise the vector
                vector = vector / np.linalg.norm(vector)

                # move the lipid away from the protein along this vector
                lipid.atoms.translate(dist_interval*vector)

    #end_time = time.time()
    #print(f"done pushing lipids")

    return universe.atoms.positions

def prevent_threaded_lipids(universe, thresh=3.0, dist_interval=1.0, selection_token='resname DLPC'):

    protein = universe.select_atoms('protein')

    # all lipids within 2 A of aromatic ring atoms
    lipids = universe.select_atoms('byres (%s) and around %s ((resname PHE or resname TYR) and (name CG CD1 CD2 CE1 CE2 CZ))' % (selection_token, thresh))
    # if this is not empty, move the lipids away from the aromatic ring and print a message

    while len(lipids) > 0:
        #print("moving suspect lipid %s" % lipids.residues.resids)
        #print("number of possible threaded lipids: ", len(lipids))
        lipids = universe.select_atoms('byres (%s) and around %s ((resname PHE or resname TYR) and (name CG CD1 CD2 CE1 CE2 CZ))' % (selection_token, thresh))

        # move the lpid
        # for each syspect lipid, get the vector between this and the com of hte prtoein
        for lipid in lipids.residues:
            vector = lipid.atoms.center_of_geometry(pbc=False) - protein.atoms.center_of_geometry(pbc=False)
            vector[2] = 0
            vector = vector / np.linalg.norm(vector)
            lipid.atoms.translate(dist_interval*vector)

    #print("Done fixing threaded lipids")
    return universe.atoms.positions
