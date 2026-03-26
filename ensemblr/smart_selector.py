# Automatic "clever" selection definer

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP                   # for secondary structure selection

def generate_selection_token(reference_pdb_file, offset=0, conserved_residues=None, excluded_residues=None, explicitly_include=None):

    """
    Tries to generate a useful MDA selection token (for PCA etc.) based on some heuristics and user-speficied info.

    Uses DSSP to get secondary structure information from a pdb file. Resulting token only includes residues that are part of secondary structure elements. 
    
    Also, anything matching conserved_residues will have certain (CG/CZ) sidechain atoms included.

    Currently doesnt work if you dont pass both optional arguments

    WARNING: IT IS ASSUMED THAT CONSERVED, EXCLUDED RESIDUES ETC ARE PASSED WITH THE OFFSET ALREADY INCLUDED

    Parameters
    ----------
    reference_pdb_file : string, Required
        Filename of a PDB file.
    conserved_residues : string, Optional, default: None
        MDA selection token for residues that may have important sidechain atoms (these will be included in output token).
    excluded_residues : string, Optional, default: None
        MDA selection token for residues to be ignored (e.g. flexible loops). Overrides conserved_residues parameter.

    Returns
    -------
    rmsd_selection : string
        The resulting MDA selection token.
    """

    # defaults
    #if conserved_residues is None:
    #    conserved_residues = ''
    #if excluded_residues is None:
    #    excluded_residues = ''
       
    # identify regions of secondary structure
    p = PDBParser()
    structure = p.get_structure('reference', reference_pdb_file)
    model = structure[0]
    dssp = DSSP(model, reference_pdb_file, dssp='mkdssp')

    helices = []
    sheets  = []
    loops   = []

    # print the string of secondary structure labels
    #print(''.join([dssp[key][2] for key in dssp.keys()] ))

    # get secondary structure labels for resIDs
    for key in dssp.keys():
        if dssp[key][2] == 'H' or dssp[key][2] == 'G' or dssp[key][2] == 'I':
            helices.append(key[1][1])
        elif dssp[key][2] == 'E' or dssp[key][2] == 'B':
            sheets.append(key[1][1])
        elif dssp[key][2] == 'T' or dssp[key][2] == 'S':
            loops.append(key[1][1])

    helices = sorted(list(set(helices)))
    sheets = sorted(list(set(sheets)))
    loops = sorted(list(set(loops)))

    helices_contiguous = []
    sheets_contiguous  = []
    loops_contiguous   = []

    # get contiguous regions of secondary structure
    for i in range(len(helices)):
        if i == 0:
            helices_contiguous.append([helices[i]])
        elif helices[i] == helices[i-1] + 1:
            helices_contiguous[-1].append(helices[i])
        else:
            helices_contiguous.append([helices[i]])

    for i in range(len(sheets)):
        if i == 0:
            sheets_contiguous.append([sheets[i]])
        elif sheets[i] == sheets[i-1] + 1:
            sheets_contiguous[-1].append(sheets[i])
        else:
            sheets_contiguous.append([sheets[i]])

    for i in range(len(loops)):
        if i == 0:
            loops_contiguous.append([loops[i]])
        elif loops[i] == loops[i-1] + 1:
            loops_contiguous[-1].append(loops[i])
        else:
            loops_contiguous.append([loops[i]])

    selection_helices = []
    selection_sheets  = []
    selection_loops   = []

    # make mdanalysis selections corresponding to these regions
    for i in range(len(helices_contiguous)):
        selection_helices.append('(resid %s-%s)' % (helices_contiguous[i][0], helices_contiguous[i][-1]))
    for i in range(len(sheets_contiguous)):
        selection_sheets.append('(resid %s-%s)' % (sheets_contiguous[i][0], sheets_contiguous[i][-1]))
    for i in range(len(loops_contiguous)):
        selection_loops.append('(resid %s-%s)' % (loops_contiguous[i][0], loops_contiguous[i][-1]))

    # conserved residues
    if conserved_residues != None:
        selection_conserved_residues = '((' + conserved_residues + ') and (name CA or name CG or name CZ* or name NZ))' 

    # format selections
    selection_helices = ','.join(selection_helices)
    selection_helices = selection_helices.replace(',', ' or ')
    selection_sheets = ','.join(selection_sheets)
    selection_sheets = selection_sheets.replace(',', ' or ')
    selection_loops = ','.join(selection_loops)
    selection_loops = selection_loops.replace(',', ' or ')

    ### deprecated block of code to use information for a difference matrix 
    ### to filter for residues that differ more between two target structures
    
        #endstates = {}
        #
        #resids_from_ca_dist_diffmat = []
        #
        ## get the resid of the residues that differ by more than the threshold in absolute terms
        #above_thresh = np.where(abs(ca_dist_difference_matrix) >= diffmat_thresh)
        #for i in range(len(above_thresh[0])):
        #    resids_from_ca_dist_diffmat.append(u.atoms[above_thresh[0][i]].resid)
        #resids_from_ca_dist_diffmat = list(set(resids_from_ca_dist_diffmat))    # get unique residues
        #
        ## make a selection token for these residues
        #selection_from_ca_dist_diffmat = []
        #for i in range(len(resids_from_ca_dist_diffmat)):
        #    selection_from_ca_dist_diffmat.append('resid %s' % resids_from_ca_dist_diffmat[i])
        #selection_from_ca_dist_diffmat = ','.join(selection_from_ca_dist_diffmat)
        #selection_from_ca_dist_diffmat = selection_from_ca_dist_diffmat.replace(',', ' or ')

    # failsafe for valid token if nothing matching helices loops or sheets

    # final rmsd_selection for analysis
    rmsd_selection = '(  ( (' + selection_helices + ') and name CA ) or ( (' + selection_loops + ') and name CA )  or ( (' + selection_sheets + ') and name CA ) )'
    if conserved_residues != None and conserved_residues != '':
        rmsd_selection += ' or ( (' + selection_helices + ') and' + selection_conserved_residues + ')'
    if excluded_residues != None and excluded_residues != '':
        rmsd_selection += ' and not (' + excluded_residues + ')'
    if explicitly_include != None and explicitly_include != '':
        rmsd_selection += ' or ((' + explicitly_include + ') and name CA )'

    return(rmsd_selection)