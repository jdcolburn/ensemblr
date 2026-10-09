import os
import requests


def fetch_uniprot_pdb_ids(uniprot_id):
    """
    Query the UniProt REST API for all PDB cross-references for a UniProt accession.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession, e.g. 'P12345'.

    Returns
    -------
    list of dict
        Each dict has keys: 'pdb_id' (str), 'chains' (list of str),
        'resolution' (float or None).
    """
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id.upper()}?format=json'
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    entries = []
    for xref in data.get('uniProtKBCrossReferences', []):
        if xref.get('database') != 'PDB':
            continue
        pdb_id = xref['id']
        chains = []
        resolution = None
        for prop in xref.get('properties', []):
            if prop['key'] == 'Chains':
                # e.g. "A/B=1-200" or "A=1-200"
                chains = [seg.split('=')[0].strip() for seg in prop['value'].split(',')]
                chains = [c for chain_part in chains for c in chain_part.split('/')]
            elif prop['key'] == 'Resolution':
                try:
                    resolution = float(prop['value'].replace(' A', '').strip())
                except ValueError:
                    pass
        entries.append({'pdb_id': pdb_id, 'chains': chains, 'resolution': resolution})

    return entries


def download_pdb(pdb_id, dest_path):
    """
    Download a PDB file from RCSB and write it to dest_path.

    Parameters
    ----------
    pdb_id : str
    dest_path : str
        Full path (including filename) to write the PDB file.
    """
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb'
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest_path, 'w') as f:
        f.write(resp.text)
