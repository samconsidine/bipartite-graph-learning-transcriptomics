from Bio.KEGG import REST
import numpy as np
from pathlib import Path


######################################
# Utilities to retrieve KEGG patways #
######################################

def list_KEGG_mouse_pathways():
    lines = REST.kegg_list('pathway', 'mmu').readlines()
    symbols = np.array([s.split('\t')[0].split(':')[-1] for s in lines])
    description = np.array([s.split('\t')[1].rstrip() for s in lines])
    return symbols, description


def get_pathway_info(pathway):
    pathway_file = REST.kegg_get(pathway).read()  # query and read each pathway

    # iterate through each KEGG pathway file, keeping track of which section
    # of the file we're in, only read the gene in each pathway
    current_section = None
    gene_symbols = set()
    diseases = set()
    drugs = set()
    for line in pathway_file.rstrip().split('\n'):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == '':
            current_section = section

        if current_section == 'DISEASE':
            disease = line[12:].split(' ')[0]
            diseases.add(disease)
        elif current_section == 'DRUG':
            drug = line[12:].split(' ')[0]
            drugs.add(drug)
        elif current_section == 'GENE':
            try:
                gene_identifiers, gene_description = line[12:].split('; ')
                gene_id, gene_symbol = gene_identifiers.split()
                gene_symbols.add(gene_symbol)
            except ValueError:
                print('WARNING: No gene found in {}'.format(line[12:]))

    return gene_symbols, diseases, drugs


def mouse_pathway_data(gene_symbols, mouse_pathways):
    mp = mouse_pathways
    nb_genes = len(gene_symbols)
    nb_pathways = len(mp)

    genes_p = np.zeros((nb_genes, nb_pathways))

    for i, p in enumerate(mp):
        gs, _, _ = get_pathway_info(p)

        # Store genes of the pathway
        idxs = np.argwhere(np.isin(gene_symbols, list(gs))).flatten()
        genes_p[idxs, i] = 1

    return genes_p


if __name__=="__main__":
    import scanpy as sc
    import pandas as pd

    data = sc.datasets.paul15()
    gene_symbols = data.to_df().columns.tolist()

    mp, mp_desc = list_KEGG_mouse_pathways()
    genes_p = mouse_pathway_data(gene_symbols, mp)  
    df = pd.DataFrame(genes_p, columns=mp.tolist(), index=gene_symbols)

    df.astype(int).to_csv('pathways.csv')
