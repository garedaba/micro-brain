"""
BEFORE RUNNING
##### Manually assign corresponding labels from histological atlas
see `MicroarrayData/all_structure_lut.csv`

##### Reannotate microarray data
microarray probes reannotated (March 2023)

##### download latest biomart data (approved symbols for genes etc, April 2023)
https://biomart.genenames.org/martform/#!/default/HGNC?datasets=hgnc_gene_mart
"""

import pandas as pd
import glob
import numpy as np
import json

from tqdm import tqdm

from itertools import combinations
from scipy.stats import spearmanr

def main(ds_cutoff = 0.2, min_rho = 50):
    args_dict = dict(zip(['ds_cutoff', 'min_rho'], [ds_cutoff, min_rho]))
    with open('MicroarrayData/proc_args.txt', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # get list of specimens # download from Allen Institute
    subs = glob.glob('MicroarrayData/origdata/H*')

    # load probe annotations (precalculated - see above)
    print('adding up-to-date probe annotations...')
    probe_annot = pd.read_csv('MicroarrayData/reannotate/probes2annotateALL_merged_readAnnotation.txt', delimiter='\t')

    # identify probe-to-gene mapping
    probe_ids = probe_annot['#PROBE_ID']
    probe_hits = probe_annot['HIT']

    probe_symbols = []
    chromosome = []
    # for each probe
    for i in tqdm(np.arange(len(probe_hits)-1), desc='gene symbols'):

        # get gene symbol entry accounting for multiple hits
        hits = probe_hits[i].split('|')[3::4]
        hits = [h.split(',')[1] for h in hits]
        chrs = probe_hits[i].split('|')[0::4][0]

        # drop any with ambiguous mapping to more than one gene
        if len(set(hits)) != 1:
            hits = np.nan
        else:
            hits = hits[0]

        probe_symbols.append(hits)
        chromosome.append(chrs)

    # load in HGNC biomart annotations - see above
    print('adding up-to-date entrez ids...')
    biomart = pd.read_csv('MicroarrayData/biomart.txt', delimiter='\t')

    # attach up-to-date entrez symbols to genes
    entrez_ids = []
    for g in tqdm(probe_symbols, desc='entrez'):
        if sum(biomart['Approved symbol']==g)>0:
            e = str(int(biomart[biomart['Approved symbol']==g]['NCBI gene ID'].values[0]))
            entrez_ids.append(e)
        else:
            entrez_ids.append(np.nan)

    # collate into new dataframe
    new_probe_annot = pd.DataFrame((probe_ids.values[:-1], probe_symbols, entrez_ids, chromosome)).T
    new_probe_annot.columns = ['probe_id','gene_symbol', 'entrez', 'chromosome']
    new_probe_annot.set_index('probe_id', inplace=True)
    new_probe_annot

    # save out
    new_probe_annot.to_csv('MicroarrayData/probe_annotations.csv')
    print('see: MicroarrayData/probe_annotations.csv')
    print('')

    # process tissue structure names
    structures = pd.read_csv('MicroarrayData/all_structure_lut.csv', delimiter=',')
    structures.set_index('structure_id', inplace=True)

    all_data = []
    # for each specimen
    print('initial processing...')
    for s in subs:
        # load expression data, metadata and present/absent calls
        expression = pd.read_csv('{:}/expression_matrix.csv'.format(s), header=None)
        rows = pd.read_csv('{:}/rows_metadata.csv'.format(s))
        columns = pd.read_csv('{:}/columns_metadata.csv'.format(s))
        structure_ids = columns['structure_id']
        pa_call = pd.read_csv('{:}/pa_call.csv'.format(s), header=None) #'1' is present

        # set index and map tissue and regions values
        columns.set_index('structure_id', inplace=True)
        columns['tissue'] = columns.index.map(structures['tissue'])
        columns['cortical_region'] = columns.index.map(structures['cortical_region'])
        columns['specimen_id'] = s.split('/')[-1]

        # set index and map new gene symbols and chromosomes
        rows = pd.read_csv('{:}/rows_metadata.csv'.format(s))
        rows.set_index('probeset_name', inplace=True)
        rows['new_gene_symbol'] = rows.index.map(new_probe_annot['gene_symbol'])
        rows['new_entrez'] = rows.index.map(new_probe_annot['entrez'])
        rows['new_chromosome'] = rows.index.map(new_probe_annot['chromosome'])
        rows.reset_index(inplace=True)

        probe_id = rows['probeset_name']
        entrez_id = rows['new_entrez']
        has_entrez = ~pd.isna(entrez_id.values)
        probe_id = probe_id[has_entrez]

        # select only expression data with valid entrez id
        expression = expression.drop(labels=0, axis='columns').T
        expression = expression.reset_index().drop('index', axis='columns')
        expression = expression.loc[:, has_entrez]
        expression.columns = probe_id

        # corresponding p/a calls
        pa_call = pa_call.drop(labels=0, axis='columns').T
        pa_call = pa_call.reset_index().drop('index', axis='columns')
        pa_call = pa_call.loc[:, has_entrez]
        pa_call.columns = probe_id

        # join to sample data
        columns = columns.reset_index()
        expression = pd.concat([columns, expression], axis='columns')
        pa_call = pd.concat([columns, pa_call], axis='columns')

        # melt
        expression_long = expression.melt(id_vars = expression.columns[:7], value_vars=expression.columns[7:], var_name='probeset_id', value_name='expression')
        pa_call_long = pa_call.melt(id_vars = pa_call.columns[:7], value_vars=pa_call.columns[7:], var_name='probeset_id', value_name='p/a')

        # add p/a column
        assert(sum(expression_long['probeset_id'] == pa_call_long['probeset_id']) == len(expression_long))
        expression_long['p/a'] = pa_call_long['p/a']

        # add entrez_id and gene symbol
        expression_long['entrez_id'] = expression_long.set_index('probeset_id').index.map(rows.set_index('probeset_name')['new_entrez'])
        expression_long['gene'] = expression_long.set_index('probeset_id').index.map(rows.set_index('probeset_name')['new_gene_symbol'])

        # drop any absent calls
        present_calls = expression_long['p/a'] == 1
        absent_calls = len(expression_long) - sum(present_calls)
        print('specimen: {:} {:} ({:.2f}%) noisy probes removed'.format(s, absent_calls, 100*(absent_calls/len(expression_long))))
        expression_long = expression_long[expression_long['p/a'] == 1]
        expression_long.drop('p/a', axis='columns', inplace = True)

        all_data.append(expression_long)

    # concatenate all data - 35M rows!
    # 1 row = expression of a single probe in a single region in a single tissue in a single subject
    all_data = pd.concat(all_data)

    print('dropping probes from non-cortical regions, in MZ or subpial zone')
    # drop any uncertain
    all_data = all_data.loc[all_data['tissue']!='???']
    all_data = all_data.loc[all_data['tissue']!='other']
    # drop any with NO_LABEL in atlas
    all_data = all_data.loc[all_data['cortical_region']!='NO_LABEL']
    # drop any without specific cortical label
    all_data = all_data.loc[all_data['cortical_region']!='brain_tissue']
    # drop any in marginal layer
    all_data = all_data.loc[all_data['tissue']!='marginal']
    # drop any in subpial layer
    all_data = all_data.loc[all_data['tissue']!='subpial']

    # check
    check_processing(all_data)

    # save out - 20M rows
    # 1 row = expression of a single probe in a single region in a single tissue in a single subject
    all_data.to_csv('MicroarrayData/all_data.csv', index=None)
    print('see: MicroarrayData/all_data.csv')
    print('')

    # calculate differential stability for each probe
    print('calculating differential stability...')
    ds = all_data.groupby('probeset_id').apply(calc_ds, min_rho) # minimum # of shared structures required to calculate rho
    ds_df = pd.DataFrame(ds).reset_index()
    ds_df.columns = ['probeset_id', 'DS']

    # merge with all_data
    all_data_ds = pd.merge(all_data, ds_df, on='probeset_id')

    # drop any with no DS (too few structures sampled)
    all_data_ds = all_data_ds[~np.isnan(all_data_ds['DS'])]

    # where more than one probe maps to a gene, choose the one with highest DS
    print('selecting top probes...')
    selected_probes = []
    grouped = all_data_ds.groupby('gene')
    for gene, groups in tqdm(grouped):
        max_probe = groups.groupby('probeset_id')['DS'].mean().idxmax()
        selected_probes.append(max_probe)

    selected_data_ds = all_data_ds[all_data_ds['probeset_id'].isin(selected_probes)]

    # drop any with DS lower than DS_CUTOFF
    selected_data_ds = selected_data_ds[selected_data_ds['DS'] > ds_cutoff]
    selected_data_ds.reset_index(inplace=True, drop=True)
    print('number of genes after probe selection: {:}'.format(len(np.unique(selected_data_ds['gene']))))

    # tissue labels
    print('tissue_labels')
    print(selected_data_ds['tissue'].unique())

    # regional labels
    print('')
    print('regional_labels')
    print(selected_data_ds['cortical_region'].unique())


    # 6.5M rows
    # 29 cortical regions
    # 5 tissue types
    # 10061 genes

    selected_data_ds.to_csv('MicroarrayData/regional_data.csv', index=None)
    print('regional microarray data: MicroarrayData/regional_data.csv')

    # take mean expression for any repeated samples within the same cortical region eg: inner and outer cortical plate
    # 3.7M rows
    # 29 cortical regions
    # 5 tissue types
    # 10061 genes
    grouped = selected_data_ds.groupby(['tissue','cortical_region','specimen_id', 'gene']).mean().drop(['structure_id', 'well_id'], axis='columns').reset_index()
    grouped.to_csv('MicroarrayData/final_data.csv', index=None)
    print('final microarray data: MicroarrayData/final_data.csv')

# define a function to calculate differential stability over each pair of specimens
def calc_ds(probeset_df, n=5):

    # iterate over each unique pair of specimens
    all_pairs = list(combinations(probeset_df['specimen_id'].unique(), 2))

    rho = []
    for i, (specimen_id1, specimen_id2) in enumerate(all_pairs):
            # get expression of each specimen
            expr_spec1 = probeset_df.loc[(probeset_df['specimen_id']==specimen_id1), :]
            expr_spec2 = probeset_df.loc[(probeset_df['specimen_id']==specimen_id2), :]
            # identify structure where expression is measured in both specimens
            shared_probes = pd.merge(expr_spec1, expr_spec2, on='structure_id')
            # if there are more than n shared structures, calculate rho
            if len(shared_probes) > n:
                rho.append(spearmanr(shared_probes[['expression_x', 'expression_y']])[0])

    if len(rho) > 0:
        return np.average(rho)
    else:
        return np.nan

def check_processing(proc_data):
    # select a random sample from processed data and make sure it matches the original data

    # select random sample
    rand_sample = proc_data.iloc[np.random.choice(len(proc_data))]

    # check it matches original data
    expression = pd.read_csv('MicroarrayData/origdata/{:}/expression_matrix.csv'.format(rand_sample['specimen_id']), header=None).loc[:,1:]
    rows = pd.read_csv('MicroarrayData/origdata/{:}/rows_metadata.csv'.format(rand_sample['specimen_id']))
    columns = pd.read_csv('MicroarrayData/origdata/{:}/columns_metadata.csv'.format(rand_sample['specimen_id']))

    d = expression.loc[np.where(rows.probeset_name == rand_sample['probeset_id'])[0]].loc[:,(columns.structure_acronym==rand_sample['structure_acronym']).values]

    assert rand_sample['expression'] == d.values

if __name__ == '__main__':
    import argparse, os

    def limited_float(x):
        x = float(x)
        if x < 0 or x > 1.0:
            raise argparse.ArgumentTypeError(x)
        return x

    parser = argparse.ArgumentParser(description='Process regional microarray data')

    parser.add_argument('-c', '--ds_cutoff', type=limited_float, required=True,
            help='cut off for differential stability')
    parser.add_argument('-n', '--min_rho', type=int, required=True,
            help='minimum number of shared structure needed for DS calculation')
    args = parser.parse_args()

    main(ds_cutoff = args.ds_cutoff, min_rho=args.min_rho)
