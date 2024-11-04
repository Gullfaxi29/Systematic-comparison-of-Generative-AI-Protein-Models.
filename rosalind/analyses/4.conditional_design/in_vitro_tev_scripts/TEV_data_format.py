import pandas as pd
from scipy import stats
import numpy as np

counts = pd.read_csv('TEV-image_counts3.csv').iloc[:,[0,3,4,5,6]] #pd.read_csv('TEV-image_counts.tsv', sep='\t').iloc[:,[0,3,4,5,6]]
counts.columns=['design_id','plate','replicate','BF','GFP']
nc = counts.loc[counts.design_id == 'NC',:].copy()
counts = counts.loc[counts.design_id != 'NC',:].copy()

designs = pd.read_csv('TEVdesigns.csv')

designs = designs.loc[:,['TEVdesign','entity_id','model','approach_x','tm_norm_design','rmsd','rmsd (motif)','whole_seq_alignment_blossum62', 'motif_seq_alignment_blossum62','non_motif_seq_alignment_blossum62']]
designs.columns = ['design_id','entity_id','model','approach','TM-Score','RMSD','RMSD (motif)','Sequence Alignment\n(BLOSSUM62)', 'Sequence Alignment\n(Motif Region)','Sequence Alignment\n(Non-Motif)']
designs['model'] = designs['model'].replace({'wildtype': 'Wild-Type'
                                       , 'rfdiffusion': 'RFdiffusion'
                                       , 'proteingenerator': 'ProteinGenerator'
                                       , 'chroma' : 'Chroma'
                                       , 'mpnn' : 'ProteinMPNN'
                                       , 'protpardelle' : 'Protpardelle'
                                       , 'evodiff': 'EvoDiff'})
#designs = designs.loc[designs.approach != 'rfdiffusion\nmotif in complex with receptor\nMPNN_redesigns_fixed',:]
designs.loc[:,'complex'] = designs.loc[:,'approach'].str.contains('complex')
designs.loc[:,'sequence_motif'] = (designs.loc[:,'approach'].str.contains('(structure)')) | (designs.loc[:,'model'] == 'EvoDiff')
designs.loc[:,'structure_motif'] = ~designs.sequence_motif
designs.loc[:,'mpnn'] = (designs.loc[:,'approach'].str.contains('MPNN_redesigns'))  | (designs.loc[:,'model'] == 'ProteinMPNN')
designs.loc[:,'mpnn_fixed'] = (designs.loc[:,'approach'].str.contains('MPNN_redesigns_fixed')) | (designs.loc[:,'model'] == 'ProteinMPNN')
designs.drop('approach',axis=1,inplace=True)
 
nc['GFP_norm_2kBFcells_NC'] = (nc.GFP/nc.BF)*2000#*25000
counts['GFP_norm_2kBFcells'] = (counts.GFP/counts.BF)*2000#*25000
counts = pd.merge(counts, nc.loc[:,['plate','replicate','GFP_norm_2kBFcells_NC']], on=['plate', 'replicate'], how='left')
counts['Fold Increase\n(From GFP+ Cells)'] = counts.GFP_norm_2kBFcells/counts.GFP_norm_2kBFcells_NC
counts['fold_increase_log2'] = np.log2(counts['Fold Increase\n(From GFP+ Cells)'])

def perform_t_test(group):
    t_stat, p_value = stats.ttest_1samp(group['fold_increase_log2'], 0)
    return pd.Series({'t_stat': t_stat, 'p_value': p_value})

t_test = counts.groupby('design_id').apply(perform_t_test).reset_index()
t_test.columns = ['design_id','t_stat (all replicates)','p_value (all replicates)']

t_test_sans_rep2 = counts.loc[counts.replicate != 'replicate2',:].groupby('design_id').apply(perform_t_test).reset_index()
t_test_sans_rep2.columns = ['design_id','t_stat (sans replicate 2)','p_value (sans replicate 2)']

data = pd.merge(counts, designs, on='design_id', how='left')
data = pd.merge(data, t_test, on='design_id', how='left')
data = pd.merge(data, t_test_sans_rep2, on='design_id', how='left')

data.to_csv('TEV_image_counts_with_designs2.csv', index=False)

print(f"completed")
