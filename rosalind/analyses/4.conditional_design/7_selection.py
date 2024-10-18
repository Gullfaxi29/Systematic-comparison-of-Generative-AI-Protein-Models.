import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

base_model_values = {
    'total_weighted_score': -188.0193162
    , 'fa_rep': 246.8021705
    , 'tm_norm_design': 1
    ,'rmsd': 0
    , 'rmsd (motif)': 0
    , 'motif_seq_alignment_blossum62': 273
    ,'whole_seq_alignment_blossum62': 1180
    ,'non_motif_seq_alignment_blossum62': 1007 
    , 'M': 0.03
    , 'H': 0.03
    , 'S': 0.08
    , 'A': 0.02
    , 'L': 0.07
    , 'C': 0.02
    , 'V': 0.05
    , 'T': 0.09
    , 'G': 0.07
    , 'R': 0.04
    , 'P': 0.06
    , 'Q': 0.05
    , 'E': 0.04
    , 'N': 0.06
    , 'F': 0.07
    , 'Y': 0.01
    , 'D': 0.04
    , 'K': 0.06
    , 'I': 0.06
    , 'W': 0.02
}

metadata = pd.DataFrame()
#for file in ["/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/generation_metadata_chroma_il10.csv"
#            ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/generation_metadata_evodiff_il10.csv"
#            ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/generation_metadata_proteingenerator_il10.csv"
#            ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/generation_metadata_protpardelle_il10.csv"
#            ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/generation_metadata_rfdiffusion_il10.csv"]:
#    print(file)
#    metadata = pd.concat([metadata,pd.read_csv(file)])
for file in ["/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/generation_metadata_chroma_tev.csv"
            ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/generation_metadata_evodiff_tev.csv"
            ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/generation_metadata_proteingenerator_tev.csv"
            ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/generation_metadata_protpardelle_tev.csv"
            ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/generation_metadata_rfdiffusion_tev.csv"]:
    print(file)
    metadata = pd.concat([metadata,pd.read_csv(file)])

metadata = metadata.loc[:,['model','entity_id','conditions']]
metadata.loc[metadata.model == 'ProteinMPNN','model'] = 'proteinmpnn'
metadata.loc[metadata.model == 'protein_generator','model'] = 'proteingenerator'
metadata.loc[metadata.model == 'protein_generator','model'] = 'proteingenerator'
metadata.loc[metadata.model == 'evodiff_OA_DM_640M','model'] = 'evodiff'
metadata.loc[metadata.model == 'RFdiffusion_150it','model'] = 'rfdiffusion'
metadata['condition'] = np.where(metadata.conditions.str.contains('complex', na=False), 'motif in complex with receptor', 'motif') 
metadata.loc[metadata.model == 'proteingenerator','condition'] = np.where(metadata.loc[metadata.model == 'proteingenerator','conditions'].str.contains('structure', na=False), 'motif (structure)', 'motif (sequence)') 
metadata = metadata.loc[:,['model','entity_id','condition']]


#energies = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_energies.csv")
energies = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_energies.csv")
energies['model'] = energies.filepath.str.split('/').str[-1].str.split('_').str[0]
energies.loc[energies.model == 'RFdiffusion','model'] = 'rfdiffusion'
uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
energies['entity_id'] = energies['filepath'].str.extract(f'({uuid_pattern})', expand=False)
energies['post'] = energies.filepath.str.split('/').str[-3]
#energies.loc[energies.post == 'il10_scaffolding','post'] = ''
energies.loc[energies.post == 'tev_scaffolding','post'] = ''
energies.drop('filepath',axis=1,inplace=True)

#aa = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_AA_enrichment.csv")
aa = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_AA_enrichment.csv")
aa = aa.iloc[1:,:]
aa['model'] = aa.filepath.str.split('/').str[-1].str.split('_').str[0]
aa.loc[aa.model == 'RFdiffusion','model'] = 'rfdiffusion'
aa['entity_id'] = aa['filepath'].str.extract(f'({uuid_pattern})', expand=False)
aa['post'] = aa.filepath.str.split('/').str[-3]
#aa.loc[aa.post == 'il10_scaffolding','post'] = ''
aa.loc[aa.post == 'tev_scaffolding','post'] = ''
aa.drop('filepath',axis=1,inplace=True)

#align_struct_full = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_struct_align_whole.csv")
align_struct_full = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_struct_align_whole.csv")
align_struct_full.columns = ['filepath'] + list(align_struct_full.columns[1:])
align_struct_full['model'] = align_struct_full.filepath.str.split('/').str[-1].str.split('_').str[0]
align_struct_full.loc[align_struct_full.model == 'RFdiffusion','model'] = 'rfdiffusion'
align_struct_full['entity_id'] = align_struct_full['filepath'].str.extract(f'({uuid_pattern})', expand=False)
align_struct_full['post'] = align_struct_full.filepath.str.split('/').str[-3]
#align_struct_full.loc[align_struct_full.post == 'il10_scaffolding','post'] = ''
align_struct_full.loc[align_struct_full.post == 'tev_scaffolding','post'] = ''
align_struct_full = align_struct_full.loc[:,['model','entity_id','post','tm_norm_design','rmsd']]

#align_struct_motif = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_struct_align_motif.csv")
align_struct_motif = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_struct_align_motif.csv")
align_struct_motif.columns = ['filepath'] + list(align_struct_motif.columns[1:])
align_struct_motif['model'] = align_struct_motif.filepath.str.split('/').str[-1].str.split('_').str[0]
align_struct_motif.loc[align_struct_motif.model == 'RFdiffusion','model'] = 'rfdiffusion'
align_struct_motif['entity_id'] = align_struct_motif['filepath'].str.extract(f'({uuid_pattern})', expand=False)
align_struct_motif['post'] = align_struct_motif.filepath.str.split('/').str[-3]
#align_struct_motif.loc[align_struct_motif.post == 'il10_scaffolding','post'] = ''
align_struct_motif.loc[align_struct_motif.post == 'tev_scaffolding','post'] = ''
align_struct_motif = align_struct_motif.loc[:,['model','entity_id','post','rmsd']]
align_struct_motif.columns = ['model','entity_id','post','rmsd (motif)']

#align_seq = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/refolded_seq_align.csv")
align_seq = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/refolded_seq_align.csv")
align_seq = align_seq.iloc[:,1:]
align_seq['model'] = align_seq.filepath.str.split('/').str[-1].str.split('_').str[0]
align_seq.loc[align_seq.model == 'RFdiffusion','model'] = 'rfdiffusion'
align_seq['entity_id'] = align_seq['filepath'].str.extract(f'({uuid_pattern})', expand=False)
align_seq['post'] = align_seq.filepath.str.split('/').str[-3]
#align_seq.loc[align_seq.post == 'il10_scaffolding','post'] = ''
align_seq.loc[align_seq.post == 'tev_scaffolding','post'] = ''
align_seq.drop('filepath',axis=1,inplace=True)

align_struct_submotif = pd.DataFrame(columns = ['model','entity_id','post'])
for i,file in enumerate([f"/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_struct_align_sub_motif_{i}.csv" for i in range (1,6)]):
    sub = pd.read_csv(file)
    sub.columns = ['filepath'] + list(sub.columns[1:])
    sub['model'] = sub.filepath.str.split('/').str[-1].str.split('_').str[0]
    sub.loc[sub.model == 'RFdiffusion','model'] = 'rfdiffusion'
    sub['entity_id'] = sub['filepath'].str.extract(f'({uuid_pattern})', expand=False)
    sub['post'] = sub.filepath.str.split('/').str[-3]
    sub.loc[sub.post == 'tev_scaffolding','post'] = ''
    sub = sub.loc[:,['model','entity_id','post','rmsd']]
    sub.columns = ['model','entity_id','post',f'rmsd (submotif{i+1})']
    align_struct_submotif = sub.merge(align_struct_submotif, on = ['model','entity_id','post'], how = 'left')

metrics = energies.merge(aa,how = 'inner', on = ['model','entity_id','post']).merge(align_struct_motif,how = 'inner', on = ['model','entity_id','post']).merge(align_struct_full,how = 'inner', on = ['model','entity_id','post']).merge(align_struct_submotif,how = 'inner', on = ['model','entity_id','post']).merge(align_seq,how = 'inner', on = ['model','entity_id','post']).merge(metadata,how = 'left', on = ['model','entity_id'])
metrics.loc[metrics.model == 'mpnn','condition'] = 'motif'
metrics['approach'] = metrics['model'] + '\n' + metrics['condition'] + '\n' + metrics['post']
metrics.fillna(0,inplace = True)
metrics = metrics.loc[:,['model','entity_id','approach','sequence','fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4',
       'lk_ball_wtd', 'fa_elec', 'pro_close', 'hbond_sr_bb', 'hbond_lr_bb',
       'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'omega', 'fa_dun', 'p_aa_pp',
       'yhh_planarity', 'ref', 'rama_prepro', 'total_weighted_score','M', 'H', 'S', 'A', 'L', 'C', 'V', 'T', 'G', 'R',
       'P', 'Q', 'E', 'N', 'F', 'Y', 'D', 'K', 'I', 'W',
       'tm_norm_design', 'rmsd','rmsd (motif)','rmsd (submotif1)','rmsd (submotif2)','rmsd (submotif3)','rmsd (submotif4)','rmsd (submotif5)', 'whole_seq_alignment_blossum62',
       'motif_seq_alignment_blossum62', 'non_motif_seq_alignment_blossum62',]]

#metrics = metrics.loc[metrics.AWHinclude > 0,:]

#backbones are blues, full structures are reds, sequences are earthy tones
custom_palette ={'rfdiffusion': '#005cdf',
'evodiff': '#93355c',
'proteingenerator': '#a4837b',
'protpardelle': '#ffc73c',
'chroma': '#809a2d',
'mpnn': 'dimgrey'}
#custom_order = ['protpardelle', 'protein_generator', 'chroma', 'framediff', 'genie', 'proteinsgm', 'foldingdiff', 'rita', 'ESM', 'evodiff', 'ProGen2', 'protgpt2','mpnn']
custom_order = ['chroma\nmotif\n'
                ,'chroma\nmotif in complex with receptor\n'
                ,'chroma\nmotif\nrefolded'
                ,'chroma\nmotif in complex with receptor\nrefolded'
                ,'protpardelle\nmotif\n', 'protpardelle\nmotif\nrefolded'
                ,'proteingenerator\nmotif (sequence)\n'
                ,'proteingenerator\nmotif (structure)\n'
                ,'proteingenerator\nmotif (sequence)\nrefolded'
                ,'proteingenerator\nmotif (structure)\nrefolded'
                ,'evodiff\nmotif\n', 'mpnn\nmotif\n'
                ,'rfdiffusion\nmotif\nMPNN_redesigns'
                ,'rfdiffusion\nmotif in complex with receptor\nMPNN_redesigns'
                ,'rfdiffusion\nmotif\nMPNN_redesigns_fixed'
                ,'rfdiffusion\nmotif in complex with receptor\nMPNN_redesigns_fixed']
with PdfPages('/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/violin.pdf') as pdf:
    numerical_columns = metrics.select_dtypes(include='number').columns
    for col in numerical_columns:
        #subset = metrics.loc[(metrics["MODEL"]==row['MODEL']) & (metrics["CONDITIONS"]==row['CONDITIONS']),:].copy(deep=True)
        plt.figure(figsize=(18, 12))
        sns.violinplot(x="approach",y=col,data=metrics, inner="box", density_norm="width", hue = "model",palette = custom_palette, order = custom_order)#, width = 10)
        plt.title(f'Metric Distributions - {col}')
        if col in base_model_values.keys():
            plt.axhline(y=base_model_values[col], color='black', linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout(pad=3.0)
        pdf.savefig()
        plt.close()
        
        
corr_mat = metrics[numerical_columns].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix')
plt.savefig('/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/correlation_matrix.pdf', format='pdf')
plt.close()

plt.figure(figsize=(12,12))
sns.pairplot(metrics[list(numerical_columns) + ["approach"]],hue= "model")
plt.savefig('/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/pairplot.pdf', format='pdf')
plt.close()


selection = metrics.copy(deep = True)
#selection = selection[selection['total_weighted_score'] <=1436.3]
#selection.groupby("approach").count()
#selection

def bottom_x_percent_threshold(group, col, x):
    return group[col].quantile(x)

# Calculate the threshold for each group
thresholds = selection.groupby('approach').apply(bottom_x_percent_threshold, 'total_weighted_score',0.25).reset_index(name='Threshold')

selection = selection.merge(thresholds, on='approach')

# Filter the DataFrame to include only the bottom 10% entries
selection = selection[selection['total_weighted_score'] <= selection['Threshold']].drop(columns='Threshold')


selection = selection.groupby('approach').apply(lambda x: x.nsmallest(10, 'rmsd (motif)')).reset_index(drop=True)

with PdfPages('/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/violin_selection.pdf') as pdf:
    numerical_columns = selection.select_dtypes(include='number').columns
    for col in numerical_columns:
        plt.figure(figsize=(18, 12))
        sns.violinplot(x="approach",y=col,data=selection, inner="box", density_norm="width", hue = "model",palette = custom_palette, order = custom_order)#, width = 10)
        plt.title(f'Metric Distributions - {col}')
        if col in base_model_values.keys():
            plt.axhline(y=base_model_values[col], color='black', linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout(pad=3.0)
        pdf.savefig()
        plt.close()

selection = selection.loc[:,['model','entity_id','approach','sequence']]
selection['selected'] = True
metrics = metrics.merge(selection, how = 'left', on = ['model','entity_id','approach','sequence']).fillna(False)

metrics.to_csv('/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/selection.csv')