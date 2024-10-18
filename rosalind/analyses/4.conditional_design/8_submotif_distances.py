import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

#calced in pymol for tev_base_monomer
com_coords = [np.array([13.487161943501913, -2.9093629614636045, 6.677691957830119])
			,np.array([13.99860513169111, 0.2698625934784939, 0.10243945461772264])
			,np.array([-4.326729141124914, 3.253298130261253, 2.7987105224228728])
			,np.array([-1.8415872433945255, 8.762379323629755, 4.387562830009738])
			,np.array([6.9100793926441195, 11.239228024096215, -0.6191587635548905])]


uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
align_struct_submotif = pd.DataFrame()
for i,file in enumerate([f"/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_struct_align_sub_motif_{i}.csv" for i in range (1,6)]):
	sub = pd.read_csv(file)
	sub.columns = ['filepath'] + list(sub.columns[1:])
	sub['model'] = sub.filepath.str.split('/').str[-1].str.split('_').str[0]
	sub.loc[sub.model == 'RFdiffusion','model'] = 'rfdiffusion'
	sub['entity_id'] = sub['filepath'].str.extract(f'({uuid_pattern})', expand=False)
	sub['post'] = sub.filepath.str.split('/').str[-3]
	sub.loc[sub.post == 'tev_scaffolding','post'] = ''
	sub = sub.loc[:,['model','entity_id','post','svd_t1', 'svd_t2',	'svd_t3', 'svd_u11', 'svd_u12', 'svd_u13', 'svd_u21', 'svd_u22', 'svd_u23', 'svd_u31', 'svd_u32', 'svd_u33']]
	sub['motif'] = i+1
	align_struct_submotif = pd.concat([align_struct_submotif,sub])

distances = align_struct_submotif.loc[:,['model','entity_id', 'post']].drop_duplicates()
combinations = align_struct_submotif.merge(align_struct_submotif, on = ['model','entity_id', 'post'], how = 'left')
combinations = combinations.loc[combinations.motif_x < combinations.motif_y,:]
for index,row in combinations.iterrows():
	rot_x = np.reshape(row[['svd_u11_x','svd_u12_x', 'svd_u13_x', 'svd_u21_x', 'svd_u22_x', 'svd_u23_x', 'svd_u31_x','svd_u32_x', 'svd_u33_x']],(3,3))
	trans_x = np.array(row[['svd_t1_x', 'svd_t2_x', 'svd_t3_x']])
	base_motif_x = com_coords[row['motif_x']-1]
	trans_motif_x = base_motif_x + trans_x
	rot_motif_x = rot_x.dot(base_motif_x)
	rot_y = np.reshape(row[['svd_u11_y','svd_u12_y', 'svd_u13_y', 'svd_u21_y', 'svd_u22_y', 'svd_u23_y', 'svd_u31_y','svd_u32_y', 'svd_u33_y']],(3,3))
	trans_y = np.array(row[['svd_t1_y', 'svd_t2_y', 'svd_t3_y']])
	base_motif_y = com_coords[row['motif_y']-1]
	trans_motif_y = base_motif_y + trans_y
	rot_motif_y = rot_y.dot(base_motif_y)
	distances.loc[(distances.model == row['model']) & (distances.entity_id == row['entity_id']) & (distances.post == row['post']),f'euc_distance_motif_{row['motif_x']}_{row['motif_y']}'] = np.linalg.norm(trans_motif_x - trans_motif_y)
	R_rel = np.dot(rot_x, rot_y.T)
	rotation = R.from_matrix(R_rel)
	euler_angles = rotation.as_euler('xyz', degrees=True)
	distances.loc[(distances.model == row['model']) & (distances.entity_id == row['entity_id']) & (distances.post == row['post']),f'eul_ang_x_motif_{row['motif_x']}_{row['motif_y']}'] = euler_angles[0]
	distances.loc[(distances.model == row['model']) & (distances.entity_id == row['entity_id']) & (distances.post == row['post']),f'eul_ang_y_motif_{row['motif_x']}_{row['motif_y']}'] = euler_angles[1]
	distances.loc[(distances.model == row['model']) & (distances.entity_id == row['entity_id']) & (distances.post == row['post']),f'eul_ang_z_motif_{row['motif_x']}_{row['motif_y']}'] = euler_angles[2]
	distances.loc[(distances.model == row['model']) & (distances.entity_id == row['entity_id']) & (distances.post == row['post']),f'euc_distance_motif_{row['motif_x']}'] = np.linalg.norm(trans_x)
	distances.loc[(distances.model == row['model']) & (distances.entity_id == row['entity_id']) & (distances.post == row['post']),f'euc_distance_motif_{row['motif_y']}'] = np.linalg.norm(trans_y)


base_model_values = {
    'euc_distance_motif_1': 0
    , 'euc_distance_motif_2': 0
    , 'euc_distance_motif_3': 0
    ,'euc_distance_motif_4': 0
    , 'euc_distance_motif_5': 0
}

for combo in itertools.combinations(range(1, 6),2):
	base_model_values[f'eul_ang_x_motif_{combo[0]}_{combo[1]}'] = 0
	base_model_values[f'eul_ang_y_motif_{combo[0]}_{combo[1]}'] = 0
	base_model_values[f'eul_ang_z_motif_{combo[0]}_{combo[1]}'] = 0
	base_model_values[f'euc_distance_motif_{combo[0]}_{combo[1]}'] = np.linalg.norm(com_coords[combo[0]-1] - com_coords[combo[1]-1])

metadata = pd.DataFrame()
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

distances = distances.merge(metadata, on = ['model','entity_id'], how = 'left')

distances['approach'] = distances['model'] + '\n' + distances['condition'] + '\n' + distances['post']

custom_palette ={'rfdiffusion': '#005cdf',
'evodiff': '#93355c',
'proteingenerator': '#a4837b',
'protpardelle': '#ffc73c',
'chroma': '#809a2d',
'mpnn': 'dimgrey'}
#custom_order = ['protpardelle', 'protein_generator', 'chroma', 'framediff', 'genie', 'proteinsgm', 'foldingdiff', 'rita', 'ESM', 'evodiff', 'ProGen2', 'protgpt2','mpnn']

selection = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/selection.csv")
selection = selection.loc[selection.selected,['entity_id','approach']]

distances = selection.merge(distances,on = ['entity_id','approach'], how = 'left')
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
with PdfPages('/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/distances_violin_selected.pdf') as pdf:
	numerical_columns = distances.select_dtypes(include='number').columns
	numerical_columns = [col for col in numerical_columns if 'ang' not in col]
	for col in numerical_columns:
		print(col)
		plt.figure(figsize=(18, 12))
		sns.violinplot(x="approach",y=col,data=distances, inner="box", density_norm="width", hue = "model",palette = custom_palette, order = custom_order)#, width = 10)
		plt.title(f'Metric Distributions - {col}')
		if col in base_model_values.keys():
			plt.axhline(y=base_model_values[col], color='black', linestyle='--')
		plt.xticks(rotation=45)
		plt.tight_layout(pad=3.0)
		pdf.savefig()
		plt.close()
