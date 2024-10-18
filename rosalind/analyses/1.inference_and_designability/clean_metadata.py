import numpy as np
import pandas as pd
import math
import os

model_name_map = {
    'RFdiffusion_150it' : 'RFdiffusion',
    'genie_swissprot_l_256' : 'Genie',
    'proteinsgm' : 'ProteinSGM',
    'foldingdiff' : 'FoldingDiff',
    'framediff': 'FrameDiff',
    'chroma' : 'Chroma',
    'protpardelle' : 'Protpardelle',
    'protein_generator' : 'Protein-Generator',
    'evodiff_OA_DM_640M' : 'EvoDiff',
    'rita_xl' : 'RITA',
    'ProGen2' : 'ProGen2',
    'protgpt2' : 'ProtGPT2',
    'ESM_Design' : 'ESM-Design',
    'omegafold' : 'OmegaFold',
    'ProteinMPNN' : 'ProteinMPNN',
    }

metadata_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/metadata"
meta = pd.concat([pd.read_csv(os.path.join(metadata_dir, file)) 
                  for file in os.listdir(metadata_dir) 
                  if os.path.isfile(os.path.join(metadata_dir, file))])
meta.loc[:,'model'] = meta.loc[:,'model'].map(model_name_map)
#
fold_meta = meta.loc[(meta.task == "Structure Prediction") & (meta.model == "OmegaFold"),:]
fold_meta = fold_meta.loc[fold_meta.output_dir_path != '/content/drive/MyDrive/Generative_Models/utilities/omegafold/bug_fixed',:]
fold_meta.loc[:,'length'] = fold_meta.loc[:,'output_dir_path'].str.split('all_len').str[-1].astype(int)
fold_meta.loc[:,'wall_time_batch'] = fold_meta.loc[:,'wall_time_batch'].str.extract(r'(\d+\.\d+)').iloc[:, 0].astype(float)
fold_meta = fold_meta[["model","task", "length","wall_time_batch","batch_size","batch_id"]]
fold_meta = fold_meta.groupby(['model','task','length'])[['wall_time_batch','batch_size']].sum().reset_index() #Later folds were done in multiple batches so need to recalc overall wall time for batch/task
fold_meta.loc[:,'wall_time_task'] = fold_meta.wall_time_batch / fold_meta.batch_size
fold_meta.to_csv('/scratch/alexb2/generative_protein_models/share/fold_metadata_cleaned.csv',index=False)
#
mpnn_meta = meta.loc[(meta.task == "Sequence Redesign"),:]
mpnn_meta = mpnn_meta[["model","task", "length", "wall_time_task", "entity_id"]]
mpnn_meta.loc[:,'wall_time_task'] = mpnn_meta.loc[:,'wall_time_task'].str.extract(r'(\d+\.\d+)').iloc[:, 0].astype(float)
mpnn_meta = mpnn_meta.loc[mpnn_meta.length <= 200,:].groupby(['length'])['wall_time_task'].mean().reset_index()
mpnn_meta.to_csv('/scratch/alexb2/generative_protein_models/share/mpnn_metadata_cleaned.csv',index=False)
#
gen_meta = meta.loc[(meta.task != "Sequence Redesign") & (meta.task != "Structure Prediction"),:]
gen_meta.loc[:,'length'] = gen_meta.loc[:,'conditions'].str.extract(r'(\d+)').iloc[:,0].astype(int)
gen_meta.loc[gen_meta.task == 'backbone generation (6D)','model'] = 'ProteinSGM'
gen_meta.loc[gen_meta.task == 'backbone generation (Rosetta)','model'] = 'ProteinSGM'
gen_meta.loc[:,'wall_time_task'] = gen_meta.loc[:,'wall_time_task'].str.extract(r'(\d+\.\d+)').iloc[:, 0].astype(float)
gen_meta = gen_meta[["batch_id","batch_size","entity_id","model","task", "length", "wall_time_task","wall_time_batch"]]
gen_meta = gen_meta.loc[~((gen_meta.model == 'FoldingDiff') & (gen_meta.length > 128)),:] #All the generations of above len 128 for foldingdiff actually just defaulted to 128, so we need to remove that metadata
gen_meta = gen_meta.loc[~(gen_meta.length > 200),:] #also just remove any generations over 200 res for simplicity
sgm = gen_meta.loc[gen_meta.model == "ProteinSGM",:]
gen_meta = gen_meta.loc[gen_meta.model != "ProteinSGM",:]
sgm = sgm.loc[sgm.task == 'backbone generation (Rosetta)',["model","entity_id","length","wall_time_task"]].merge(sgm.loc[sgm.task == 'backbone generation (6D)',["batch_size","batch_id","length","wall_time_task"]],how = 'left', on = 'length')
sgm.columns = ['model', 'entity_id', 'length', 'wall_time_refine', 'batch_size', 'batch_id', 'wall_time_gen']
sgm.loc[:,'wall_time_task'] = sgm.wall_time_gen + sgm.wall_time_refine
sgm['task'] = 'backbone_pdb_generation'
gen_meta = pd.concat([gen_meta, sgm])
#
gen_meta = gen_meta.merge(mpnn_meta,how='left',on='length', suffixes=('', '_mpnn')).merge(fold_meta.loc[:,['length','wall_time_task']],how='left',on='length', suffixes=('', '_fold'))
#
gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_bb'] = gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_task']
gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_seq'] = gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_bb'] + gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_task_mpnn']
gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_aa'] = gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_seq'] + gen_meta.loc[gen_meta.task == 'backbone_pdb_generation','wall_time_task_fold']
#
gen_meta.loc[gen_meta.task == 'sequence_generation','wall_time_bb'] = None
gen_meta.loc[gen_meta.task == 'sequence_generation','wall_time_seq'] = gen_meta.loc[gen_meta.task == 'sequence_generation','wall_time_task']
gen_meta.loc[gen_meta.task == 'sequence_generation','wall_time_aa'] = gen_meta.loc[gen_meta.task == 'sequence_generation','wall_time_seq'] + gen_meta.loc[gen_meta.task == 'sequence_generation','wall_time_task_fold']
#
gen_meta.loc[gen_meta.task == 'all_atom_pdb_generation','wall_time_bb'] = None
gen_meta.loc[gen_meta.task == 'all_atom_pdb_generation','wall_time_seq'] = None
gen_meta.loc[gen_meta.task == 'all_atom_pdb_generation','wall_time_aa'] = gen_meta.loc[gen_meta.task == 'all_atom_pdb_generation','wall_time_task']
#
gen_meta = gen_meta.loc[:,['model','length','task','wall_time_gen','wall_time_bb','wall_time_seq','wall_time_aa']]
gen_meta[['length', 'wall_time_gen', 'wall_time_bb','wall_time_seq','wall_time_']] = gen_meta[['length','wall_time_gen','wall_time_bb','wall_time_seq','wall_time_aa']].apply(pd.to_numeric, errors='coerce')

gen_meta.loc[gen_meta.task == 'all_atom_pdb_generation',['model', 'wall_time_gen','wall_time_bb','wall_time_seq','wall_time_aa']]

gen_meta.to_csv('/scratch/alexb2/generative_protein_models/share/generation_metadata_cleaned.csv',index=False)