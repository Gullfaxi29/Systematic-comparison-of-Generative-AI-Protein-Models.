import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from collections import defaultdict
import os
import argparse
from Bio.SVDSuperimposer import SVDSuperimposer
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

#combined_data = pd.read_csv('/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_mpnn_consistency_20240628_113033.csv')
#combined_data = pd.read_csv('/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_mpnn_fixed_consistency_20240628_115111.csv')
#combined_data = pd.read_csv('/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_mpnn_consistency_20240630_141546.csv')
combined_data = pd.read_csv('/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_mpnn_fixed_consistency_20240630_142549.csv')
#combined_data['entity_id'] = combined_data.design_file.str.split('_').str[3]
combined_data['entity_id'] = combined_data.design_file.str.split('_').str[4]
scores = combined_data.loc[:,['model','entity_id','design_file','tm_norm_aa','rmsd','omegafold_confidence']]
scores = scores.loc[scores.groupby('entity_id')['tm_norm_aa'].idxmax()].sort_values(by=['model','entity_id'])
#scores.to_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/mpnn_selected_designs.csv",index=False)
#scores.to_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/mpnn_fixed_selected_designs.csv",index=False)
#scores.to_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/mpnn_selected_designs.csv",index=False)
scores.to_csv("/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/mpnn_fixed_selected_designs.csv",index=False)

#selected_dir = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected"
#selected_dir = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected"
#selected_dir = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected"
selected_dir = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected"
aa_filepaths = [os.path.join(selected_dir, file) for file in os.listdir(selected_dir) if os.path.isfile(os.path.join(selected_dir, file))]
#discarded_dir = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns/structures_discarded"
#discarded_dir = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_discarded"
#discarded_dir = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns/structures_discarded"
discarded_dir = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_discarded"
aa_filepaths = aa_filepaths +  [os.path.join(discarded_dir, file) for file in os.listdir(discarded_dir) if os.path.isfile(os.path.join(discarded_dir, file))]

for path in aa_filepaths:
    if os.path.basename(path) in scores.design_file.unique():
        #new_path = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected/" + os.path.basename(path)
        #new_path = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected/" + os.path.basename(path)
        #new_path = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected/" + os.path.basename(path)
        new_path = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected/" + os.path.basename(path)
        if new_path == path:
            print(f"{os.path.basename(path)} Do nothing")
        else:
            print(f"{os.path.basename(path)} Move to selected")
            shutil.move(path, new_path)
    else:
        #new_path = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns/structures_discarded/" + os.path.basename(path)
        #new_path = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_discarded/" + os.path.basename(path)
        #new_path = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns/structures_discarded/" + os.path.basename(path)
        new_path = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_discarded/" + os.path.basename(path)
        if new_path == path:
            print(f"{os.path.basename(path)} Do nothing")
        else:
            print(f"{os.path.basename(path)} Move to discard")
            shutil.move(path, new_path)





