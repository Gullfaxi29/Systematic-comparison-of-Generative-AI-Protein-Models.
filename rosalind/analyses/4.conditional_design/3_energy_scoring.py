import pyrosetta; pyrosetta.init()
from pyrosetta import *
import os
import argparse
import pandas as pd
pyrosetta.init()

#parser = argparse.ArgumentParser(description="Process PDB files in a directory and log scores.")
#parser.add_argument('directory', type=str, help="The directory containing PDB files.")
#args = parser.parse_args()
sfxn = get_score_function(True)
data = []

#directory = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/chroma"
#for root, dirs, files in os.walk(directory):

for directory in ["/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/chroma"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/protpardelle"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/protein_generator"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/refolded/chroma"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/refolded/protpardelle"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/refolded/protein_generator"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/evodiff"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/mpnn_solo"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected"
                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected"]:
    print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if ".pdb" in file:
                scores = {
                    'fa_atr' : None
                    ,'fa_rep' : None
                    ,'fa_sol' : None
                    ,'fa_intra_rep' : None
                    ,'fa_intra_sol_xover4' : None
                    ,'lk_ball_wtd' : None
                    ,'fa_elec' : None
                    ,'pro_close' : None
                    ,'hbond_sr_bb' : None
                    ,'hbond_lr_bb' : None
                    ,'hbond_bb_sc' : None
                    ,'hbond_sc' : None
                    ,'dslf_fa13' : None
                    ,'omega' : None
                    ,'fa_dun' : None
                    ,'p_aa_pp' : None
                    ,'yhh_planarity' : None
                    ,'ref' : None
                    ,'rama_prepro' : None
                }
                filepath = os.path.join(root, file)
                pose = pose_from_pdb(filepath)
                chain_A = pose.split_by_chain(1)
                for key in scores:
                    scores[key] = sfxn.score_by_scoretype(chain_A,getattr(rosetta.core.scoring.ScoreType, key))
                scores['total_weighted_score'] = sum(scores[key] for key in scores if key != 'total_weighted_score')
                scores['filepath'] = filepath
                data.append(scores)

df = pd.DataFrame(data)

#output_name = directory.split('/')[-1] + ".csv"
output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_energies.csv"
df.to_csv(output_name, index=False, mode = 'w')

