import numpy as np
import pandas as pd
import os
import shutil
import re
uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'

def list_files(directory):
    if directory:
        return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    else:
        return []

sc = pd.read_csv('/scratch/alexb2/generative_protein_models/share/SC_Scoring_20240920_180051.csv')
scores = sc.loc[:,['model','entity_id','design_file','tm_norm_aa','rmsd','omegafold_confidence']]
scores = scores.loc[scores.groupby('entity_id')['tm_norm_aa'].idxmax()].sort_values(by=['model','entity_id'])
scores.to_csv("/scratch/alexb2/generative_protein_models/share/selected_designs.csv",index=False)

final_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/final_designs/"
discarded_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/discarded_redesigns/"
design_paths = list_files(final_designs_dir) + list_files(discarded_designs_dir)
design_paths = [file_path for file_path in design_paths if re.search(uuid_pattern, file_path).group(0) in scores.entity_id.unique()]

for path in design_paths:
    if os.path.basename(path) in scores.design_file.unique():
        new_path = final_designs_dir + os.path.basename(path)
        if new_path == path:
            print(f"{os.path.basename(path)} Do nothing")
        else:
            print(f"{os.path.basename(path)} Move to main")
            shutil.move(path, new_path)
    else:
        new_path = discarded_designs_dir + os.path.basename(path)
        if new_path == path:
            print(f"{os.path.basename(path)} Do nothing")
        else:
            print(f"{os.path.basename(path)} Move to discard")
            shutil.move(path, new_path)
