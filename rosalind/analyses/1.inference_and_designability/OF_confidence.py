import os
import argparse
import datetime
import re

from collections import defaultdict
import numpy as np
import pandas as pd

from tmtools.io import get_structure


def list_files(directory):
    if directory:
        return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    else:
        return []

def get_b_factors(structure):
    return [sum(b_factors) / len(b_factors)
                for model in structure
                for chain in model
                for residue in chain
                if (b_factors := [atom.get_bfactor() for atom in residue if atom.get_bfactor() != ' '])
                ]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    args = parser.parse_args()
    designs_dir = args.designs
    output_path = args.output_path
    #FOR INTERACTIVE
    #final_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/final_designs"
    #init_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/backbones"
    #final_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/refolds"
    #init_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/final_designs"
    #discarded_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/discarded_redesigns"
    #discarded_designs_dir = ""
    #output_path = "/scratch/alexb2/generative_protein_models/share/SC_Scoring"
    designs = list_files(designs_dir)
    alignments = defaultdict(dict)
    uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
    #
    for design_path in designs:
        print(f'\t- {os.path.basename(design_path)}')
        design = get_structure(design_path)
        res_b_factors = get_b_factors(design)
        #Append result
        model = os.path.basename(design_path).split('_')[0]
        alignments[model][os.path.basename(design_path)] = np.concatenate([np.array([np.mean(res_b_factors)])])
            
        
    flattened = {(key, subkey): value for key, sub_dict in alignments.items() for subkey, value in sub_dict.items()}
    column_names = ['omegafold_confidence']
    results = pd.DataFrame(flattened.values(), index=pd.MultiIndex.from_tuples(flattened.keys()), columns=column_names)
    results.reset_index(inplace = True)
    results.columns = ["model","design_file"] + list(results.columns[2:])
    results['entity_id'] = results.design_file.str.extract(f'({uuid_pattern})', expand=False)
    results.model.unique()
    output_path = "/scratch/alexb2/generative_protein_models/share/OF_Confidence"
    results.to_csv(output_path + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", index = False)



if __name__ == "__main__":
    main()
()