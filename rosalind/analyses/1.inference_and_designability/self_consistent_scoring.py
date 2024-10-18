import os
import argparse
import datetime
import re

from collections import defaultdict
import numpy as np
import pandas as pd

from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from Bio.SVDSuperimposer import SVDSuperimposer

def list_files(directory):
    if directory:
        return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    else:
        return []

def get_aa_coords(structure):
    #Filtering out hydrogens as they are only in ProteinGenerator outputs and not refolds
    return np.array([atom.get_coord()
                    for model in structure
                    for chain in model
                    for residue in chain
                    for atom in residue
                    if not (atom.get_name().startswith('H') or atom.get_name()[0].isdigit() and atom.get_name()[1] == 'H')]) 

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
    parser.add_argument("--final_designs", type=str, help="")
    parser.add_argument("--initial_designs", type=str, help="")
    parser.add_argument("--discarded_designs", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    args = parser.parse_args()
    final_designs_dir = args.final_designs
    init_designs_dir = args.initial_designs
    discarded_designs_dir = args.discarded_designs
    output_path = args.output_path
    #FOR INTERACTIVE
    #final_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/final_designs"
    #init_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/backbones"
    #final_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/refolds"
    #init_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/final_designs"
    #discarded_designs_dir = "/scratch/alexb2/generative_protein_models/raw_data/unconditional_generation/14-200/discarded_redesigns"
    #discarded_designs_dir = ""
    #output_path = "/scratch/alexb2/generative_protein_models/share/SC_Scoring"
    final_designs = list_files(final_designs_dir) + list_files(discarded_designs_dir)
    init_designs = list_files(init_designs_dir)
    failed = []
    alignments = defaultdict(dict)
    uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
    #
    for init_design_path in init_designs:
        print(f'- {os.path.basename(init_design_path)}')
        for final_design_path in [file_path for file_path in final_designs if re.search(uuid_pattern, file_path).group(0) == re.search(uuid_pattern, init_design_path).group(0)]:
            print(f'\t- {os.path.basename(final_design_path)}')
            init_design = get_structure(init_design_path)
            init_coords, init_seq = get_residue_data(next(init_design.get_chains()))
            final_design = get_structure(final_design_path)
            final_coords, final_seq = get_residue_data(next(final_design.get_chains()))
            if len(final_seq) == len(init_seq):
                #Calculate the TM-Align Score
                align = tm_align(init_coords, final_coords, init_seq, final_seq)
                tm_translation = align.t
                tm_rotation = align.u.flatten()
                tm_norm = np.array([align.tm_norm_chain1, align.tm_norm_chain2])
                #Perform singular value decomposition to align and calculate RMSD
                # If we are doing refolds, need to use all atom coords rather than just backbone carbons
                if 'refold' in final_design_path: 
                    init_coords = get_aa_coords(init_design)
                    final_coords = get_aa_coords(final_design)
                sup = SVDSuperimposer()
                sup.set(init_coords, final_coords)
                sup.run()
                rmsd = np.array([sup.get_rms()])
                svd_rot, svd_tran = sup.get_rotran()
                svd_rot = svd_rot.flatten()
                #get the average 'b-factor' per atom per residue (in omegafold structures this is actually the confidence PLDDT)
                res_b_factors = get_b_factors(final_design)
                #Append result
                model = os.path.basename(init_design_path).split('_')[0]
                alignments[model][os.path.basename(final_design_path)] = np.concatenate([tm_norm, tm_translation, tm_rotation, rmsd, svd_tran, svd_rot, np.array([np.mean(res_b_factors)])])
            else:
                print(f'Final design length does not match initial - {final_design_path}')
                failed.append(final_design_path)
            
        
    flattened = {(key, subkey): value for key, sub_dict in alignments.items() for subkey, value in sub_dict.items()}
    column_names = ['tm_norm_bb','tm_norm_aa','tm_t1','tm_t2','tm_t3','tm_u11','tm_u12','tm_u13','tm_u21','tm_u22','tm_u23','tm_u31','tm_u32','tm_u33','rmsd','svd_t1','svd_t2','svd_t3','svd_u11','svd_u12','svd_u13','svd_u21','svd_u22','svd_u23','svd_u31','svd_u32','svd_u33','omegafold_confidence']
    results = pd.DataFrame(flattened.values(), index=pd.MultiIndex.from_tuples(flattened.keys()), columns=column_names)
    results.reset_index(inplace = True)
    results.columns = ["model","design_file"] + list(results.columns[2:])
    results['entity_id'] = results.design_file.str.extract(f'({uuid_pattern})', expand=False)
    results.model.unique()
    output_path = "/scratch/alexb2/generative_protein_models/share/SC_Scoring"
    results.to_csv(output_path + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", index = False)
    print('These files failed to be scored: ')
    print(failed)



if __name__ == "__main__":
    main()
