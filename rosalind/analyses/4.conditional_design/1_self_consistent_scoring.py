import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from collections import defaultdict
import os
import argparse
import datetime

from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import PDB

#bb_dir = "/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/backbones"
bb_dir = "/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/backbones"
bb_filepaths = [os.path.join(bb_dir, file) for file in os.listdir(bb_dir) if os.path.isfile(os.path.join(bb_dir, file))]

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

failed = []
alignments = defaultdict(dict)
for backbone_file in bb_filepaths:
    #matching_design_filepaths = [design_file for design_file in aa_filepaths if os.path.basename(design_file).split("_design")[0] + ".pdb" == os.path.basename(backbone_file)]
    matching_design_filepaths = [design_file for design_file in aa_filepaths if os.path.basename(design_file).split('_')[-2] == os.path.basename(backbone_file).split('_')[-1].split('.')[-2]]
    if len(matching_design_filepaths) > 0: print(backbone_file)
    for design_file in matching_design_filepaths:
        print(design_file)
        backbone = get_structure(backbone_file)
        bb_coords, bb_seq = get_residue_data(next(backbone.get_chains()))
        design = get_structure(design_file)
        aa_coords, aa_seq = get_residue_data(next(design.get_chains()))
        if len(aa_seq) == len(bb_seq):
            align = tm_align(bb_coords, aa_coords, bb_seq, aa_seq)
            tm_translation = align.t
            tm_rotation = align.u.flatten()
            tm_norm = np.array([align.tm_norm_chain1, align.tm_norm_chain2])
            #Perform singular value decomposition to align and calculate RMSD
            sup = SVDSuperimposer()
            sup.set(bb_coords, aa_coords)
            sup.run()
            rmsd = np.array([sup.get_rms()])
            svd_rot, svd_tran = sup.get_rotran()
            svd_rot = svd_rot.flatten()
            #get the average 'b-factor' per atom per residue (in omegafold structures this is actually the confidence PLDDT)
            res_b_factors = []
            for aamodel in design:
                for chain in aamodel:
                    for residue in chain:
                        # Extract B factor for each atom in the residue and calculate the average
                        b_factors = [atom.get_bfactor() for atom in residue if atom.get_bfactor() != ' ']
                        if b_factors:
                            res_b_factors.append(sum(b_factors) / len(b_factors))
            #Append result
            alignments[os.path.basename(backbone_file).split('_')[0]][os.path.basename(design_file)] = np.concatenate([tm_norm, tm_translation, tm_rotation, rmsd, svd_tran, svd_rot, np.array([np.mean(res_b_factors)])])
            print("------------")
        else:
            failed.append(backbone_file)

        
    

flattened_dict = {}
for key, sub_dict in alignments.items():
    for subkey, value in sub_dict.items():
        flattened_dict[(key, subkey)] = value

column_names = ['tm_norm_bb','tm_norm_aa','tm_t1','tm_t2','tm_t3','tm_u11','tm_u12','tm_u13','tm_u21','tm_u22','tm_u23','tm_u31','tm_u32','tm_u33','rmsd','svd_t1','svd_t2','svd_t3','svd_u11','svd_u12','svd_u13','svd_u21','svd_u22','svd_u23','svd_u31','svd_u32','svd_u33','omegafold_confidence']
results = pd.DataFrame(flattened_dict.values(), index=pd.MultiIndex.from_tuples(flattened_dict.keys()), columns=column_names)
results.reset_index(inplace = True)
results.columns = ["model","design_file"] + list(results.columns[2:])
# Reset index if needed
results.model.unique()
#output_path = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_mpnn_consistency_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
#output_path = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_mpnn_fixed_consistency_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
#output_path = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_mpnn_consistency_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
output_path = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_mpnn_fixed_consistency_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
results.to_csv(output_path, index = False)