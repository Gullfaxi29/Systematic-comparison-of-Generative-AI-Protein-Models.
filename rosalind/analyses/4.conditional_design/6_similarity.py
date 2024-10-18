import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from collections import defaultdict
import os
import argparse

from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import PDB

#parser = argparse.ArgumentParser(description="Process PDB files in a directory and log scores.")
#parser.add_argument('directory', type=str, help="The directory containing PDB files.")
#args = parser.parse_args()

#base_model = get_structure("/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/base_structures/IL10_Mutant_model1.pdb")
base_model = get_structure("/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/base_structures/tev_monomer.pdb")
base_coords, base_seq = get_residue_data(next(base_model.get_chains()))
#base_seq = ''.join([base_seq[i] for i in list(range(24,50))+list(range(90,125))])
#base_coords = np.array([base_coords[i] for i in list(range(24,50))+list(range(90,125))])
for i, motif_coords in enumerate([list(range(27,33)),list(range(46,51)),list(range(139,152)),list(range(167,179)),list(range(211,221))]):
    print(motif_coords)
    print(i)
    base_model = get_structure("/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/base_structures/tev_monomer.pdb")
    base_coords, base_seq = get_residue_data(next(base_model.get_chains()))
    base_seq = ''.join([base_seq[i] for i in motif_coords])
    base_coords = np.array([base_coords[i] for i in motif_coords])
    alignments = defaultdict()
    #for directory in ["/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/chroma"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/protpardelle"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/protein_generator"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/refolded/chroma"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/refolded/protpardelle"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/refolded/protein_generator"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/evodiff"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/mpnn_solo"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected"
    #                    ,"/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected"]:
    for directory in ["/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/chroma"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/protpardelle"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/protein_generator"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/refolded/chroma"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/refolded/protpardelle"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/refolded/protein_generator"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/evodiff"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/mpnn_solo"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns/structures_selected"
                        ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/rfdiffusion/MPNN_redesigns_fixed/structures_selected"]:
        print(directory)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if ".pdb" in file:
                    design = get_structure(os.path.join(root, file))
                    design_coords, design_seq = get_residue_data(next(design.get_chains()))
                    #design_seq = ''.join([design_seq[i] for i in list(range(24,50))+list(range(90,125))])
                    #design_coords = np.array([design_coords[i] for i in list(range(24,50))+list(range(90,125))])
                    design_seq = ''.join([design_seq[i] for i in motif_coords])
                    design_coords = np.array([design_coords[i] for i in motif_coords])
                    #perform TMalign
                    align = tm_align(design_coords, base_coords, design_seq, base_seq)
                    tm_translation = align.t
                    tm_rotation = align.u.flatten()
                    tm_norm = np.array([align.tm_norm_chain1, align.tm_norm_chain2])
                    #Perform singular value decomposition to align and calculate RMSD
                    sup = SVDSuperimposer()
                    sup.set(design_coords, base_coords[:len(design_coords)])
                    sup.run()
                    rmsd = np.array([sup.get_rms()])
                    svd_rot, svd_tran = sup.get_rotran()
                    svd_rot = svd_rot.flatten()
                    #get the average 'b-factor' per atom per residue (in omegafold structures this is actually the confidence PLDDT)
                    res_b_factors = []
                    for aamodel in design:
                        for chain in aamodel:
                            print(chain)
                            for residue in chain:
                                # Extract B factor for each atom in the residue and calculate the average
                                b_factors = [atom.get_bfactor() for atom in residue if atom.get_bfactor() != ' ']
                                if b_factors:
                                    res_b_factors.append(sum(b_factors) / len(b_factors))
                            break #only want the first chain (monomer)
                    #Append result
                    alignments[os.path.join(root, file)] = np.concatenate([tm_norm, tm_translation, tm_rotation, rmsd, svd_tran, svd_rot, np.array([np.mean(res_b_factors)])])
    column_names = ['tm_norm_base','tm_norm_design','tm_t1','tm_t2','tm_t3','tm_u11','tm_u12','tm_u13','tm_u21','tm_u22','tm_u23','tm_u31','tm_u32','tm_u33','rmsd','svd_t1','svd_t2','svd_t3','svd_u11','svd_u12','svd_u13','svd_u21','svd_u22','svd_u23','svd_u31','svd_u32','svd_u33','omegafold_confidence']
    df = pd.DataFrame.from_dict(alignments, columns=column_names, orient='index')
    output_name = f"/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_struct_align_sub_motif_{i+1}.csv"
    df.to_csv(output_name, mode = 'w')
    

#output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_struct_align_whole.csv"
#output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_struct_align_motif.csv"
#output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_struct_align_whole.csv"








[base_coords[i] for i in list(range(24,50))+list(range(90,125))]