import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from collections import defaultdict
import os
import argparse
from Bio import PDB
from Bio.Data import IUPACData
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Align import substitution_matrices
substitution_matrix = substitution_matrices.load('BLOSUM62')

AAs = list(IUPACData.protein_letters)

#parser = argparse.ArgumentParser(description="Process PDB files in a directory and log scores.")
#parser.add_argument('directory', type=str, help="The directory containing PDB files.")
#args = parser.parse_args()

base_model = get_structure("/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/base_structures/IL10_Mutant_model1.pdb")
base_coords, base_seq = get_residue_data(next(base_model.get_chains()))


filepaths = []
whole_seq_alignment = []
motif_alignment = []
non_motif_alignment = []
#base_seq = ''.join([base_seq[i] for i in list(range(24,50))+list(range(90,125))])
#base_coords = np.array([base_coords[i] for i in list(range(24,50))+list(range(90,125))])

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
                design = get_structure(os.path.join(root, file))
                design_coords, design_seq = get_residue_data(next(design.get_chains()))
                filepaths.append(os.path.join(root, file))
                #no gaps alignment
                whole_seq_alignment.append(pairwise2.align.globalds(base_seq[:len(design_seq)],design_seq,substitution_matrix, -9999999999, -9999999999)[0].score)
                motif_alignment.append(pairwise2.align.globalds(''.join([base_seq[i] for i in list(range(24,50))+list(range(90,125))]),''.join([design_seq[i] for i in list(range(24,50))+list(range(90,125))]),substitution_matrix, -9999999999, -9999999999)[0].score)
                inverse_motif = [i for i in range(len(design_seq)) if i not in list(range(24, 50)) + list(range(90, 125))]
                non_motif_alignment.append(pairwise2.align.globalds(''.join([base_seq[i] for i in inverse_motif]),''.join([design_seq[i] for i in inverse_motif]),substitution_matrix, -9999999999, -9999999999)[0].score)



df = pd.DataFrame({"filepath": filepaths,"whole_seq_alignment_blossum62": whole_seq_alignment,"motif_seq_alignment_blossum62": motif_alignment, "non_motif_seq_alignment_blossum62": non_motif_alignment})
output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/refolded_seq_align.csv"
df.to_csv(output_name, mode = 'w')


#pairwise2.align.globalds(''.join([base_seq[i] for i in inverse_motif]),''.join([base_seq[i] for i in inverse_motif]),substitution_matrix, -9999999999, -9999999999)[0].score