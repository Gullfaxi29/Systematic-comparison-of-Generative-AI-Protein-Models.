import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from collections import defaultdict
import os
import argparse
from Bio import PDB
from Bio.Data import IUPACData
from collections import Counter

AAs = list(IUPACData.protein_letters)

enrichment_data = []
filepath_data = []
sequence_data = []
#parser = argparse.ArgumentParser(description="Process PDB files in a directory and log scores.")
#parser.add_argument('directory', type=str, help="The directory containing PDB files.")
#args = parser.parse_args()

#base_model = get_structure("/scratch/alexb2/generative_protein_models/raw_data/il10_scaffolding/base_structures/IL10_Mutant_model1.pdb")
base_model = get_structure("/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/base_structures/tev_monomer.pdb")
base_coords, base_seq = get_residue_data(next(base_model.get_chains()))

enrichment_data.append(Counter({key: value / len(base_seq) for key, value in Counter(base_seq).items()}))

filepath_data.append("Base Monomer Model")
sequence_data.append(base_seq)
#base_seq = ''.join([base_seq[i] for i in list(range(24,50))+list(range(90,125))])
#base_coords = np.array([base_coords[i] for i in list(range(24,50))+list(range(90,125))])

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
                enrichment_data.append(Counter({key: value / len(design_seq) for key, value in Counter(design_seq).items()}))
                filepath_data.append(os.path.join(root, file))
                sequence_data.append(design_seq)
            #design_seq = ''.join([design_seq[i] for i in list(range(24,50))+list(range(90,125))])
            #design_coords = np.array([design_coords[i] for i in list(range(24,50))+list(range(90,125))])


df = pd.DataFrame(enrichment_data,index = [filepath_data,sequence_data])
df.index.names = ['filepath','sequence']
#output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/il10/il10_AA_enrichment.csv"
output_name = "/scratch/alexb2/generative_protein_models/analyses/conditional_design/tev/tev_AA_enrichment.csv"
df.to_csv(output_name, mode = 'w')