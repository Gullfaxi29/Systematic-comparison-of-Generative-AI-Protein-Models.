import pandas as pd
import os
from Bio import SeqIO

from pyrosetta import *
from collections import Counter
init()
directories = ["/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/chroma"
                ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/protein_generator"
                ,"/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/protpardelle"]
records= []
for directory in directories:
    for file in os.listdir(directory):
        if file.endswith(".pdb"):
            print(os.path.join(directory, file))
            pose = pose_from_pdb(os.path.join(directory, file))
            num_chains = pose.num_chains()
            if num_chains > 1: 
                print(f" Too many chains: {num_chains}")
                continue
            chain = pose.split_by_chain(1)
            id = file.split('.')[0]
            sequence = chain.sequence()
            record = SeqIO.SeqRecord(seq=sequence, id=id, description="", name="")
            records.append(record)

with open('/scratch/alexb2/generative_protein_models/raw_data/tev_scaffolding/refolds_tev.fa', 'w',) as f:
    SeqIO.write(records, f, 'fasta')




