from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

#This code VVVV was executed on Rosalind to create the appropriate fastas to embed on a local machine with a GPU

"""
#Make a fasta file for all our generated sequences
generated = pd.read_csv('/scratch/alexb2/generative_protein_models/share/14_200_per_chain.csv')
uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
generated['entity_id'] = generated['filepath'].str.extract(f'({uuid_pattern})', expand=False)
generated = generated.loc[:,['entity_id','chain_sequence']]

records = []
for index, row in generated.iterrows():
    # Create a SeqRecord for each sequence
    record = SeqRecord(
        Seq(row['chain_sequence']),
        id=f"{row['entity_id']}",
        description=f"{row['entity_id']}"
    )
    records.append(record)

with open('generated_seqs_14_200.fasta', 'w') as fasta_file:
    SeqIO.write(records, fasta_file, 'fasta')
"""

"""
#Reservoir sampling of 1% of uniref50 
bins = np.array([13,51,101,151,201,251,301,351,401,451,501,551,601,651,701,751,801,851,901,951,1001,1101,1201,1301,1401,1501,1601,1701,1801,1901,2001,2101,2201,2301,2401,2501,34350])
swissprot_reviewed = np.array([0,9968,43534,59796,59574,58452,52413,52846,45901,37706,30572,22287,15830,13156,9403,7870,5700,4889,5301,4109,3007,4124,2897,2207,2070,1675,834,642,587,503,395,272,386,340,234,195,1462])
bin_ratios = swissprot_reviewed / np.sum(swissprot_reviewed)
sample_size = 627598 #1% of uniref50.fasta sequences
sample_sizes = bin_ratios * sample_size 
samples = [[] for _ in range(len(sample_sizes))]
seen = [0]*len(sample_sizes)
with open("uniref50.fasta", 'r') as file:
    for i, record in enumerate(SeqIO.parse(file, 'fasta')):
        length = len(record.seq)
        bin_idx = np.searchsorted(bins, length, side='right')
        if (bin_idx == 0) or (bin_idx == len(sample_sizes)): continue
        if len(samples[bin_idx]) < sample_sizes[bin_idx]:
            samples[bin_idx].append(record)
        else:
            # Randomly replace elements in the reservoir
            j = random.randint(0, seen[bin_idx])
            if j < sample_sizes[bin_idx]:
                samples[bin_idx][j] = record
        seen[bin_idx] = seen[bin_idx] + 1
sample = [protein for bin in samples for protein in bin]

with open('uniref50_subsample.fasta', "a") as output_handle:
    SeqIO.write(sample, output_handle, "fasta")
"""

#This code VVVV was executed on a local machine with a GPU

import pathlib
import torch
import pandas as pd
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import json

#CHANGE this to the appropriate file
sub_file = "generated_seqs_14_200.fasta"

fasta_file = pathlib.Path("../" + sub_file)
toks_per_batch = 4096*4 #4096

dataset = FastaBatchedDataset.from_file(fasta_file)

batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
with open("../" + sub_file.split(".")[0]+"_batches.json", "w") as f:
    json.dump(batches, f)

torch.set_default_tensor_type(torch.cuda.FloatTensor)
model_location = "esm2_t33_650M_UR50D"
truncation_seq_length = 1000
include = "mean"
repr_layers = [-1]

model, alphabet = pretrained.load_model_and_alphabet(model_location)
model.eval()
if isinstance(model, MSATransformer):
    raise ValueError(
        "This script currently does not handle models with MSA input (MSA Transformer)."
    )

if torch.cuda.is_available():
    model = model.cuda()
    print("Transferred model to GPU")

with open("../" + sub_file.split(".")[0]+"_batches.json", "r") as f:
    batches = json.load(f)

data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
)
print(f"Read {fasta_file} with {len(dataset)} sequences")

assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        df = pd.DataFrame()
        print(
            f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
        )
        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)
        out = model(toks, repr_layers=repr_layers, return_contacts=False)
        logits = out["logits"].to(device="cpu")
        representations = {
            layer: t.to(device="cpu") for layer, t in out["representations"].items()
        }
        for i, label in enumerate(labels):
            result= {"Label": label}
            truncate_len = min(truncation_seq_length, len(strs[i]))
            result["mean_representations"] = {
                layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                for layer, t in representations.items()}
            df = df.append(pd.Series(result["mean_representations"][33].numpy(), name = label))
        df.to_csv("../" + sub_file.split(".")[0]+"_embeddings.csv", mode="a", header=False, index_label="id")
        with open("../" + sub_file.split(".")[0]+"_batches.json", "w") as f:
          json.dump(batches[batch_idx+1:], f)