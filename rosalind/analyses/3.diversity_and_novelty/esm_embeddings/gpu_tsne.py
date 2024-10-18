#This script was run on a local pc with a GPU
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import pickle
import os
import glob
from Bio import SeqIO
from tsnecuda import TSNE
import gc # using to save memory in my WSL session (otherwise the datasets are sometimes too large and crashes occur)

pdb = pd.read_csv("./pisces_embeddings.csv", index_col = 0, header = None)
pdb.loc[:,'model'] = 'pisces_pdb'
pdb.loc[:,'entity_id'] = pdb.index.str.split('_').str[-2:].str.join('_')
pdb.loc[:,'length'] = pdb.index.str.split('_').str[1].str.split('len').str[-1].astype(int)
#pdb = pdb.loc[pdb.length >= 14,:]
pdb = pdb.reset_index(drop=True)
gc.collect()

uni = pd.read_csv("./uniref50_subsample_embeddings.csv", index_col = 0, header = None, dtype = {1: 'object', **{i: 'float32' for i in range(2, 1281)}})
uni.loc[:,'model'] = 'uniref50'
uni.loc[:,'entity_id'] = uni.index.str.split(' ').str[0]
lengths = {}
for record in SeqIO.parse("./uniref50_subsample.fasta","fasta"):
    lengths[record.id] = len(record.seq)
uni['length'] = uni['entity_id'].apply(lambda x: lengths.get(x.split(' ')[0], None))
del lengths; gc.collect()
uni.drop(uni[uni.length < 14].index, inplace=True)
uni = uni.reset_index(drop=True)

gen = pd.read_csv("./generated_seqs_14_200_embeddings.csv", index_col = 0, header = None)
gen.loc[:,'model'] = gen.index.str.split('_').str[0]
uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
gen.loc[:,'entity_id'] = gen.index.str.extract(f'({uuid_pattern})', expand=False)
#gen.loc[:,'length'] = gen.index.to_series().str.extract(r'len(\d+)').astype(int)
#gen =  gen[~((gen['model'] == 'foldingdiff') & (gen['length'] > 128))] # removing the longer foldingdiff designs that were actually just 128
gen = gen.reset_index(drop=True)


data = pd.concat([gen,uni,pdb],ignore_index=True)
del gen, uni, pdb; gc.collect()
X = pd.DataFrame()
#popping, again to save memory
for i in range(1, 1281):
    X[i] = data.pop(i)
X_scaled = StandardScaler().fit_transform(X)
del X; gc.collect()
tsne = TSNE(n_components=2, perplexity=50, learning_rate=200).fit_transform(X_scaled)

tsne_result = pd.DataFrame(tsne, columns = ['x','y'])
tsne_result.loc[:,'entity_id'] = data.loc[:,'entity_id']
tsne_result.to_csv("gpu_tsne_result.csv")

