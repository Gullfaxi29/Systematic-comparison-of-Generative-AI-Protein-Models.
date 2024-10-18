import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import os
from Bio import SeqIO




uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'

foldseek = pd.DataFrame()
directory = "/scratch/alexb2/generative_protein_models/analyses/3.diversity_and_novelty/novelty_foldseek"
for filename in os.listdir(directory):
    if filename.startswith("len"):
        file_path = os.path.join(directory, filename)
        try:
            foldseek = pd.concat([foldseek, pd.read_csv(file_path,sep="\t", names = ["query","target","alntmscore","qtmscore","ttmscore","lddt","prob"])], axis = 0)
        except pd.errors.EmptyDataError:
            print(f'{file_path} Is empty')

foldseek.loc[:,'entity_id'] = foldseek.loc[:,'query'].str.extract(f'({uuid_pattern})', expand=False)
foldseek = foldseek.groupby(['entity_id'])['qtmscore'].agg(['max', 'min', 'mean', 'median']).reset_index(drop=False)
foldseek.columns = ['entity_id','tm_max','tm_min','tm_mean','tm_median']
foldseek = foldseek.loc[:,['entity_id','tm_max']]#,'tm_min','tm_mean','tm_median']]

maxcluster = pd.read_csv("/scratch/alexb2/generative_protein_models/analyses/3.diversity_and_novelty/diversity_maxcluster/combined_clusters.tsv", sep='\t')
maxcluster.loc[:,'entity_id'] = maxcluster.loc[:,'path'].str.extract(f'({uuid_pattern})', expand=False)
#maxcluster.loc[:,'model'] = maxcluster.path.str.split('/').str[-1].str.split('_').str[-1].str.split('.').str[0]
maxcluster = maxcluster.loc[:,['entity_id','cluster']]

generated = pd.read_csv('/scratch/alexb2/generative_protein_models/share/14_200_per_chain.csv')
generated.loc[:,'filename'] = generated.filepath.str.split('/').str[-1]
generated.loc[:,'model'] = generated.filename.str.split('_').str[0]
generated['entity_id'] = generated['filepath'].str.extract(f'({uuid_pattern})', expand=False)
generated.loc[:,'length'] = generated.chain_sequence.str.len().astype(int)
generated = generated.loc[:,['model','entity_id','length']]

pisces = []
for record in SeqIO.parse("./esm_embeddings/pisces.fasta","fasta"):
    pisces.append({'model': 'PISCES','entity_id': '_'.join(record.id.split('_')[-2:]), 'length': len(record.seq)})

pisces = pd.DataFrame(pisces)
pisces.drop_duplicates(inplace = True)

uni = []
for record in SeqIO.parse("./esm_embeddings/uniref50_subsample.fasta","fasta"):
    uni.append({'model': 'UniRef50','entity_id': record.id, 'length': len(record.seq)})

uni = pd.DataFrame(uni)

#Concat into full dataset
data = pd.concat([generated,pisces,uni])
data = data.merge(foldseek, on = 'entity_id', how='left')
data = data.merge(maxcluster, on = 'entity_id', how='left')

tsne_result = pd.read_csv("./esm_embeddings/gpu_tsne_result.csv",index_col=0)
data = pd.merge(tsne_result, data, how = 'left', on = 'entity_id')
data = data.loc[~data.model.isna(),:]
#data = data.loc[:,'tm_max'].fillna(0)


plt.rcParams.update({
    'axes.spines.top': False,        # Hide top spine
    'axes.spines.right': False,      # Hide right spine
    'axes.spines.left': False,        # Show left spine
    'axes.spines.bottom': False,      # Show bottom spine
    'axes.linewidth': 0.25,          # Set spine linewidth
    'axes.edgecolor': 'dimgrey',     # Set spine edge color
    'axes.facecolor': (0, 0, 0, 0) ,        # Set axes background color (if needed)
    'axes.grid': False,  # Disable grid lines on all plots
    'font.size': 6,          # Set default font size
    'axes.labelsize': 4,       # Font size for x and y labels
    'axes.titlesize': 7,       # Font size for titles
    'legend.fontsize': 6,      # Font size for legends
    'xtick.labelsize': 4,      # Font size for x-axis tick labels
    'ytick.labelsize': 4,       # Font size for y-axis tick labels
    'font.family': 'sans-serif',  # Set default font family to sans-serif
    'font.sans-serif': ['Arial', 'Helvetica', 'Calibri', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'],  # Specify preferred sans-serif fonts
    'xtick.direction': 'out',   # Set tick direction to outward
    'ytick.direction': 'out',
    'xtick.major.size': 3,      # Set major tick size
    'ytick.major.size': 3,
    'xtick.major.width': 0.5,   # Set major tick width
    'ytick.major.width': 0.5,
    'xtick.minor.visible': True,    # Show minor ticks on x-axis
    'ytick.minor.visible': True,    # Show minor ticks on y-axis
    'xtick.minor.size': 2,      # Set minor tick size
    'ytick.minor.size': 2,
    'xtick.minor.width': 0.25,   # Set minor tick width
    'ytick.minor.width': 0.25,
    'xtick.color': 'dimgrey',   # Set tick color
    'ytick.color': 'dimgrey',
    'xtick.labelcolor': 'black',    # Set tick label color
    'ytick.labelcolor': 'black'    # Set tick label color
})

colors = {'UniRef50': 'silver',
'PISCES': 'dimgrey',
'RFdiffusion': '#005cdf',
'Genie': '#43e4ff',
'ProteinSGM': '#586386',
'FoldingDiff': '#7373ae',
'FrameDiff': '#30aa8f',
'Chroma': '#809a2d',
'Protpardelle': '#ffc73c',
'Protein-Generator': '#a4837b',
'EvoDiff': '#93355c',
'RITA': '#cf4f00',
'ProGen2': '#e44fff',
'ProtGPT2': '#9e0049',
'ESM-Design': '#ff7bb9',
}

data['color'] = data['model'].map(colors)

#tsne_result['length_color'] = np.interp(tsne_result.length, (tsne_result.length.min(), tsne_result.length.max()),(0.3, 1))
#tsne_result['alpha'] = 1 #- np.interp(tsne_result.tm_mean, (tsne_result.tm_mean.min(), tsne_result.tm_mean.max()),(0.1, 1))
data = data.sample(frac=1).reset_index(drop=True) #shuffle
background = data.model.isin(['PISCES','UniRef50'])
foreground = ~data.model.isin(['PISCES','UniRef50'])
gen_length = (data.length <= 200) & (data.length >= 14)
#If there were no hits against the generated sequence from foldseek, we'll say the max TM-Score is 0
#data.loc[foreground,'tm_max'] = data.loc[foreground,'tm_max'].fillna(0)

aspect_ratio = 8.3 / 11.7 #A4 aspect ratio
figure_height = 11.7 #Update this for different figure scale, 11.69 for full A4.
with PdfPages("figure_3.pdf") as pdf:
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=300)
    _= fig.text(0.02, 0.970, 'A)', fontsize=10, ha='left', va='center')
    _= fig.text(0.02, 0.690, 'B)', fontsize=10, ha='left', va='center')
    _= fig.text(0.02, 0.268, 'C)', fontsize=10, ha='left', va='center')
    alpha = 0.2
    gs = GridSpec(3,1, figure=fig, left =0.01, bottom = 0.03, right=0.99,top = 0.98,  hspace = 0.09, height_ratios = [1,1.25,0.75])
    gs_A = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[0,0], wspace = 0.01)
    #Main TSNE
    background_tsne = fig.add_subplot(gs_A[0,0]) 
    _= background_tsne.scatter(data.loc[data.model == 'UniRef50','x'], data.loc[data.model == 'UniRef50','y'], c = data.loc[data.model == 'UniRef50','color'], alpha = alpha-0.1, marker = ',', s=1, edgecolor='none', rasterized=True)
    _= background_tsne.scatter(data.loc[data.model == 'PISCES','x'], data.loc[data.model == 'PISCES','y'], c = data.loc[data.model == 'PISCES','color'], alpha = alpha, marker = ',', s=1, edgecolor='none', rasterized=True)
    _ = (lambda ax: (ax.set_title('"Protein Universe"',pad = 0, fontsize = 7),ax.set_xticks([]), ax.set_xticklabels([]), ax.set_yticks([]), ax.set_yticklabels([])))(background_tsne)
    foreground_tsne = fig.add_subplot(gs_A[0,1])
    _= foreground_tsne.scatter(data.loc[foreground,'x'], data.loc[foreground,'y'],color=data.loc[foreground,'color'],edgecolor='none', alpha = alpha, marker = ',',s=1, rasterized=True)
    _ = (lambda ax: (ax.set_title('Generated',pad = 0, fontsize = 7), ax.set_xticks([]), ax.set_xticklabels([]), ax.set_yticks([]), ax.set_yticklabels([])))(foreground_tsne)
    #Individual Models
    gs_B = gridspec.GridSpecFromSubplotSpec(3,5, subplot_spec=gs[1,0], wspace = 0.05,hspace = 0.1)
    for i, model in enumerate(colors.keys()):
        model_ax = fig.add_subplot(gs_B[i//5,i%5])
        _= model_ax.scatter(data.loc[(data.model == model) & gen_length,'x'], data.loc[(data.model == model)  & gen_length,'y'], color=data.loc[(data.model  == model)  & gen_length,'color'], alpha = alpha - (i == 0)*0.17, s = 1, marker = ',',edgecolor='none',rasterized=True)
        _ = (lambda ax: (ax.set_title(model,pad=2), ax.set_xticks([]), ax.set_xticklabels([]), ax.set_yticks([]), ax.set_yticklabels([]), ax.set_xlim(tsne_result.x.min(),tsne_result.x.max()), ax.set_ylim(data.y.min(),data.y.max())))(model_ax)
        if i<=1: _= model_ax.set_title(model + " (Length 14-200)",pad=2)
        _= [spine.set_visible(True) for spine in model_ax.spines.values()]
    #
    gs_bottom = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=gs[2,0], wspace = 0.2,hspace = 0.4, width_ratios = [0.01,1,1.2], height_ratios = [1,0.5])
    #clustering
    clusters_ax = fig.add_subplot(gs_bottom[:,1])
    clustering_data = data.loc[foreground,:].groupby(['model','color'])['cluster'].max().reset_index()
    clustering_data['model'] = pd.Categorical(clustering_data['model'], categories=list(colors.keys())[2:], ordered=True)
    _= clusters_ax.bar(clustering_data.sort_values('model').model,clustering_data.sort_values('model').cluster, color = clustering_data.sort_values('model').color)
    for p in clusters_ax.patches:
        _= clusters_ax.text(p.get_x() + p.get_width() / 2,p.get_height(),int(p.get_height()),ha='center', va='bottom',fontsize = 4) 
    _= (lambda ax: (
        #ax.set_title('C)',pad = 8, fontsize =10,loc='left')
        ax.set_ylabel('# Structural Clusters (MaxCluster)', fontsize = 6)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.tick_params(axis='x', which='both', size=0)
        #,ax.yaxis.tick_right()
        #,ax.yaxis.set_label_position('right')
        ,ax.tick_params(axis='y',labelsize=5)
        ,ax.tick_params(axis='x', which='major', labelrotation=30, labelsize=4)
        ))(clusters_ax)
    #foldseek (novelty)
    novelty_ax = fig.add_subplot(gs_bottom[0,2])
    novelty = data.loc[foreground].copy()
    novelty['model'] = pd.Categorical(novelty['model'], categories=list(colors.keys())[2:], ordered=True)
    _= sns.violinplot(x='model',y='tm_max',ax=novelty_ax,data=novelty.sort_values('model'), hue = 'model', palette=colors, linewidth=0.3,cut=0)
    _= (lambda ax: (
        ax.set_title('D)',pad = 8, fontsize =10,loc='left')
        ,ax.set_ylabel('TM-Score (Closest Hit - FoldSeek, ESM-Atlas30)',fontsize = 6)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.set_xlabel('')
        ,ax.tick_params(axis='x', which='both', size=0)
        ,ax.tick_params(axis='x', which='major', labelrotation=30,labelsize = 4)
        ,ax.tick_params(axis='y',labelsize=5)
        ))(novelty_ax)
    no_hits_ax = fig.add_subplot(gs_bottom[1,2])
    no_hits = novelty.loc[novelty.tm_max.isna(),:].groupby(['model','color'])['entity_id'].count().reset_index()
    hits = novelty.loc[~novelty.tm_max.isna(),:].groupby(['model','color'])['entity_id'].count().reset_index()
    hits.columns = ['model','color','counts']
    hits = hits.loc[hits.counts > 0]
    no_hits.columns = ['model','color','counts']
    no_hits = no_hits.loc[no_hits.counts > 0]
    no_hits['perc'] = 100 * (no_hits.counts / (no_hits.counts + hits.counts))
    no_hits['perc'] = no_hits['perc'].round(1)
    #_= no_hits_ax.bar(no_hits.sort_values('model').model, no_hits.sort_values('model').counts, color = no_hits.sort_values('model').color)
    _= sns.barplot(x='model',y='perc',ax=no_hits_ax,data=no_hits.sort_values('model'), hue = 'model', palette=colors)
    for p in no_hits_ax.patches:
        _= no_hits_ax.text(p.get_x() + p.get_width() / 2,p.get_height(),f'{round(p.get_height(),1)}%',ha='center', va='bottom',fontsize = 4) 
    _= (lambda ax: (
        #ax.set_title('E)',pad = 0, fontsize =10)
        ax.set_ylabel('% (No Hit)',fontsize = 6)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.set_xlabel('')
        ,ax.tick_params(axis='x', which='both', size=0)
        #,ax.tick_params(axis='x', which='major', labelrotation=30,labelsize = 4)
        ,ax.set_xticks([])
        ,ax.tick_params(axis='y',labelsize=5)
        ))(no_hits_ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    plt.close('all')









"""
Depreciated


maxcluster = pd.DataFrame()
for root, dirs, files in os.walk("/scratch/alexb2/generative_protein_models/analyses/figure3/diversity"):
    pattern = os.path.join(root, 'clusters.tsv')
    matched_files = glob.glob(pattern)
    for file in matched_files:
        maxcluster = pd.concat([maxcluster, pd.read_csv(file,sep="\t", names = ["item","cluster","path"])], axis = 0)

umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1)
start = time.time()
embedding = umap_model.fit(X_standardized)
end = time.time()
print(f"UMAP took {end - start:.2f} seconds")

with open('umap_model.pkl', 'wb') as file:
    pickle.dump(umap_model, file)

with open('umap_model.pkl', 'rb') as file:
    umap_model = pickle.load(file)

umap_result = pd.DataFrame(umap_model.transform(X), columns = ['x','y'])
umap_result.loc[:,'model'] = data.loc[:,'model']
umap_result.loc[:,'entity_id'] = data.loc[:,'entity_id']
umap_result.loc[:,'length'] = data.loc[:,'length']
umap_result.loc[umap_result.model == 'ProGen2','model'] = 'progen2'
umap_result.loc[umap_result.model == 'ESM','model'] = 'esmdesign'
umap_result.loc[umap_result.model == 'RFdiffusion','model'] = 'rfdiffusion'
umap_result.to_csv("test_umap_result.csv")


pisces = pd.read_csv('/scratch/alexb2/generative_protein_models/analyses/feasibility/pisces_reference_per_chain.csv')
pisces['model'] = 'pisces'
pisces['entity_id'] = pisces['filepath'].str.split('/').str[-1].str.split('.').str[0] + "_chain" + pisces.chain_no.astype(str)
pisces.loc[:,'length'] = pisces.chain_sequence.str.len().astype(str)
pisces['id'] = pisces.model + "_len" + pisces['length'] + "_" + pisces['entity_id']

pisces_seqs = []
for i, row in pisces.iterrows():
    pisces_seqs.append(SeqRecord(Seq(row['chain_sequence']),id=row['id'],description=""))
SeqIO.write(pisces_seqs, "pisces.fasta","fasta")

    for i in range(gs.nrows*gs.ncols):
        bound = fig.add_subplot(gs[i //gs.nrows,i % gs.ncols])
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor='dimgrey', alpha=0.5, linestyle = (0, (5, 7)), facecolor='none')
        _= bound.add_patch(rect)
        _= bound.axis('off')



tsne_result['source'] = tsne_result['source'].map({'pdb':'PDB', 'framediff': 'framediff', 'RFdiffusion_150it': 'rfdiffusion', 'chroma':'chroma', 'rita_xl':'rita','protpardelle':'protpardelle', 'genie_swissprot_l_256':'genie', 'evodiff_OA_DM_640M': 'evodiff','ESM_Design': 'esmdesign', 'foldingdiff': 'foldingdiff', 'proteinsgm': 'proteinsgm', 'protein_generator': 'proteingenerator','protgpt2': 'protgpt', 'ProGen2': 'progen2'})

custom_palette = ['lightgrey'] + sns.color_palette('tab10')[1:]
alpha = tsne_result['source'].replace(tsne_result['source'].value_counts().index[0], 0.1).replace(tsne_result['source'].value_counts().index[1:], 0.9)
size = tsne_result['source'].replace(tsne_result['source'].value_counts().index[0], 3).replace(tsne_result['source'].value_counts().index[1:], 5)

sns.set(style='whitegrid')
sns.scatterplot(data=tsne_result.loc[tsne_result.source == 'PDB'], x='x', y='y', hue='source', palette= custom_palette, s=4, alpha = 0.4, marker='o',linewidth=0.01, edgecolor='grey')
sns.scatterplot(data=tsne_result.loc[tsne_result.source != 'PDB'], x='x', y='y', hue='source', s=8, alpha = alpha, marker='o',linewidth=0.1, edgecolor='white')
sns.despine(left=True,bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
plt.savefig('output_plot.png', bbox_inches='tight', dpi=300)
plt.clf()


#pca = PCA(n_components=2)
#pca_result = pca.fit_transform(X_standardized)
#umap = UMAP(n_components=2)
#umap_result = umap.fit_transform(X_standardized)


fig, axes = plt.subplots(1, 2, figsize=(18, 6))


marker_size = 0.5
axes[0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s = marker_size)
axes[0].set_title('PCA')

axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, s = marker_size)
axes[1].set_title('t-SNE')

plt.savefig('output_plot.pdf')

plt.savefig('output_plot.png')



axes[1].scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.5)
axes[1].set_title('UMAP')
"""
