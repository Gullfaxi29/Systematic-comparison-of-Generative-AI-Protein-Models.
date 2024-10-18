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
foldseek_raw = foldseek.copy(deep = True)
foldseek_raw = foldseek_raw.loc[~foldseek.entity_id.isna(),:]
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
foldseek_raw = foldseek_raw.merge(data.loc[:,['entity_id','model','length']], on = 'entity_id', how = 'left')
#data = pd.concat([generated])
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
    'axes.titlesize': 6,       # Font size for titles
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
data = data.loc[(data.length <= 200) & (data.length >= 14),:]
data['length_color'] = np.interp(data.length, (data.length.min(), data.length.max()),(0, 1))
data['tm_max_color'] = np.interp(data.tm_max, (data.tm_max.min(), data.tm_max.max()),(0, 1))
data['cluster_color'] = np.interp(data.cluster, (data.cluster.min(), data.cluster.max()),(0, 1))

cluster_sizes = data.groupby(['model','color','cluster'])['entity_id'].count().reset_index().rename(columns={'entity_id': 'cluster_size'})
cluster_sizes['model'] = pd.Categorical(cluster_sizes['model'], categories=list(colors.keys())[2:], ordered=True)
foldseek_raw['model'] = pd.Categorical(foldseek_raw['model'], categories=list(colors.keys())[2:], ordered=True)

aspect_ratio = 8.3 / 11.7 #A4 aspect ratio
figure_height = 11.7 #Update this for different figure scale, 11.69 for full A4.


with PdfPages("supplementary_figures_3.pdf") as pdf:
    fig= plt.figure(figsize=(8.27,11.7/2),dpi=300)
    alpha = 0.8
    gs = GridSpec(5,5,figure=fig, left =0.02, bottom = 0.02, right=0.98 ,top = 0.96, wspace = 0.05,hspace = 0.12, height_ratios = [0.1,0.3,1,1,1])
    for i, model in enumerate(colors.keys()):
        model_ax = fig.add_subplot(gs[(i//5)+2,i%5])
        if i == 0:
            scatter= model_ax.scatter(data.loc[(data.model == model) & gen_length,'x'], data.loc[(data.model == model)  & gen_length,'y'], c=data.loc[(data.model  == model)  & gen_length,'length_color'], cmap = 'magma', vmin=0,vmax=1, alpha = alpha - (i == 0)*0.17, s = 1, marker = ',',edgecolor='none',rasterized=True)
        else:
            _= model_ax.scatter(data.loc[(data.model == model) & gen_length,'x'], data.loc[(data.model == model)  & gen_length,'y'], c=data.loc[(data.model  == model)  & gen_length,'length_color'], cmap = 'magma', vmin=0,vmax=1, alpha = alpha - (i == 0)*0.17, s = 1, marker = ',',edgecolor='none',rasterized=True)
        _ = (lambda ax: (
            ax.set_title(model,pad=2)
            , ax.set_xticks([])
            , ax.set_xticklabels([])
            , ax.set_yticks([])
            , ax.set_yticklabels([])
            , ax.set_xlim(tsne_result.x.min(),tsne_result.x.max())
            , ax.set_ylim(data.y.min()
            , data.y.max())
            ))(model_ax)
        _= [spine.set_visible(True) for spine in model_ax.spines.values()]
        if i<=1: _= model_ax.set_title(model + " (Length 14-200)",pad=2)
    cbar_ax = fig.add_subplot(gs[0,1:4])
    cbar = fig.colorbar(scatter, cax=cbar_ax,orientation = 'horizontal')
    cbar.set_label('Length',fontsize = 8)
    cbar.set_ticks([0,1]) 
    cbar.set_ticklabels([str(int(tick)) for tick in np.interp(np.array([tick for tick in cbar.get_ticks()]), (0, 1), (data.length.min(), data.length.max()))])
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig= plt.figure(figsize=(8.27,11.7/2),dpi=300)
    alpha = 0.8
    gs = GridSpec(5,5,figure=fig, left =0.02, bottom = 0.02, right=0.98 ,top = 0.96, wspace = 0.05,hspace = 0.12, height_ratios = [0.1,0.3,1,1,1])
    for i, model in enumerate(colors.keys()):
        model_ax = fig.add_subplot(gs[(i//5)+2,i%5])
        scatter= model_ax.scatter(data.loc[(data.model == model) & gen_length,'x'], data.loc[(data.model == model)  & gen_length,'y'], c=data.loc[(data.model  == model)  & gen_length,'tm_max_color'], cmap = 'magma',vmin=0,vmax=1, alpha = alpha - (i == 0)*0.17, s = 1, marker = ',',edgecolor='none',rasterized=True)
        _ = (lambda ax: (
            ax.set_title(model,pad=2)
            , ax.set_xticks([])
            , ax.set_xticklabels([])
            , ax.set_yticks([])
            , ax.set_yticklabels([])
            , ax.set_xlim(tsne_result.x.min(),tsne_result.x.max())
            , ax.set_ylim(data.y.min()
            , data.y.max())
            ))(model_ax)
        _= [spine.set_visible(True) for spine in model_ax.spines.values()]
    cbar_ax = fig.add_subplot(gs[0,1:4])
    cbar = fig.colorbar(scatter, cax=cbar_ax,orientation = 'horizontal')
    cbar.set_label('TM-Score (Best Hit - FoldSeek, ESM-Atlas30)',fontsize = 8)
    cbar.set_ticks([0,1]) 
    cbar.set_ticklabels([str(tick) for tick in np.interp(np.array([tick for tick in cbar.get_ticks()]), (0, 1), (data.tm_max.min(), data.tm_max.max()))])
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig= plt.figure(figsize=(8.27,11.7/2),dpi=300)
    alpha = 0.8
    gs = GridSpec(5,5,figure=fig, left =0.02, bottom = 0.02, right=0.98 ,top = 0.96, wspace = 0.05,hspace = 0.12, height_ratios = [0.1,0.3,1,1,1])
    for i, model in enumerate(colors.keys()):
        model_ax = fig.add_subplot(gs[(i//5)+2,i%5])
        scatter= model_ax.scatter(data.loc[(data.model == model) & gen_length,'x'], data.loc[(data.model == model)  & gen_length,'y'], c=data.loc[(data.model  == model)  & gen_length,'cluster_color'], cmap = 'magma',vmin=0,vmax=1, alpha = alpha - (i == 0)*0.17, s = 1, marker = ',',edgecolor='none',rasterized=True)
        _ = (lambda ax: (
            ax.set_title(model,pad=2)
            , ax.set_xticks([])
            , ax.set_xticklabels([])
            , ax.set_yticks([])
            , ax.set_yticklabels([])
            , ax.set_xlim(tsne_result.x.min(),tsne_result.x.max())
            , ax.set_ylim(data.y.min()
            , data.y.max())
            ))(model_ax)
        _= [spine.set_visible(True) for spine in model_ax.spines.values()]
    cbar_ax = fig.add_subplot(gs[0,1:4])
    cbar = fig.colorbar(scatter, cax=cbar_ax,orientation = 'horizontal')
    cbar.set_label('Cluster #',fontsize = 8)
    cbar.set_ticks([0,1]) 
    cbar.set_ticklabels([str(int(tick)) for tick in np.interp(np.array([tick for tick in cbar.get_ticks()]), (0, 1), (data.cluster.min(), data.cluster.max()))])
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig, ax = plt.subplots(figsize=(8.27,8.27))
    _= sns.violinplot(x='model',y='cluster_size',ax=ax,data=cluster_sizes.sort_values('model'), hue = 'model', palette=colors, linewidth=0.3, inner = 'box',cut=0)
    _= sns.stripplot(data=cluster_sizes.sort_values('model'), x="model", y="cluster_size", ax=ax, size=0.75, color = 'black', jitter = 0.35, alpha = 0.8, rasterized=True)
    _= (lambda ax: (
        ax.set_ylabel('Cluster Size',fontsize = 8)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.set_xlabel('')
        ,ax.set_ylim(0,100)
        ,ax.tick_params(axis='x', which='both', size=0)
        ,ax.tick_params(axis='x', which='major', labelrotation=30,labelsize = 6)
        ,ax.tick_params(axis='y',labelsize=5)
        ))(ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig, ax = plt.subplots(figsize=(8.27,8.27))
    _= sns.violinplot(x='model',y='qtmscore',ax=ax,data=foldseek_raw.sort_values('model'), hue = 'model', palette=colors, linewidth=0.3, inner = 'box',cut=0)
    _= sns.stripplot(data=foldseek_raw.sample(frac=0.01).sort_values('model'), x="model", y="qtmscore", ax=ax, size=0.7, color = 'black', jitter = 0.3, alpha = 0.05, rasterized=True)
    _= sns.stripplot(data=foldseek_raw.groupby(['model','entity_id'])['qtmscore'].max().reset_index().dropna().sample(frac=0.05).sort_values('model'), x="model", y="qtmscore", ax=ax, size=1.5, color = 'dimgrey', jitter = 0.3, alpha = 0.9, rasterized=True)
    _= (lambda ax: (
        ax.set_ylabel('TM-Score (FoldSeek, ESM-Atlas30)',fontsize = 8)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.set_xlabel('')
        ,ax.set_ylim(0,1)
        ,ax.tick_params(axis='x', which='both', size=0)
        ,ax.tick_params(axis='x', which='major', labelrotation=30,labelsize = 6)
        ,ax.tick_params(axis='y',labelsize=5)
        ))(ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    plt.close('all')

