import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap  # Make sure to import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import patches, lines
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats.mstats import winsorize



colors = {'Wild-Type': 'dimgrey',
'ProteinMPNN': 'darkgrey',
'RFdiffusion': '#005cdf',
'EvoDiff': '#93355c',
'Chroma': '#809a2d',
'Protpardelle': '#ffc73c',
'ProteinGenerator': '#a4837b',
}

data = pd.read_csv('/scratch/alexb2/generative_protein_models/analyses/4.conditional_design/in_vitro_tev_scripts/TEV_image_counts_with_designs2.csv')

data['model'] = pd.Categorical(data['model'], categories=list(colors.keys()), ordered=True)
#data = data.sort_values(by=['model','mpnn','mpnn_fixed','complex','structure_motif','sequence_motif','design_id'])
data.rename(columns={'Sequence Alignment\n(BLOSSUM62)': 'Sequence Alignment\n(BLOSUM62)'}, inplace=True)
data = data.loc[data.replicate!= 'replicate2',:]    #Dropping due to outliers

#We've got some big outliers in our pc data so we'll windsorize 
data.loc[data.design_id == 'PC','Fold Increase\n(From GFP+ Cells)'] = winsorize(data.loc[data.design_id == 'PC','Fold Increase\n(From GFP+ Cells)'], limits=[0.1, 0.1])

data['mean_fold_change'] = data.groupby('design_id')['Fold Increase\n(From GFP+ Cells)'].transform('mean')
#data = data.sort_values(by=['Fold Increase\n(From GFP+ Cells)'],ascending=False)
data = data.sort_values(by=['mean_fold_change'],ascending=False)
#data = data.loc[data['Fold Increase\n(From GFP+ Cells)'] < 11,:]#data = data.iloc[1:,:].reset_index(drop=True) #Drop the outlier in the positive controls

#test = data.groupby(['design_id','model'])["Fold Increase\n(From GFP+ Cells)"].min().reset_index().dropna()
#test.columns = ['design_id','model','min_fold_increase']
#test = test.sort_values(by=['min_fold_increase'],ascending=False)
#test = test.loc[test.min_fold_increase >=1,:]

#data.loc[:,'highlight'] = 0.3 * data.design_id.isin(list(test.loc[test.design_id != 'PC',:].design_id.unique())) # Designs where all were above fold =1
data.loc[:,'highlight'] = 5 * ((data['p_value (sans replicate 2)'] <= 0.05) & (data.design_id != 'PC'))
data.loc[data.highlight == 0,'highlight'] = np.nan



# List of features to plot
features = ['TM-Score','RMSD','RMSD (motif)','Sequence Alignment\n(BLOSUM62)', 'Sequence Alignment\n(Motif Region)','Sequence Alignment\n(Non-Motif)']


fig = plt.figure(figsize=(12, 14),dpi = 300)
gs = gridspec.GridSpec(7, 1, height_ratios=[4, 1, 1, 1, 1, 1, 1], hspace=0.3)
axes = [fig.add_subplot(gs[i, 0]) for i in range(7)]

box = sns.boxplot(ax= axes[0], data = data, x = 'design_id',y = 'Fold Increase\n(From GFP+ Cells)', hue = 'model', palette = colors,legend = False, zorder=0)
strip = sns.stripplot(ax= axes[0], data = data, x = 'design_id',y = 'Fold Increase\n(From GFP+ Cells)',color ='silver',edgecolor = 'black',legend = False,size=2,zorder=1)
means = sns.scatterplot(ax= axes[0], data = data, x = 'design_id',y = 'mean_fold_change',color ='black', legend = False,marker='x',zorder=2,s=5)
_= sns.stripplot(ax= axes[0], data = data.loc[:,['design_id','highlight']].drop_duplicates(), x = 'design_id',y = 'highlight',color ='black',legend = False, marker='*',s=5)
_=axes[0].yaxis.label.set_size(9)
_=axes[0].set_ylim(0,35)
_=axes[0].tick_params(axis='y', labelsize=6) 
_=axes[0].set_yticks([0,1,5,10,15,20,25,30])
_=axes[0].set_xlabel('')
#_=axes[i+1].set_yticks([])
_=axes[0].set_xticks([])
_=axes[0].spines['top'].set_visible(False)
_=axes[0].spines['right'].set_visible(False)
_=axes[0].spines['bottom'].set_position(('data', 0))
_=axes[0].axhline(y=1, color='black', linestyle=':', linewidth=1)

for i, feature in enumerate(features):
    #multiple rows with identical valued for these features so need to drop
    unq = data.loc[:,['design_id','model',feature,'complex','sequence_motif','structure_motif','mpnn_fixed']].drop_duplicates()
    
    bar = sns.barplot(ax= axes[i+1], data = unq, x = 'design_id',y = feature, hue = 'model', palette = colors,legend = False)
    _=axes[i+1].yaxis.label.set_size(7)
    _=axes[i+1].tick_params(axis='y', labelsize=6) 
    _=axes[i+1].set_xlabel('')
    #_=axes[i+1].set_yticks([])
    _=axes[i+1].set_xticks([])
    _=axes[i+1].spines['top'].set_visible(False)
    _=axes[i+1].spines['right'].set_visible(False)
    _=axes[i+1].spines['bottom'].set_position(('data', 0))
    sorted_patches = sorted(bar.patches, key=lambda x: x.get_x())
    for rect, row in zip(sorted_patches,unq.iterrows()):
        rect.set_edgecolor('black') 
        rect.set_linewidth(0.05) 
        row = row[1]
        if row.model == 'RFdiffusion':
            if row.complex == True:
                if row.mpnn_fixed == True:
                    rect.set_hatch('\\\\')
                else:
                    rect.set_hatch('xxxx')
            else:
                if row.mpnn_fixed == False:
                    rect.set_hatch('////')
        if row.model == 'Chroma' and row.complex == True:
                rect.set_hatch('\\\\')
        if row.model == 'ProteinGenerator':
            if row.sequence_motif == True:
                rect.set_hatch('\\\\')
            else:
                rect.set_hatch('////')
            
        
    


#leg_ax = fig.add_subplot(gs[0:3, 0]) 
leg_ax = fig.add_axes([0.33, 0.725, 0.4, 0.2])
legend_elements = [
    patches.Patch(facecolor='dimgrey', edgecolor='black', label='Wild type'),
    patches.Patch(facecolor='darkgrey', edgecolor='black', label='ProteinMPNN'),
    lines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=5, label='Mean'),
    lines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=5, label='p\u22640.05'),
    patches.Patch(facecolor='#005cdf', edgecolor='black', label='RFdiffusion'),
    patches.Patch(facecolor='#005cdf', edgecolor='black', hatch='\\\\', label='Template Structure in Complex'),
    patches.Patch(facecolor='#005cdf', edgecolor='black', hatch='////', label='Flexible Motif Sequence'),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='#93355c', edgecolor='black', label='EvoDiff'), 
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='#809a2d', edgecolor='black', label='Chroma'), 
    patches.Patch(facecolor='#809a2d', edgecolor='black', hatch='\\\\', label='Template Structure in Complex'),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='#ffc73c', edgecolor='black', label='Protpardelle'),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
    patches.Patch(facecolor='#a4837b', edgecolor='black', label='ProteinGenerator'),
    patches.Patch(facecolor='#a4837b', edgecolor='black', hatch='\\\\', label='Sequence Motif'),
    patches.Patch(facecolor='#a4837b', edgecolor='black', hatch='////', label='Structural Motif'),
    patches.Patch(facecolor='white', edgecolor='white', label=''),
]

legend = leg_ax.legend(handles=legend_elements, loc='center', ncol=6, fontsize=8)
legend.get_frame().set_linewidth(0)
legend.get_frame().set_facecolor('white') 
leg_ax.axis('off')

plt.savefig('figure4_by_mean_fold.pdf', format='pdf', bbox_inches='tight',dpi = 300)
plt.clf()
plt.close('all')

