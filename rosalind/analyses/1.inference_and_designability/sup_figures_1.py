#imports
#----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import math
import os

#global matplotlib settings
#----------------------------------------------------------------------------------------------------
plt.rcParams.update({
    'axes.spines.top': False,        # Hide top spine
    'axes.spines.right': False,      # Hide right spine
    'axes.spines.left': True,        # Show left spine
    'axes.spines.bottom': True,      # Show bottom spine
    'axes.linewidth': 0.25,          # Set spine linewidth
    'axes.edgecolor': 'dimgrey',     # Set spine edge color
    'axes.facecolor': (0, 0, 0, 0) ,        # Set axes background color (if needed)
    'axes.grid': False,  # Disable grid lines on all plots
    'font.size': 5,          # Set default font size
    'axes.labelsize': 5,       # Font size for x and y labels
    'axes.titlesize': 6,       # Font size for titles
    'legend.fontsize': 5,      # Font size for legends
    'xtick.labelsize': 4,      # Font size for x-axis tick labels
    'ytick.labelsize': 4,       # Font size for y-axis tick labels
    'font.family': 'sans-serif',  # Set default font family to sans-serif
    'font.sans-serif': ['Arial', 'Helvetica', 'Calibri', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'],  # Specify preferred sans-serif fonts
    'xtick.direction': 'out',   # Set tick direction to outward
    'ytick.direction': 'out',
    'xtick.major.size': 2,      # Set major tick size
    'ytick.major.size': 2,
    'xtick.major.width': 0.25,   # Set major tick width
    'ytick.major.width': 0.25,
    'xtick.minor.visible': True,    # Show minor ticks on x-axis
    'ytick.minor.visible': True,    # Show minor ticks on y-axis
    'xtick.minor.size': 2,      # Set minor tick size
    'ytick.minor.size': 2,
    'xtick.minor.width': 0.5,   # Set minor tick width
    'ytick.minor.width': 0.5,
    'xtick.color': 'dimgrey',   # Set tick color
    'ytick.color': 'dimgrey',
    'xtick.labelcolor': 'black',    # Set tick label color
    'ytick.labelcolor': 'black'    # Set tick label color
})
#
sc = pd.read_csv('/scratch/alexb2/generative_protein_models/share/SC_Scoring_20240920_180051.csv')
refold = pd.read_csv('/scratch/alexb2/generative_protein_models/share/SC_Scoring_Refolds_20240921_134506.csv')
sc = pd.concat([sc,refold])
sc['length'] = sc.loc[:,'design_file'].str.split('_len').str[-1].str.split('_').str[0].astype(int)
sc = sc.loc[:,['model','length','design_file','tm_norm_aa','rmsd','omegafold_confidence']]

ofc = pd.read_csv('/scratch/alexb2/generative_protein_models/share/OF_Confidence_20241008_174604.csv')
ofc['length'] = ofc.loc[:,'design_file'].str.split('_len').str[-1].str.split('_').str[0].astype(int)
ofc = ofc.loc[:,['model','length','omegafold_confidence']]

generated = pd.read_csv('/scratch/alexb2/generative_protein_models/share/14_200_per_chain.csv')
#Cleanup
generated.loc[:,'filename'] = generated.filepath.str.split('/').str[-1]
generated.loc[:,'model'] = generated.filename.str.split('_').str[0]
uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
generated['entity_id'] = generated['filepath'].str.extract(f'({uuid_pattern})', expand=False)
#Load in (Per chain) Outputs from PyRosetta on the reference Pisces chains
pisces = pd.read_csv('/scratch/alexb2/generative_protein_models/share/pisces_reference_per_chain.csv')
#Cleanup
pisces['model'] = 'PISCES'
pisces['entity_id'] = pisces['filepath'].str.split('/').str[-1].str.split('.').str[0] + "_chain" + pisces.chain_no.astype(str)
#Concat into full dataset
data = pd.concat([generated,pisces])
data.loc[:,'length'] = data.chain_sequence.str.len().astype(int)
data = data.loc[(data.length <= 200) & (data.length >= 14),:]

def calc_boxplot_stats(group_values):
    q1 = group_values.quantile(0.25)
    q2 = group_values.median()
    q3 = group_values.quantile(0.75)
    q0 = group_values.min()
    q4 = group_values.max()
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    return pd.Series({
        'min': q0,
        'q1': q1,
        'median': q2,
        'q3': q3,
        'max': q4,
        'iqr': iqr,
        'lower_whisker': lower_whisker,
        'upper_whisker': upper_whisker,
    })

#
def create_bins(group_values, num_bins):
    min_value = group_values.min()
    max_value = group_values.max()
    bins = np.linspace(min_value, max_value, num_bins + 1)
    bin_number = np.digitize(group_values, bins) - 1 
    return pd.Series(bin_number, index=group_values.index)
#Generate plot/s
#----------------------------------------------------------------------------------------------------
colors = {'PISCES': 'dimgrey',
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
#
output_name = 'supplementary_figures_1.pdf'
jitter_range = 0.8

with PdfPages(output_name) as pdf:
    fig, ax = plt.subplots(figsize=(8.27,8.27))
    tally = data.groupby('model')['entity_id'].count().reset_index()
    tally.columns = ['model','# Monomers (14-200)']
    tally['model'] = pd.Categorical(tally['model'], categories=list(colors.keys()), ordered=True)
    _= sns.barplot(x='model',y='# Monomers (14-200)',ax=ax,data=tally.sort_values('model'), hue = 'model', palette=colors)
    for p in ax.patches:
        _= ax.text(p.get_x() + p.get_width() / 2,p.get_height(),int(p.get_height()),ha='center', va='bottom') 
    _= (lambda ax: (
        #ax.set_title('E)',pad = 0, fontsize =10)
        ax.set_ylabel('# Monomers (14-200)',fontsize = 8)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.set_xlabel('')
        ,ax.minorticks_off()
        ,ax.tick_params(axis='y',labelsize=6)
        ,ax.tick_params(axis='x', labelsize=5,rotation=90)
        ))(ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig, ax = plt.subplots(figsize=(8.27,8.27))
    _= (lambda ax: (
        #ax.set_title('E)',pad = 0, fontsize =10)
        ax.set_ylabel('Probability Density',fontsize = 8)
        ,ax.spines['left'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.set_xlabel('Length')
        ,ax.minorticks_off()
        ,ax.set_xticks(np.arange(25,201,25))
        ,ax.tick_params(axis='y',labelsize=6)
        ))(ax)
    _= ax.hist(x=data.loc[~data.model.isin(['PISCES','FoldingDiff','ProteinSGM']),:].length,bins=range(14,202),density=True, color = 'grey')
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig = plt.figure(figsize=(8.27,11.69))
    gs = GridSpec(8,3, figure=fig, left =0.06, right=0.97,top = 0.97, bottom = 0.03)
    #
    for i, m in enumerate(['ProteinSGM', 'Genie', 'RFdiffusion', 'FrameDiff', 'FoldingDiff','Chroma', 'Protein-Generator','Protpardelle']):
        print(i)
        sc_ax = fig.add_subplot(gs[i,0])
        sc_ax.set_ylim(0,1)
        sc_ax.set_xlim(10,205)
        sc_ax.set_ylabel("scTM-Score", rotation=90)
        sc_ax.text(15, 1.05, f'{m.replace('Protein-Generator','ProteinGenerator')}', fontsize=6, ha='left', va='top', color=colors[m])
        if i == 0:
            sc_ax.set_title("scTM-Score \n (Backbone Sequence Designs)",loc='right')
        if i == 5:
            sc_ax.set_title("scTM-Score \n (Refolded All-Atom Structures)",loc='right')
        if not i in [7]:
            sc_ax.spines['bottom'].set_visible(False)
            sc_ax.tick_params(which = 'both', bottom=False, labelbottom=False) 
        else:
            sc_ax.set_xlabel('Monomer Length')
        boxplot_stats = sc.groupby(['length', 'model'])['tm_norm_aa'].apply(calc_boxplot_stats).unstack().reset_index()
        raw_data = pd.merge(sc, boxplot_stats.loc[:,['length','model','lower_whisker','upper_whisker','min','max']], how='left', on=['length','model'])
        raw_data.loc[:,'outlier'] = (raw_data['tm_norm_aa'] < raw_data.lower_whisker) | (raw_data['tm_norm_aa'] > raw_data.upper_whisker)
        raw_data.loc[:,'bin'] = raw_data.groupby('model')['tm_norm_aa'].apply(create_bins, num_bins=100).reset_index(drop=True)
        raw_data.loc[:,'bin_rank'] = raw_data.groupby(['model', 'length', 'bin'])['tm_norm_aa'].rank(method='min', ascending=False)
        #
        model_boxplot_stats = boxplot_stats[boxplot_stats['model'] == m]
        model_raw_data = raw_data[raw_data['model']== m]
        jm = int(model_raw_data.bin_rank.max())
        offset_adjust = [-(jitter_range/jm)* i if i%2==0 else (jitter_range/jm)* (i-1) for i in range(1,jm+1)]
        model_raw_data.loc[:,'offset_adjust'] = model_raw_data.bin_rank.apply(lambda x: offset_adjust[int(x)-1])
        sc_ax.scatter(model_raw_data[~model_raw_data.outlier]['length'] + model_raw_data[~model_raw_data.outlier]['offset_adjust'], model_raw_data[~model_raw_data.outlier]['tm_norm_aa'], color=colors[m], alpha=0.4, edgecolor='none', s=0.1, rasterized=True)
        sc_ax.scatter(model_raw_data[model_raw_data.outlier]['length'] + model_raw_data[model_raw_data.outlier]['offset_adjust'], model_raw_data[model_raw_data.outlier]['tm_norm_aa'], color='black', alpha=0.6, marker = 'X',edgecolor='none', s=0.3,rasterized=True)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['median'], label='', color=colors[m], linewidth=0.5)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q3'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q1'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        sc_ax.fill_between(model_boxplot_stats['length'], model_boxplot_stats['q1'], model_boxplot_stats['q3'], color=colors[m], alpha=0.1)
    for i, m in enumerate(['ProteinSGM', 'Genie', 'RFdiffusion', 'FrameDiff', 'FoldingDiff','Chroma', 'Protein-Generator','Protpardelle']):
        print(i)
        sc_ax = fig.add_subplot(gs[i,1])
        sc_ax.set_ylim(0,55)
        sc_ax.set_xlim(10,205)
        sc_ax.set_ylabel("scRMSD", rotation=90)
        sc_ax.text(15, 55, f'{m.replace('Protein-Generator','ProteinGenerator')}', fontsize=6, ha='left', va='top', color=colors[m])
        if i == 0:
            sc_ax.set_title("scRMSD \n (Backbone Sequence Designs)",loc='right')
        if i == 5:
            sc_ax.set_title("scRMSD \n (Refolded All-Atom Structures)",loc='right')
        if not i in [7]:
            sc_ax.spines['bottom'].set_visible(False)
            sc_ax.tick_params(which = 'both', bottom=False, labelbottom=False) 
        else:
            sc_ax.set_xlabel('Monomer Length')
        boxplot_stats = sc.groupby(['length', 'model'])['rmsd'].apply(calc_boxplot_stats).unstack().reset_index()
        raw_data = pd.merge(sc, boxplot_stats.loc[:,['length','model','lower_whisker','upper_whisker','min','max']], how='left', on=['length','model'])
        raw_data.loc[:,'outlier'] = (raw_data['rmsd'] < raw_data.lower_whisker) | (raw_data['rmsd'] > raw_data.upper_whisker)
        raw_data.loc[:,'bin'] = raw_data.groupby('model')['rmsd'].apply(create_bins, num_bins=100).reset_index(drop=True)
        raw_data.loc[:,'bin_rank'] = raw_data.groupby(['model', 'length', 'bin'])['rmsd'].rank(method='min', ascending=False)
        #
        model_boxplot_stats = boxplot_stats[boxplot_stats['model'] == m]
        model_raw_data = raw_data[raw_data['model']== m]
        jm = int(model_raw_data.bin_rank.max())
        offset_adjust = [-(jitter_range/jm)* i if i%2==0 else (jitter_range/jm)* (i-1) for i in range(1,jm+1)]
        model_raw_data.loc[:,'offset_adjust'] = model_raw_data.bin_rank.apply(lambda x: offset_adjust[int(x)-1])
        sc_ax.scatter(model_raw_data[~model_raw_data.outlier]['length'] + model_raw_data[~model_raw_data.outlier]['offset_adjust'], model_raw_data[~model_raw_data.outlier]['rmsd'], color=colors[m], alpha=0.4, edgecolor='none', s=0.1, rasterized=True)
        sc_ax.scatter(model_raw_data[model_raw_data.outlier]['length'] + model_raw_data[model_raw_data.outlier]['offset_adjust'], model_raw_data[model_raw_data.outlier]['rmsd'], color='black', alpha=0.6, marker = 'X',edgecolor='none', s=0.3,rasterized=True)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['median'], label='', color=colors[m], linewidth=0.5)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q3'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q1'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        sc_ax.fill_between(model_boxplot_stats['length'], model_boxplot_stats['q1'], model_boxplot_stats['q3'], color=colors[m], alpha=0.1)
    for i, m in enumerate(['ProteinSGM', 'Genie', 'RFdiffusion', 'FrameDiff', 'FoldingDiff','Chroma', 'Protein-Generator','Protpardelle']):
        print(i)
        sc_ax = fig.add_subplot(gs[i,2])
        sc_ax.set_ylim(0,100)
        sc_ax.set_xlim(10,205)
        sc_ax.set_ylabel("Avg. Omegafold Confidence", rotation=90)
        sc_ax.text(15, 105, f'{m.replace('Protein-Generator','ProteinGenerator')}', fontsize=6, ha='left', va='top', color=colors[m])
        if i == 0:
            sc_ax.set_title("Avg. Omegafold Confidence \n (Backbone Sequence Designs)",loc='right')
        if i == 5:
            sc_ax.set_title("Avg. Omegafold Confidence \n (Refolded All-Atom Structures)",loc='right')
        if not i in [7]:
            sc_ax.spines['bottom'].set_visible(False)
            sc_ax.tick_params(which = 'both', bottom=False, labelbottom=False) 
        else:
            sc_ax.set_xlabel('Monomer Length')
        boxplot_stats = sc.groupby(['length', 'model'])['omegafold_confidence'].apply(calc_boxplot_stats).unstack().reset_index()
        raw_data = pd.merge(sc, boxplot_stats.loc[:,['length','model','lower_whisker','upper_whisker','min','max']], how='left', on=['length','model'])
        raw_data.loc[:,'outlier'] = (raw_data['omegafold_confidence'] < raw_data.lower_whisker) | (raw_data['omegafold_confidence'] > raw_data.upper_whisker)
        raw_data.loc[:,'bin'] = raw_data.groupby('model')['omegafold_confidence'].apply(create_bins, num_bins=100).reset_index(drop=True)
        raw_data.loc[:,'bin_rank'] = raw_data.groupby(['model', 'length', 'bin'])['omegafold_confidence'].rank(method='min', ascending=False)
        #
        model_boxplot_stats = boxplot_stats[boxplot_stats['model'] == m]
        model_raw_data = raw_data[raw_data['model']== m]
        jm = int(model_raw_data.bin_rank.max())
        offset_adjust = [-(jitter_range/jm)* i if i%2==0 else (jitter_range/jm)* (i-1) for i in range(1,jm+1)]
        model_raw_data.loc[:,'offset_adjust'] = model_raw_data.bin_rank.apply(lambda x: offset_adjust[int(x)-1])
        sc_ax.scatter(model_raw_data[~model_raw_data.outlier]['length'] + model_raw_data[~model_raw_data.outlier]['offset_adjust'], model_raw_data[~model_raw_data.outlier]['omegafold_confidence'], color=colors[m], alpha=0.4, edgecolor='none', s=0.1, rasterized=True)
        sc_ax.scatter(model_raw_data[model_raw_data.outlier]['length'] + model_raw_data[model_raw_data.outlier]['offset_adjust'], model_raw_data[model_raw_data.outlier]['omegafold_confidence'], color='black', alpha=0.6, marker = 'X',edgecolor='none', s=0.3,rasterized=True)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['median'], label='', color=colors[m], linewidth=0.5)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q3'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q1'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        sc_ax.fill_between(model_boxplot_stats['length'], model_boxplot_stats['q1'], model_boxplot_stats['q3'], color=colors[m], alpha=0.1)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig = plt.figure(figsize=(8.27,11.69))
    gs = GridSpec(10,1, figure=fig, left =0.06, right=0.97,top = 0.97, bottom = 0.03)
    for i, m in enumerate(['RFdiffusion', 'Genie', 'ProteinSGM', 'FoldingDiff', 'FrameDiff','EvoDiff', 'RITA', 'ProGen2', 'ProtGPT2', 'ESM-Design']):
        print(i)
        ofc_ax = fig.add_subplot(gs[i,0])
        ofc_ax.set_ylim(0,100)
        ofc_ax.set_xlim(10,205)
        ofc_ax.set_ylabel("Avg. Omegafold Confidence", rotation=90)
        ofc_ax.text(15, 105, f'{m.replace('Protein-Generator','ProteinGenerator')}', fontsize=6, ha='left', va='top', color=colors[m])
        if i == 0:
            ofc_ax.set_title("Avg. Omegafold Confidence \n (Selected Backbone Sequence Designs)",loc='right')
        if i == 5:
            ofc_ax.set_title("Avg. Omegafold Confidence \n (Generated Sequences)",loc='right')
        if not i in [9]:
            ofc_ax.spines['bottom'].set_visible(False)
            ofc_ax.tick_params(which = 'both', bottom=False, labelbottom=False) 
        else:
            ofc_ax.set_xlabel('Monomer Length')
        boxplot_stats = ofc.groupby(['length', 'model'])['omegafold_confidence'].apply(calc_boxplot_stats).unstack().reset_index()
        raw_data = pd.merge(ofc, boxplot_stats.loc[:,['length','model','lower_whisker','upper_whisker','min','max']], how='left', on=['length','model'])
        raw_data.loc[:,'outlier'] = (raw_data['omegafold_confidence'] < raw_data.lower_whisker) | (raw_data['omegafold_confidence'] > raw_data.upper_whisker)
        raw_data.loc[:,'bin'] = raw_data.groupby('model')['omegafold_confidence'].apply(create_bins, num_bins=100).reset_index(drop=True)
        raw_data.loc[:,'bin_rank'] = raw_data.groupby(['model', 'length', 'bin'])['omegafold_confidence'].rank(method='min', ascending=False)
        #
        model_boxplot_stats = boxplot_stats[boxplot_stats['model'] == m]
        model_raw_data = raw_data[raw_data['model']== m]
        jm = int(model_raw_data.bin_rank.max())
        offset_adjust = [-(jitter_range/jm)* i if i%2==0 else (jitter_range/jm)* (i-1) for i in range(1,jm+1)]
        model_raw_data.loc[:,'offset_adjust'] = model_raw_data.bin_rank.apply(lambda x: offset_adjust[int(x)-1])
        ofc_ax.scatter(model_raw_data[~model_raw_data.outlier]['length'] + model_raw_data[~model_raw_data.outlier]['offset_adjust'], model_raw_data[~model_raw_data.outlier]['omegafold_confidence'], color=colors[m], alpha=0.7, edgecolor='none', s=0.1, rasterized=True)
        ofc_ax.scatter(model_raw_data[model_raw_data.outlier]['length'] + model_raw_data[model_raw_data.outlier]['offset_adjust'], model_raw_data[model_raw_data.outlier]['omegafold_confidence'], color='black', alpha=0.8, marker = 'X',edgecolor='none', s=0.3,rasterized=True)
        ofc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['median'], label='', color=colors[m], linewidth=0.5)
        ofc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q3'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        ofc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q1'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        ofc_ax.fill_between(model_boxplot_stats['length'], model_boxplot_stats['q1'], model_boxplot_stats['q3'], color=colors[m], alpha=0.1)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig, ax = plt.subplots(figsize=(8.27,8.27))
    for i, m in enumerate(['ProteinSGM', 'Genie', 'RFdiffusion', 'FrameDiff', 'FoldingDiff']):
        subset = sc.loc[sc['model'] == m,:]
        ax.scatter(subset.tm_norm_aa, subset.rmsd, color='None', alpha=0.3, edgecolor=colors[m], linewidths=0.5, s=10, rasterized=True, label=m)
        ax.set_xlabel('scTM-Score', fontsize=10)
        ax.set_ylabel('scRMSD',fontsize=10)
    legend = ax.legend(fontsize=10, markerscale=3)
    legend.get_frame().set_facecolor('white')
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig, ax = plt.subplots(figsize=(8.27,8.27))
    for i, m in enumerate(['Chroma', 'Protein-Generator','Protpardelle']):
        subset = sc.loc[sc['model'] == m,:]
        ax.scatter(subset.tm_norm_aa, subset.rmsd, color='None', alpha=0.3, edgecolor=colors[m], linewidths=0.5, s=10, rasterized=True, label=m)
        ax.set_xlabel('scTM-Score', fontsize=10)
        ax.set_ylabel('scRMSD',fontsize=10)
    legend = ax.legend(fontsize=10, markerscale=3)
    legend.get_frame().set_facecolor('white')
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    plt.close('all')





