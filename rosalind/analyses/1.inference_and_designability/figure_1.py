#imports
#----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import math

#Data load
#----------------------------------------------------------------------------------------------------
gen_meta = pd.read_csv('/scratch/alexb2/generative_protein_models/share/generation_metadata_cleaned.csv',index_col=False)
mpnn_meta = pd.read_csv('/scratch/alexb2/generative_protein_models/share/mpnn_metadata_cleaned.csv',index_col=False)
fold_meta = pd.read_csv('/scratch/alexb2/generative_protein_models/share/fold_metadata_cleaned.csv',index_col=False)
#
sc = pd.read_csv('/scratch/alexb2/generative_protein_models/share/SC_Scoring_20240920_180051.csv')
refold = pd.read_csv('/scratch/alexb2/generative_protein_models/share/SC_Scoring_Refolds_20240921_134506.csv')
sc = pd.concat([sc,refold])
sc['length'] = sc.loc[:,'design_file'].str.split('_len').str[-1].str.split('_').str[0].astype(int)
sc = sc.loc[:,['model','length','design_file','tm_norm_aa','rmsd','omegafold_confidence']]

#global matplotlib settings
#----------------------------------------------------------------------------------------------------
plt.rcParams.update({
    'axes.spines.top': False,        # Hide top spine
    'axes.spines.right': False,      # Hide right spine
    'axes.spines.left': False,        # Show left spine
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
#Additional plotting functions
#----------------------------------------------------------------------------------------------------
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


def create_bins(group_values, num_bins):
    min_value = group_values.min()
    max_value = group_values.max()
    bins = np.linspace(min_value, max_value, num_bins + 1)
    bin_number = np.digitize(group_values, bins) - 1 
    return pd.Series(bin_number, index=group_values.index)


def place_legends(fig, positions):
    leg_ax1 = fig.add_axes(positions[0])
    legend_elements = [Line2D([0], [0], color='#586386', lw=2, label='ProteinSGM'),
    Line2D([0], [0], color='#b3f4ff', lw=2, label='Genie'),
    Line2D([0], [0], color='#005cdf', lw=2, label='RFdiffusion'),
    Line2D([0], [0], color='#30aa8f', lw=2, label='FrameDiff'),
    Line2D([0], [0], color='#7373ae', lw=2, label='FoldingDiff'),
    Line2D([0], [0], color='white', lw=2, label=''),
    Line2D([0], [0], color='black', lw=0.8, linestyle = ':', label='Backbone'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '--', label='Sequence'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '-', label='Full Structure')]
    legend = leg_ax1.legend(handles=legend_elements, loc='center', ncol=1, fontsize=6)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white') 
    leg_ax1.axis('off')
    #
    leg_ax2 = fig.add_axes(positions[1])
    legend_elements = [Line2D([0], [0], color='#ff7bb9', lw=2, label='ESM-Design'),
    Line2D([0], [0], color='#cf4f00', lw=2, label='RITA'),
    Line2D([0], [0], color='#e44fff', lw=2, label='ProGen2'),
    Line2D([0], [0], color='#93355c', lw=2, label='EvoDiff'),
    Line2D([0], [0], color='#9e0049', lw=2, label='ProtGPT2'),
    Line2D([0], [0], color='white', lw=2, label=''),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '--', label='Sequence'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '-', label='Full Structure')]
    legend = leg_ax2.legend(handles=legend_elements, loc='center', ncol=1, fontsize=6)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white') 
    leg_ax2.axis('off')
    #
    leg_ax3 = fig.add_axes(positions[2])
    legend_elements = [Line2D([0], [0], color='#809a2d', lw=2, label='Chroma'),
    Line2D([0], [0], color='#a4837b', lw=2, label='ProteinGenerator'),
    Line2D([0], [0], color='#ffc73c', lw=2, label='Protpardelle'),
    Line2D([0], [0], color='white', lw=2, label=''),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '-', label='Full Structure')]
    legend = leg_ax3.legend(handles=legend_elements, loc='center', ncol=1, fontsize=6)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white') 
    leg_ax3.axis('off')
    #
    leg_ax4 = fig.add_axes(positions[3])
    legend_elements = [Line2D([0], [0], color='black', lw=0.8, linestyle = '--', label='Sequence Design'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '-', label='Structure Prediction')]
    legend = leg_ax4.legend(handles=legend_elements, loc='center', ncol=1, fontsize=6)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white') 
    leg_ax4.axis('off')
    #
    leg_ax5 = fig.add_axes(positions[4])
    legend_elements = [Line2D([0], [0], color='#586386', lw=2, label='ProteinSGM'),
    Line2D([0], [0], color='white', lw=2, label=''),
    Line2D([0], [0], color='black', lw=0.8, linestyle = ':', label='Backbone'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '-.', label='Backbone (Refined)'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '--', label='Sequence'),
    Line2D([0], [0], color='black', lw=0.8, linestyle = '-', label='Full Structure')]
    legend = leg_ax5.legend(handles=legend_elements, loc='center', ncol=1, fontsize=6)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('white') 
    leg_ax5.axis('off')

#Figure 1
#----------------------------------------------------------------------------------------------------
colors = {
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
output_name = 'figure_1.pdf'
jitter_range = 0.8
aspect_ratio = 11.7/8.3 #A4 (landscape) aspect ratio
figure_height = 8.3  #Update this for different figure scale, 11.69 for full A4.

with PdfPages(output_name) as pdf:
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=300)
    place_legends(fig, [[0.17, 0.5, 0.2, 0.4]
                        ,[0.3, 0.1, 0.2, 0.4]
                        ,[0.62, 0.12, 0.2, 0.4]
                        ,[0.35, 0.43, 0.2, 0.4]
                        ,[0.5, 0.65, 0.2, 0.4]])
    gs = GridSpec(3,3, figure=fig, left =0.04, right=0.98,top = 0.96, bottom = 0.04,wspace = 0.03, hspace = 0.05, height_ratios = [0.7,0.7,1])
    #Plot the gen times for the backbone generative models
    #----------------------------------------------------------------------------------------------------
    bb_ax = fig.add_subplot(gs[:,0])
    _= (lambda ax: (
            ax.set_ylim(0,560)
            ,ax.set_xlim(14,200)
            ,ax.set_xlabel('Monomer Length')
            ,ax.set_ylabel("Wall-time (Seconds)", rotation=90)
            ,ax.spines['left'].set_visible(True)
            ,ax.set_xticks([25,50,75,100,125,150,175,200])
            ,ax.set_xticks(np.arange(20,200,5),minor=True)
            ,ax.set_title('A)',loc='center',fontsize=12)
            #,ax.title.set_position([-0.1,1])
            ))(bb_ax)
    for m in ['RFdiffusion','FrameDiff', 'FoldingDiff', 'Genie']:
        data = gen_meta.loc[(gen_meta.model == m),:].groupby(['length'])[['wall_time_bb','wall_time_aa','wall_time_seq']].mean().reset_index()
        for column, linestyle, alpha in [('wall_time_bb',':',0.3), ('wall_time_seq','--',0.5), ('wall_time_aa','-',0.5)]:
            _= bb_ax.plot(data.length, data[column], label=f'{m} Mean', color=colors[m], linewidth=0.5,linestyle = linestyle,alpha = alpha)
        _= bb_ax.fill_between(data.length, data.wall_time_aa, data.wall_time_seq, color=colors[m], alpha=0.05)
        _= bb_ax.fill_between(data.length, data.wall_time_seq, data.wall_time_bb, color=colors[m], alpha=0.03)
    data = gen_meta.loc[(gen_meta.model == 'ProteinSGM'),:].groupby(['length'])[['wall_time_bb','wall_time_gen','wall_time_seq','wall_time_aa']].mean().reset_index()
    for column, linestyle, alpha in [('wall_time_gen',':',0.3), ('wall_time_bb','-.',0.4), ('wall_time_seq','--',0.5), ('wall_time_aa','-',0.5)]:
        _= bb_ax.plot(data.length, data[column], label=f'{m} Mean', color=colors['ProteinSGM'], linewidth=0.5,linestyle = linestyle,alpha = alpha)
    _= bb_ax.fill_between(data.length, data.wall_time_bb, data.wall_time_gen, color=colors['ProteinSGM'], alpha=0.025)
    _= bb_ax.fill_between(data.length, data.wall_time_seq, data.wall_time_bb, color=colors['ProteinSGM'], alpha=0.03)
    _= bb_ax.fill_between(data.length, data.wall_time_aa, data.wall_time_seq, color=colors['ProteinSGM'], alpha=0.04)
    #Plot a line linking the ProteinSGM part to an outlier plot for ProteinSGM
    #----------------------------------------------------------------------------------------------------
    _= bb_ax.plot([80,500],[560,480],linewidth = 0.5, linestyle = (0, (5, 10)), color = colors['ProteinSGM'], alpha = 0.5, drawstyle='steps', marker = 'o',markerfacecolor='none', markeredgecolor=colors['ProteinSGM'],markeredgewidth = 0.5)
    outlier_bound = fig.add_subplot(gs[0,1])
    _= outlier_bound.axis('off')
    _= outlier_bound.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor=colors['ProteinSGM'], linestyle = (0, (5, 7)), facecolor='none'))
    out_ax = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0,1], height_ratios = [1,9,1], width_ratios = [1,14,0.2] )[1,1])
    _= (lambda ax: (
            ax.set_xlim(14,200)
            ,ax.set_xlabel('Monomer Length')
            ,ax.set_ylabel("Wall-time (Seconds)", rotation=90)
            ,ax.spines['left'].set_visible(True)
            ,ax.set_xticks([25,50,75,100,125,150,175,200])
            ,ax.set_xticks(np.arange(20,200,5),minor=True)
            ,ax.set_title('E)',loc='center',fontsize=12)
            ))(out_ax)
    for column, linestyle, alpha in [('wall_time_gen',':',0.3), ('wall_time_bb','-.',0.4), ('wall_time_seq','--',0.5), ('wall_time_aa','-',0.5)]:
        _= out_ax.plot(data.length, data[column], label=f'{m} Mean', color=colors['ProteinSGM'], linewidth=0.5,linestyle = linestyle,alpha = alpha)
    _= out_ax.fill_between(data.length, data.wall_time_bb, data.wall_time_gen, color=colors['ProteinSGM'], alpha=0.025)
    _= out_ax.fill_between(data.length, data.wall_time_seq, data.wall_time_bb, color=colors['ProteinSGM'], alpha=0.03)
    _= out_ax.fill_between(data.length, data.wall_time_aa, data.wall_time_seq, color=colors['ProteinSGM'], alpha=0.04)
    #Plot the gen times for the sequence generative models
    #----------------------------------------------------------------------------------------------------
    seq_ax = fig.add_subplot(gs[:,1])
    _= (lambda ax: (
            ax.set_ylim(0,560)
            ,ax.set_xlim(14,200)
            ,ax.set_xlabel('Monomer Length')
            ,ax.set_xticks([25,50,75,100,125,150,175,200])
            ,ax.set_xticks(np.arange(20,200,5),minor=True)
            ,ax.tick_params(which = 'both', left=False, labelleft=False) 
            ,ax.text(107,210, 'B)',ha='center', va='center', fontsize=12)
            ))(seq_ax)
    for m in ['ProtGPT2','ProGen2', 'RITA', 'ESM-Design', 'EvoDiff']:
        data = gen_meta.loc[(gen_meta.model == m),:].groupby(['length'])[['wall_time_bb','wall_time_aa','wall_time_seq']].mean().reset_index()
        for column, linestyle, alpha in [('wall_time_seq','--',0.5), ('wall_time_aa','-',0.5)]:
            _= seq_ax.plot(data.length, data[column], label=f'{m} Mean', color=colors[m], linewidth=0.5,linestyle = linestyle,alpha = alpha)
        _= seq_ax.fill_between(data.length, data.wall_time_aa, data.wall_time_seq, color=colors[m], alpha=0.05)
    #Plot the gen times for the all-atom generative models
    #----------------------------------------------------------------------------------------------------
    aa_ax = fig.add_subplot(gs[:,2])
    _= (lambda ax: (
            ax.set_ylim(0,560)
            ,ax.set_xlim(14,200)
            ,ax.set_xlabel('Monomer Length')
            ,ax.set_xticks([25,50,75,100,125,150,175,200])
            ,ax.set_xticks(np.arange(20,200,5),minor=True)
            ,ax.tick_params(which = 'both', left=False, labelleft=False)
            ,ax.text(107,210, 'C)',ha='center', va='center', fontsize=12)
            ))(aa_ax)
    for m in ['Protpardelle','Chroma', 'Protein-Generator']:
        data = gen_meta.loc[(gen_meta.model == m),:].groupby(['length'])[['wall_time_bb','wall_time_aa','wall_time_seq']].mean().reset_index()
        _= aa_ax.plot(data.length, data.wall_time_aa, label=f'{m} Mean', color=colors[m], linewidth=0.5)
    #Plot the times need for the postprocessing steps
    #----------------------------------------------------------------------------------------------------
    post_bound = fig.add_subplot(gs[1,1])
    _= post_bound.axis('off')
    _= post_bound.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none'))
    post_ax = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1,1], height_ratios = [1,9,1], width_ratios = [1,19,0.2] )[1,1])
    _= (lambda ax: (
            ax.set_xlim(14,200)
            ,ax.set_xlabel('Monomer Length')
            ,ax.set_ylabel("Wall-time (Seconds)", rotation=90,)
            ,ax.spines['left'].set_visible(True)
            ,ax.set_xticks([25,50,75,100,125,150,175,200])
            ,ax.set_xticks(np.arange(20,200,5),minor=True)
            ,ax.set_title('D)',loc='center',fontsize=12)
            ))(post_ax)
    _= post_ax.plot(fold_meta.length, fold_meta.wall_time_task, color='black', linewidth=0.5, linestyle = '-')
    _= post_ax.plot(mpnn_meta.length, mpnn_meta.wall_time_task, color='black', linewidth=0.5, linestyle ='--')
    #Now the Self consistency scoring
    #----------------------------------------------------------------------------------------------------
    sc_bound = fig.add_subplot(gs[:2,2])
    _= sc_bound.axis('off')
    _= sc_bound.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none'))
    sc_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[:2,2], height_ratios = [1,100,1], width_ratios = [3,100,0.01] )
    sc_gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=sc_gs[1,1],hspace = 0.7)
    boxplot_stats = sc.groupby(['length', 'model'])['tm_norm_aa'].apply(calc_boxplot_stats).unstack().reset_index()
    raw_data = pd.merge(sc, boxplot_stats.loc[:,['length','model','lower_whisker','upper_whisker','min','max']], how='left', on=['length','model'])
    raw_data.loc[:,'outlier'] = (raw_data['tm_norm_aa'] < raw_data.lower_whisker) | (raw_data['tm_norm_aa'] > raw_data.upper_whisker)
    raw_data.loc[:,'bin'] = raw_data.groupby('model')['tm_norm_aa'].apply(create_bins, num_bins=100).reset_index(drop=True)
    raw_data.loc[:,'bin_rank'] = raw_data.groupby(['model', 'length', 'bin'])['tm_norm_aa'].rank(method='min', ascending=False)
    for i, m in enumerate(['ProteinSGM', 'Genie', 'RFdiffusion', 'FrameDiff', 'FoldingDiff','Chroma', 'Protein-Generator','Protpardelle']):
        sc_ax = fig.add_subplot(sc_gs[i,0])
        _= (lambda ax: (
            ax.set_xlim(14,200)
            ,ax.set_ylim(0,1)
            ,ax.spines['left'].set_visible(True)
            ,ax.set_xticks([25,50,75,100,125,150,175,200])
            ,ax.set_xticks(np.arange(20,200,5),minor=True)
            ))(sc_ax)
        if i == 7:
            _= sc_ax.set_xlabel('Monomer Length')
        else:
            _= sc_ax.spines['bottom'].set_visible(False)
            _= sc_ax.tick_params(which = 'both', bottom=False, labelbottom=False) 
        if i == 0:
            _= sc_ax.set_title("scTM-Score\n(Backbone Sequence Designs)",loc='right')
            _= sc_ax.text(103,1.5, 'F)',ha='center', va='center', fontsize=12)
        if i == 5:
            _= sc_ax.set_title("scTM-Score\n(Refolded All-Atom Structures)",loc='right')
        model_boxplot_stats = boxplot_stats.loc[boxplot_stats.model == m,:]
        model_raw_data = raw_data.loc[raw_data.model== m,:].copy()
        jm = int(model_raw_data.bin_rank.max())
        offset_adjust = [-(jitter_range/jm)* i if i%2==0 else (jitter_range/jm)* (i-1) for i in range(1,jm+1)]
        model_raw_data.loc[:,'offset_adjust'] = model_raw_data.loc[:,'bin_rank'].apply(lambda x: offset_adjust[int(x)-1])
        #_= sc_ax.scatter(model_raw_data[~model_raw_data.outlier]['length'] + model_raw_data[~model_raw_data.outlier]['offset_adjust'], model_raw_data[~model_raw_data.outlier]['tm_norm_aa'], color=colors[m], alpha=0.4, edgecolor='none', s=0.1, rasterized=True)
        #_= sc_ax.scatter(model_raw_data[model_raw_data.outlier]['length'] + model_raw_data[model_raw_data.outlier]['offset_adjust'], model_raw_data[model_raw_data.outlier]['tm_norm_aa'], color='black', alpha=0.6, marker = 'X',edgecolor='none', s=0.3,rasterized=True)
        _= sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['median'], label='', color=colors[m], linewidth=0.5)
        _= sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q3'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        _= sc_ax.plot(model_boxplot_stats['length'], model_boxplot_stats['q1'], linestyle='-', color=colors[m], linewidth=0.3, alpha = 0.5)
        _= sc_ax.fill_between(model_boxplot_stats['length'], model_boxplot_stats['q1'], model_boxplot_stats['q3'], color=colors[m], alpha=0.1)
    #Save and clear
    #----------------------------------------------------------------------------------------------------
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    plt.close('all')

