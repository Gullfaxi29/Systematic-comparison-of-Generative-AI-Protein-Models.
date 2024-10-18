import numpy as np
import pandas as pd
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from decimal import Decimal

from Bio import SeqIO
from collections import defaultdict

#-----------------------------------------
#Data Load/Clean/Transform
#-----------------------------------------

#Load in (Per chain) Outputs from PyRosetta on our generated monomers
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

#extract energy data
energy = data.copy(deep=True).loc[:,['model','entity_id','length','fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro']].drop_duplicates().reset_index(drop=True)
#Full energy function is the sum of all terms
energy.loc[:,'REF2015'] = energy.iloc[:,3:].sum(axis=1)
energy = pd.melt(energy, id_vars=['model','entity_id','length'], var_name='term', value_name='value')

#Calculate enrichment for each res across positions
split_sequences = data.copy(deep=True).loc[:,['model','entity_id','length','chain_sequence']]['chain_sequence'].apply(lambda x: pd.Series(list(x)))
split_sequences.columns = [i + 1 for i in range(split_sequences.shape[1])]
split_sequences = split_sequences.iloc[:,0:200]
enrichment = pd.melt(pd.concat([data.copy(deep=True).loc[:,['model','entity_id','length']], split_sequences], axis=1), id_vars=['model','entity_id','length'], var_name='position', value_name='res')
#rows with zero non-NaN values for res are missing data that will need to be excluded (rather than filled with zeros)
drop = enrichment.groupby(['model','position'])['res'].count().reset_index()
drop = drop.loc[drop.res == 0,['model','position']].drop_duplicates()
enrichment = enrichment.dropna()
#Pull out an overall enrichment summary
summary = (enrichment.groupby(['model','res']).size()/enrichment.groupby(['model']).size()).reset_index()
summary.columns = list(summary.columns[:-1]) + ['perc_enrichment']
summary.perc_enrichment = summary.perc_enrichment * 100
summary = pd.DataFrame([(mod, res) for mod in summary.model.unique() for res in summary.res.dropna().unique()], columns=['model', 'res']).merge(summary, on=['model','res'], how='outer')
summary.perc_enrichment = summary.perc_enrichment.fillna(0)
#Calculate enrichment per position
enrichment = (enrichment.groupby(['model','position','res']).size()/enrichment.groupby(['model','position']).size()).reset_index()
enrichment.columns = list(enrichment.columns[:-1]) + ['perc_enrichment']
enrichment.perc_enrichment = enrichment.perc_enrichment * 100
#Need to fill in the missing 0 data, and remove NaNs from missing data
enrichment = pd.DataFrame([(mod, pos, res) for mod in enrichment.model.unique() for pos in enrichment.position.unique() for res in enrichment.res.dropna().unique()], columns=['model', 'position','res']).merge(enrichment, on=['model', 'position','res'], how='outer')
enrichment.perc_enrichment = enrichment.perc_enrichment.fillna(0)
enrichment = enrichment[~enrichment[['model', 'position']].apply(tuple, axis=1).isin(drop.apply(tuple, axis=1))]
enrichment.res = '% ' + enrichment.res
#Repeat enrichment calcs but for secondary structure annotations
ss = data.copy(deep=True).loc[:,['model','entity_id','length','ss']]['ss'].apply(lambda x: pd.Series(list(x)))
ss.columns = [i + 1 for i in range(ss.shape[1])]
ss = ss.iloc[:,0:200]
ss = pd.concat([data.copy(deep=True).loc[:,['model','entity_id','length']], ss], axis=1)
ss = pd.melt(ss, id_vars=['model','entity_id','length'], var_name='position', value_name='ss_anot')
ss = ss.dropna()
ss = (ss.groupby(['model','position','ss_anot']).size()/ss.groupby(['model','position']).size()).reset_index()
ss.columns = list(ss.columns[:-1]) + ['perc_ss_enrichment']
ss.perc_ss_enrichment = ss.perc_ss_enrichment * 100
ss = pd.DataFrame([(mod, pos, anot) for mod in ss.model.unique() for pos in ss.position.unique() for anot in ss.ss_anot.dropna().unique()], columns=['model', 'position','ss_anot']).merge(ss, on=['model', 'position','ss_anot'], how='outer')
ss.perc_ss_enrichment = ss.perc_ss_enrichment.fillna(0)
ss = ss[~ss[['model', 'position']].apply(tuple, axis=1).isin(drop.apply(tuple, axis=1))]
ss = ss.pivot(index=['model','position'], columns='ss_anot',values='perc_ss_enrichment').fillna(0)
ss.E = ss.H + ss.E
ss.L = ss.E + ss.L
ss.reset_index(inplace=True)
ss['axis'] = 0

#AA enrichments for uniref
#----------------------------
#count_dict = defaultdict(int)
#for record in SeqIO.parse('/scratch/alexb2/generative_protein_models/analyses/3.diversity_and_novelty/esm_embeddings/uniref50.fasta', "fasta"):
#    seq_length = len(record.seq)
#    if 14 <= seq_length <= 200:
#        for res in record.seq:
#            count_dict[res] += 1
#uniref_enrichment_summary = pd.DataFrame(count_dict.items(), columns=['res', 'counts'])
#uniref_enrichment_summary = uniref_enrichment_summary.loc[uniref_enrichment_summary.res.isin(summary.res.unique())]
#uniref_enrichment_summary['perc_enrichment'] = uniref_enrichment_summary['counts'] / uniref_enrichment_summary.counts.sum()
#uniref_enrichment_summary.perc_enrichment = uniref_enrichment_summary.perc_enrichment * 100
#uniref_enrichment_summary.to_csv('/scratch/alexb2/generative_protein_models/share/uniref_enrichment_summary.csv',index = False)
uniref_enrichment_summary = pd.read_csv('/scratch/alexb2/generative_protein_models/share/uniref_enrichment_summary.csv')

#-----------------------------------------
#Plotting
#-----------------------------------------

mpl.rcParams['hatch.linewidth'] = 0.02
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
    'axes.titlesize': 10,       # Font size for titles
    'legend.fontsize': 6,      # Font size for legends
    'xtick.labelsize': 6,      # Font size for x-axis tick labels
    'ytick.labelsize': 6,       # Font size for y-axis tick labels
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

#Comment out the colors to exclude different models
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

def format_cbar_tick(tick):
    tick = f'{tick:.2E}'
    base = float(tick.split('E')[0])
    exp = int(tick.split('E')[-1])
    if exp <=2 and exp >= -1:
        return f'{base * (10**exp):.3G}'
    else:
        return tick


ref2015_terms = ['REF2015', 'fa_rep', 'ref', 'fa_atr', 'fa_intra_rep'
                ,  'fa_sol', 'lk_ball_wtd', 'fa_intra_sol_xover4', 'dslf_fa13', 'pro_close'
                , 'fa_elec', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc'
                , 'rama_prepro', 'p_aa_pp', 'fa_dun', 'omega',  'yhh_planarity']
#Residue order  Hydrophobic Side Chains,Charged Side Chains, Polar Uncharged Side Chains, Special
residues = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W','R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P']
summary['res'] = pd.Categorical(
    summary['res'], 
    categories=residues, 
    ordered=True
)

uniref_enrichment_summary['res'] = pd.Categorical(
    uniref_enrichment_summary['res'], 
    categories=residues, 
    ordered=True
)
uniref_enrichment_summary = uniref_enrichment_summary.sort_values('res')

aspect_ratio = 8.3 / 11.7 #A4 aspect ratio
figure_height = 11.7 #Update this for different figure scale, 11.69 for full A4.


with PdfPages("figure_2.pdf") as pdf:
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=300)
    _= fig.text(0.02, 0.985, 'A)', fontsize=10, ha='left', va='center')
    gs = GridSpec(2,2, figure=fig, left = 0.08, bottom = 0.02, right=0.97,top = 0.98, wspace = 0.08, width_ratios=[1.5,1])
    gs_LHS = gridspec.GridSpecFromSubplotSpec(11, 2, subplot_spec=gs[:,0], hspace = 0.15, wspace = 0.03)
    #gs_C = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[5:,0])
    cmap = plt.colormaps['plasma'].copy()
    cmap.set_bad(color='whitesmoke') #Want to highlight missing data with a different color
    for i, var in enumerate(ref2015_terms):
        if i > 10: i = i + 2 #Shifting after the first column so we can put in out filtered heatmaps
        heatmap_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_LHS[i%11,i//11], hspace = 0.05,height_ratios = [1,15], width_ratios = [1,1,3])
        energy_heatmap = energy.loc[(energy['term'] == var) & (energy.model.isin(colors.keys())), :].groupby(['term', 'length', 'model'])['value'].mean().reset_index().pivot(index='model', columns='length', values='value').reindex(colors.keys())
        #,left =0.01, right=0.99, top = 0.9, bottom = 0.1, hspace = 0.01, wspace = 0))
        heatmap_ax = fig.add_subplot(heatmap_gs[1,:])
        cbax = fig.add_subplot(heatmap_gs[0,2])
        im = heatmap_ax.imshow(energy_heatmap, cmap = cmap, aspect='auto',rasterized=False)
        cb = fig.colorbar(im, cax=cbax, orientation = 'horizontal')
        #Adjustments to axes
        _= (lambda ax: (
            ax.set_title(var,fontsize = 6,pad=2,loc = 'left',fontweight='bold')#
            ,ax.minorticks_off()
            ,ax.xaxis.set_tick_params(pad=3,labelsize=2)
            ,ax.set_yticks(np.arange(len(energy_heatmap.index)))
            ,ax.set_xticks(np.insert(np.arange(11, energy_heatmap.shape[1], 25),0,0))
            ,ax.yaxis.set_tick_params(pad=3,labelsize=3) 
            ,ax.set_xlabel('Monomer Length', fontsize = 4, labelpad=3)
            ,ax.set_xticklabels(['']+list(range(25,201,25)),rotation = 0, ha='right',va='center',fontsize=4)
            ,ax.set_yticklabels(energy_heatmap.index.str.replace('Protein-Generator', 'ProteinGenerator'),rotation = 0, ha='right',va='center',fontsize=4)
            ))(heatmap_ax)
        _ = [t.set_color(colors[t.get_text().replace('ProteinGenerator','Protein-Generator')]) for t in heatmap_ax.yaxis.get_ticklabels()]
        if i//11!=0:
            _= (lambda ax: (
                ax.set_yticks([])
                ,ax.set_yticklabels([])
                ))(heatmap_ax)
        if (i+1)%11!=0:
            _= (lambda ax: (
                ax.set_xticks([])
                ,ax.set_xticklabels([])
                ,ax.set_xlabel('')
                ))(heatmap_ax)
        _= (lambda ax: (
            ax.tick_params(rotation=0)
            ,ax.minorticks_off()
            ,ax.xaxis.set_ticks_position('top')
            ,ax.xaxis.set_tick_params(pad=0) 
            ,ax.set_xticks(np.linspace(cb.vmin,cb.vmax, 3))
            ,ax.set_xticklabels([format_cbar_tick(tick) for tick in cbax.get_xticks()], fontsize = 4, ha='right', rotation = 0)
            ))(cbax)
    rect = patches.Rectangle((0.337, 0.812), 0.311, 0.177, linewidth=0,edgecolor='none', facecolor='darkgrey',alpha = 0.1, transform=fig.transFigure,zorder=0)# linestyle = '--', 
    _= fig.add_artist(rect)
    for i, var in [(11,'REF2015'),(12,'fa_rep')]:
        heatmap_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_LHS[i%11,i//11], hspace = 0.05,height_ratios = [1,15], width_ratios = [1,1,3])
        energy_heatmap = energy.loc[(energy['term'] == var) & (energy.model.isin(colors.keys())), :].groupby(['term', 'length', 'model'])['value'].mean().reset_index().pivot(index='model', columns='length', values='value').reindex(colors.keys())
        energy_heatmap = energy_heatmap.loc[~energy_heatmap.index.isin(['Protein-Generator','Protpardelle','Chroma']),:]
        heatmap_ax = fig.add_subplot(heatmap_gs[1,:])
        cbax = fig.add_subplot(heatmap_gs[0,2])
        im = heatmap_ax.imshow(energy_heatmap, cmap = cmap, aspect='auto',rasterized=False)
        cb = fig.colorbar(im, cax=cbax, orientation = 'horizontal')
        #Adjustments to axes
        _= (lambda ax: (
            ax.set_title(var + ' (Subset)',fontsize = 6,pad=2,loc = 'left', color = 'dimgrey')#fontweight='bold',
            ,ax.minorticks_off()
            ,ax.set_yticks(np.arange(len(energy_heatmap.index)))
            ,ax.yaxis.tick_right()
            ,ax.xaxis.set_tick_params(pad=1,labelsize=2)
            ,ax.yaxis.set_tick_params(pad=3,labelsize=3) 
            ,ax.set_xticks([])
            ,ax.set_xticklabels([])
            ,ax.set_yticklabels(energy_heatmap.index,rotation = 0, ha='left',va='center',fontsize=4)
            ))(heatmap_ax)
        _ = [t.set_color(colors[t.get_text()]) for t in heatmap_ax.yaxis.get_ticklabels()]
        _= (lambda ax: (
            ax.tick_params(rotation=0)
            ,ax.minorticks_off()
            ,ax.xaxis.set_ticks_position('top')
            ,ax.xaxis.set_tick_params(pad=0) 
            ,ax.set_xticks(np.linspace(cb.vmin,cb.vmax, 3))
            ,ax.set_xticklabels([format_cbar_tick(tick) for tick in cbax.get_xticks()], fontsize = 4, ha='right', rotation = 0)
            ))(cbax)
    #
    gs_RHS = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[:,1], height_ratios = [1,4.6],hspace = 0.05)
    energy_summary_axis = ssax = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(3,3, subplot_spec=gs_RHS[0,0],width_ratios = [0.5,0.5,9], height_ratios = [0.5,9,2])[1,2])
    energy_summary = sns.kdeplot(data = energy.loc[energy.term == 'REF2015',['model','value']], x = 'value', hue = 'model',palette = colors, legend = False,linewidth = 0.1, common_norm = False)
    _= (lambda ax: (
            ax.minorticks_off()
            ,ax.set_title('B)',loc='left',fontsize=10)
            ,ax.tick_params(axis='x', direction='out', length=1, width=0.1, labelsize = 4,pad =3)
            ,ax.set_xlim(-1000,5000)
            ,ax.set_xticklabels(['... -1000','0','1000','2000','3000','4000','5000 ...'])
            #,ax.set_yticks([])
            #,ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)),  # Enable scientific notation
            ,ax.yaxis.set_tick_params(labelsize=4)
            ,ax.set_ylabel('Probability Density')
            ,ax.set_xlabel('REF2015')
            ,ax.yaxis.set_label_position('left')
            ,ax.spines['bottom'].set_visible(True)
            ,ax.spines['left'].set_visible(True)
        ))(energy_summary)
    gs_RHS = gridspec.GridSpecFromSubplotSpec(15,2, subplot_spec=gs_RHS[1,0], wspace = 0.05,hspace = 0.4, width_ratios = [1,1.5])
    #Plot Secondary Structure Enrichment
    for i, model in enumerate(colors.keys()):
        ssax = fig.add_subplot(gs_RHS[i+1,0])
        if i ==0: _= ssax.set_title('C)',loc='center',fontsize=10,pad=54)
        for ss_type in [('E'),('H'),('L')]:
            _= ssax.plot(ss.loc[ss.model == model,'position'], ss.loc[ss.model == model,ss_type], linestyle='-', color='black', linewidth=0.01, alpha = 1)
        for fill_params in [('L','E','|||||||'),('E','H','///////'),('H','axis', '')]:
            fill = ssax.fill_between(ss.loc[ss.model == model,'position'], ss.loc[ss.model == model,fill_params[0]],ss.loc[ss.model == model,fill_params[1]], color=colors[model], alpha=1)
            fill.set_hatch(fill_params[2])
            fill.set_edgecolor('k')
            fill.set_linewidth(0)
        fill = ssax.fill_between(ss.loc[ss.model == model,'position'], ss.loc[ss.model == model,'H'], color=colors[model], alpha=1)
        _= (lambda ax: (
            ax.minorticks_off()
            ,ax.tick_params(axis='x', direction='inout', length=1, width=0.1, labelsize = 4,pad =3)
            ,ax.set_xlim(0,200)
            ,ax.tick_params(axis='y', direction='inout', length=1, width=0.1, labelsize = 4,pad = 3)
            ,ax.yaxis.tick_left()
            ,ax.yaxis.set_label_position('right')
            ,ax.set_ylim(0,100)
            ,ax.spines['left'].set_visible(True)
            ,ax.spines['bottom'].set_visible(True)
            ,ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ))(ssax)
        if i+1 == 14: _= ssax.set_xlabel('Residue Position', fontsize = 4, labelpad=2)
    #add legend to the ss plots
    leg_ax = fig.add_axes([0.66, 0.729,0.05,0.05])
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='white',linewidth=0.1, edgecolor='dimgrey', label='α')
        ,patches.Rectangle((0, 0), 1, 1, facecolor='white',linewidth=0.1, edgecolor='dimgrey', hatch='///////', label='β')
        ,patches.Rectangle((0, 0), 1, 1, facecolor='white',linewidth=0.1, edgecolor='dimgrey', hatch='|||||||', label='L')
    ]
    legend = leg_ax.legend(handles=legend_elements, loc='center', ncol=3, fontsize=8, columnspacing=0.5)
    _= legend.get_frame().set(linewidth=0, facecolor='white')
    _= leg_ax.axis('off')
    #Plot (Overall) Amino Acid Enrichment
    for i, model in enumerate(colors.keys()):
        sax = fig.add_subplot(gs_RHS[i+1,1])
        bars = sax.bar(summary.loc[summary.model == model].sort_values('res').res, summary.loc[summary.model == model].sort_values('res').perc_enrichment,color=colors[model])
        _= (lambda ax: (
            ax.set_yticks([0, 10, 20, 30])
            ,ax.text(0,25, model.replace('Protein-Generator','ProteinGenerator'),ha='left', va='center', fontsize=9, color=colors[model])
            ,ax.tick_params(which='major', axis='x', direction='inout', length=0, width=0.1, labelsize = 4)
            ,ax.tick_params(which='minor', axis='x', direction='inout', length=0, width=0, labelsize = 4)
            ,ax.tick_params(which='minor', axis='y', direction='inout', length=0, width=0, labelsize = 4)
            ,ax.tick_params(which='major', axis='y', direction='inout', length=2, width=0.1, labelsize = 4,pad=3)
            ,ax.yaxis.tick_right()
            ,ax.yaxis.set_label_position('left')
            #,ax.set_title(model.upper(),loc='center',color=colors[model],fontsize=4,pad=2)
            #,ax.set_ylim(0,math.ceil(summary.perc_enrichment.max()))
            ,ax.spines['right'].set_visible(True)
            ,ax.spines['bottom'].set_visible(True)
            ,ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ))(sax)
        if i+1 == 14: _= sax.set_xlabel('Residue', fontsize = 4, labelpad=2)
    #Plot the additional enrichment bar for UNIREF50
    sax = fig.add_subplot(gs_RHS[0,1])
    bars = sax.bar(uniref_enrichment_summary.res, uniref_enrichment_summary.perc_enrichment,color='black')
    _= sax.set_title('D)',loc='center',fontsize=10,pad=10)
    _= (lambda ax: (
        ax.set_yticks([0, 10, 20, 30])
        ,ax.text(0,25, 'UniRef50',ha='left', va='center', fontsize=9, color='black')
        ,ax.tick_params(which='major', axis='x', direction='inout', length=0, width=0.1, labelsize = 4)
        ,ax.tick_params(which='minor', axis='x', direction='inout', length=0, width=0, labelsize = 4)
        ,ax.tick_params(which='minor', axis='y', direction='inout', length=0, width=0, labelsize = 4)
        ,ax.tick_params(which='major', axis='y', direction='inout', length=2, width=0.1, labelsize = 4,pad=3)
        ,ax.yaxis.tick_right()
        ,ax.yaxis.set_label_position('left')
        #,ax.set_title(model.upper(),loc='center',color=colors[model],fontsize=4,pad=2)
        #,ax.set_ylim(0,math.ceil(summary.perc_enrichment.max()))
        ,ax.spines['right'].set_visible(True)
        ,ax.spines['bottom'].set_visible(True)
        ,ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ))(sax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    plt.close('all')
    



