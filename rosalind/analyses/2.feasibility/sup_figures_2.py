import numpy as np
import pandas as pd
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from decimal import Decimal
import seaborn as sns
import roman

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

#AA enrichments for uniref
#----------------------------
#from Bio import SeqIO
#from collections import defaultdict
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
uniref_enrichment_summary = pd.read_csv('/scratch/alexb2/generative_protein_models/share/uniref_enrichment_summary.csv')


#For refolded AA outputs
#--------------------------------------------
#Load in (Per chain) Outputs from PyRosetta on our generated monomers
refolded = pd.read_csv('/scratch/alexb2/generative_protein_models/share/aa_refolds_per_chain.csv')
#Cleanup
refolded.loc[:,'filename'] = refolded.filepath.str.split('/').str[-1]
refolded.loc[:,'model'] = refolded.filename.str.split('_').str[0]
refolded['entity_id'] = refolded['filepath'].str.extract(f'({uuid_pattern})', expand=False)
#Concat into full dataset
refolded = pd.concat([refolded,pisces])
refolded.loc[:,'length'] = refolded.chain_sequence.str.len().astype(int)
refolded = refolded.loc[(refolded.length <= 200) & (refolded.length >= 14),:]

#extract energy data
refolded_energy = refolded.copy(deep=True).loc[:,['model','entity_id','length','fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro']].drop_duplicates().reset_index(drop=True)
#Full energy function is the sum of all terms
refolded_energy.loc[:,'REF2015'] = refolded_energy.iloc[:,3:].sum(axis=1)
refolded_energy = pd.melt(refolded_energy, id_vars=['model','entity_id','length'], var_name='term', value_name='value')

#Repeat enrichment calcs but for secondary structure annotations
refolded_ss = refolded.copy(deep=True).loc[:,['model','entity_id','length','ss']]['ss'].apply(lambda x: pd.Series(list(x)))
refolded_ss.columns = [i + 1 for i in range(refolded_ss.shape[1])]
refolded_ss = refolded_ss.iloc[:,0:200]
refolded_ss = pd.concat([refolded.copy(deep=True).loc[:,['model','entity_id','length']], refolded_ss], axis=1)
refolded_ss = pd.melt(refolded_ss, id_vars=['model','entity_id','length'], var_name='position', value_name='ss_anot')
refolded_ss = refolded_ss.dropna()
refolded_ss = (refolded_ss.groupby(['model','position','ss_anot']).size()/refolded_ss.groupby(['model','position']).size()).reset_index()
refolded_ss.columns = list(refolded_ss.columns[:-1]) + ['perc_ss_enrichment']
refolded_ss.perc_ss_enrichment = refolded_ss.perc_ss_enrichment * 100
refolded_ss = pd.DataFrame([(mod, pos, anot) for mod in refolded_ss.model.unique() for pos in refolded_ss.position.unique() for anot in refolded_ss.ss_anot.dropna().unique()], columns=['model', 'position','ss_anot']).merge(refolded_ss, on=['model', 'position','ss_anot'], how='outer')
refolded_ss.perc_ss_enrichment = refolded_ss.perc_ss_enrichment.fillna(0)
refolded_ss = refolded_ss[~refolded_ss[['model', 'position']].apply(tuple, axis=1).isin(drop.apply(tuple, axis=1))]
refolded_ss = refolded_ss.pivot(index=['model','position'], columns='ss_anot',values='perc_ss_enrichment').fillna(0)
refolded_ss.E = refolded_ss.H + refolded_ss.E
refolded_ss.L = refolded_ss.E + refolded_ss.L
refolded_ss.reset_index(inplace=True)
refolded_ss['axis'] = 0


#Load in (Per res) Outputs from PyRosetta on our generated monomers
generated = pd.read_csv('/scratch/alexb2/generative_protein_models/share/14_200_per_res.csv')
#Cleanup
generated.loc[:,'filename'] = generated.filepath.str.split('/').str[-1]
generated.loc[:,'model'] = generated.filename.str.split('_').str[0]
uuid_pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
generated['entity_id'] = generated['filepath'].str.extract(f'({uuid_pattern})', expand=False)
pisces = pd.read_csv('/scratch/alexb2/generative_protein_models/share/pisces_reference_per_res.csv')
#Cleanup
pisces['model'] = 'PISCES'
pisces['entity_id'] = pisces['filepath'].str.split('/').str[-1].str.split('.').str[0] + "_chain" + pisces.chain_no.astype(str)
#Concat into full dataset
data = pd.concat([generated,pisces])
data.loc[:,'length'] = data.chain_sequence.str.len().astype(int)
data = data.loc[(data.length <= 200) & (data.length >= 14),:]

#-----------------------------------------
#Plotting
#-----------------------------------------

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

ref2015_terms = ['REF2015', 'fa_rep', 'ref', 'fa_atr', 'fa_intra_rep'
                ,  'fa_sol', 'lk_ball_wtd', 'fa_intra_sol_xover4', 'dslf_fa13', 'pro_close'
                , 'fa_elec', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc'
                , 'rama_prepro', 'p_aa_pp', 'fa_dun', 'omega',  'yhh_planarity']
ONE_CHI ="CEKSTV"
TWO_CHI ="DFHILNPWY"
THREE_CHI ="EMQ"
FOUR_CHI ="KR"

aspect_ratio = 8.3 / 11.7 #A4 aspect ratio
figure_height = 11.7 #Update this for different figure scale, 11.69 for full A4.


#heatmap = enrichment.loc[(enrichment.res == res) & (energy.model.isin(colors.keys())),:].pivot(index='model', columns = ['position'], values = 'perc_enrichment').reindex(colors.keys())#.fillna(0)

def format_cbar_tick(tick):
    tick = f'{tick:.2E}'
    base = float(tick.split('E')[0])
    exp = int(tick.split('E')[-1])
    if exp <=2 and exp >= -1:
        return f'{base * (10**exp):.3G}'
    else:
        return tick

def plot_heatmap(fig, gs, data, heatmap_lambda, voffset,cmap, colname, colvals,xaxes_label, cbscale = None):
    for i, var in enumerate(colvals):
        axes_y_start = 3 + voffset + 36*(i%7)
        axes_y_end = axes_y_start+26
        axes_x_start = 13 + ((i//7)* 115)
        axes_x_end = axes_x_start + 114
        #
        cbaxes_y_start = axes_y_start -3
        cbaxes_y_end = cbaxes_y_start + 2
        cbaxes_x_start = axes_x_start + 5
        cbaxes_x_end = axes_x_end - 5
        #Heatmap & colorbar
        heatmap = heatmap_lambda(data,colname, var)
        ax = fig.add_subplot(gs[axes_y_start:axes_y_end,axes_x_start:axes_x_end])
        cbax = fig.add_subplot(gs[cbaxes_y_start:cbaxes_y_end,cbaxes_x_start:cbaxes_x_end])
        if cbscale:
            im = ax.imshow(heatmap, cmap = cmap, norm = mcolors.Normalize(vmin=cbscale[0],vmax = cbscale[1]), aspect='auto',rasterized=False)
            cb = fig.colorbar(im, cax=cbax, orientation = 'horizontal')
        else:
            im = ax.imshow(heatmap, cmap = cmap, aspect='auto',rasterized=False)
            cb = fig.colorbar(im, cax=cbax, orientation = 'horizontal')
        #Adjustments to axes
        _= (lambda ax: (
            ax.set_title(var,fontsize = 5,pad=15)
            ,ax.minorticks_off()
            ,ax.xaxis.set_tick_params(pad=1,labelsize=2)
            ,ax.set_yticks(np.arange(len(heatmap.index)))
            #,ax.set_xticks(np.insert(np.arange(11, heatmap.shape[1], 25),0,0))
            ,ax.yaxis.set_tick_params(pad=0,labelsize=3) 
            ,ax.set_xlabel(xaxes_label, fontsize = 4, labelpad=3)
            #,ax.set_xticklabels(['']+list(range(25,201,25)),rotation = 0, ha='center',va='center')
            ,ax.set_yticklabels(heatmap.index.str.replace('Protein-Generator', 'ProteinGenerator'),rotation = 0, ha='right',va='center')
            ))(ax)
        [t.set_color(colors[t.get_text().replace('ProteinGenerator', 'Protein-Generator')]) for t in ax.yaxis.get_ticklabels()]
        if i//7!=0:
            _= (lambda ax: (
                ax.set_yticks([])
                ,ax.set_yticklabels([])
                ))(ax)
        if (i+1)%7!=0 and i!=19:
            _= (lambda ax: (
                ax.set_xticks([])
                ,ax.set_xticklabels([])
                ,ax.set_xlabel('')
                ))(ax)
        _= (lambda ax: (
            ax.tick_params(rotation=0)
            ,ax.minorticks_off()
            ,ax.xaxis.set_ticks_position('top')
            ,ax.xaxis.set_tick_params(pad=0) 
            ,ax.set_xticks(np.linspace(cb.vmin,cb.vmax, 3))
            ,ax.set_xticklabels([format_cbar_tick(tick) for tick in cbax.get_xticks()], fontsize = 3, ha='center', rotation = 0)
            ))(cbax)

with PdfPages("sup_figures_2.pdf") as pdf:
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=300)
    _= fig.text(0.02, 0.985, 'A)', fontsize=10, ha='left', va='center')
    gs = GridSpec(2,1, figure=fig, left = 0.08, bottom = 0.02, right=0.97,top = 0.98, wspace = 0.08, height_ratios=[1.5,1])
    gs_upper = gridspec.GridSpecFromSubplotSpec(7, 3, subplot_spec=gs[0,0], hspace = 0.2, wspace = 0.05)
    cmap = plt.colormaps['plasma'].copy()
    cmap.set_bad(color='whitesmoke') #Want to highlight missing data with a different color
    for i, var in enumerate(ref2015_terms):
        heatmap_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_upper[i%7,i//7], hspace = 0.05,height_ratios = [1,15], width_ratios = [1,1,3])
        refolded_energy_heatmap = refolded_energy.loc[(refolded_energy['term'] == var) & (refolded_energy.model.isin(colors.keys())), :].groupby(['term', 'length', 'model'])['value'].mean().reset_index().pivot(index='model', columns='length', values='value').reindex(['PISCES','Chroma','Protpardelle','Protein-Generator'])
        #,left =0.01, right=0.99, top = 0.9, bottom = 0.1, hspace = 0.01, wspace = 0))
        heatmap_ax = fig.add_subplot(heatmap_gs[1,:])
        cbax = fig.add_subplot(heatmap_gs[0,2])
        im = heatmap_ax.imshow(refolded_energy_heatmap, cmap = cmap, aspect='auto',rasterized=False)
        cb = fig.colorbar(im, cax=cbax, orientation = 'horizontal')
        #Adjustments to axes
        _= (lambda ax: (
            ax.set_title(var,fontsize = 6,pad=2,loc = 'left',fontweight='bold')
            ,ax.minorticks_off()
            ,ax.xaxis.set_tick_params(pad=3,labelsize=2)
            ,ax.set_yticks(np.arange(len(refolded_energy_heatmap.index)))
            ,ax.set_xticks(np.insert(np.arange(11, refolded_energy_heatmap.shape[1], 25),0,0))
            ,ax.yaxis.set_tick_params(pad=3,labelsize=3) 
            ,ax.set_xlabel('Monomer Length', fontsize = 4, labelpad=3)
            ,ax.set_xticklabels(['']+list(range(25,201,25)),rotation = 0, ha='right',va='center',fontsize=4)
            ,ax.set_yticklabels(refolded_energy_heatmap.index.str.replace('Protein-Generator', 'ProteinGenerator'),rotation = 0, ha='right',va='center',fontsize=4)
            ))(heatmap_ax)
        _ = [t.set_color(colors[t.get_text().replace('ProteinGenerator', 'Protein-Generator')]) for t in heatmap_ax.yaxis.get_ticklabels()]
        if i//7!=0:
            _= (lambda ax: (
                ax.set_yticks([])
                ,ax.set_yticklabels([])
                ))(heatmap_ax)
        if (i+1)%7!=0 and (i!=19):
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
    #
    energy_summary_axis = fig.add_subplot(gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs_upper[6,2],width_ratios = [0.5,0.05],height_ratios = [0.2,0.8])[1,0])
    refolded_energy_summary = sns.kdeplot(data = refolded_energy.loc[refolded_energy.term == 'REF2015',['model','value']], x = 'value', hue = 'model',palette = colors, legend = False,linewidth = 0.1, common_norm = False)
    _= (lambda ax: (
            ax.minorticks_off()
            ,ax.set_title('B)',loc='left',fontsize=10)
            ,ax.tick_params(axis='x', direction='out', length=1, width=0.1, labelsize = 4,pad =3)
            ,ax.set_xlim(-1000,5000)
            ,ax.set_xticklabels(['... -1000','0','1000','2000','3000','4000','5000 ...'])
            #,ax.set_yticks([])
            ,ax.yaxis.set_tick_params(labelsize=4)
            ,ax.set_ylabel('Probability Density')
            ,ax.set_xlabel('REF2015')
            ,ax.yaxis.set_label_position('right')
            ,ax.yaxis.tick_right()
            ,ax.spines['bottom'].set_visible(True)
            ,ax.spines['right'].set_visible(True)
        ))(energy_summary_axis)
    gs_ss = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=gs[1,0], wspace = 0.1,hspace = 0.1)
    #Plot Secondary Structure Enrichment
    for i, model in enumerate(['PISCES','Chroma','Protpardelle','Protein-Generator']):
        ssax = fig.add_subplot(gs_ss[i//2,i%2])
        if i ==0: _= ssax.set_title('C)',loc='Left',fontsize=10,pad=20)
        for ss_type in [('E'),('H'),('L')]:
            _= ssax.plot(refolded_ss.loc[refolded_ss.model == model,'position'], refolded_ss.loc[refolded_ss.model == model,ss_type], linestyle='-', color='black', linewidth=0.01, alpha = 1)
        for fill_params in [('L','E','|||||||'),('E','H','///////'),('H','axis', '')]:
            fill = ssax.fill_between(refolded_ss.loc[refolded_ss.model == model,'position'], refolded_ss.loc[refolded_ss.model == model,fill_params[0]],refolded_ss.loc[refolded_ss.model == model,fill_params[1]], color=colors[model], alpha=1)
            fill.set_hatch(fill_params[2])
            fill.set_edgecolor('k')
            fill.set_linewidth(0)
        fill = ssax.fill_between(refolded_ss.loc[refolded_ss.model == model,'position'], refolded_ss.loc[refolded_ss.model == model,'H'], color=colors[model], alpha=1)
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
        if i in [1,3]: _= ssax.set_xlabel('Residue Position', fontsize = 4, labelpad=2)
    #add legend to the ss plots
    leg_ax = fig.add_axes([0.4, 0.35,0.25,0.1])
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='white',linewidth=0.1, edgecolor='dimgrey', label='α')
        ,patches.Rectangle((0, 0), 1, 1, facecolor='white',linewidth=0.1, edgecolor='dimgrey', hatch='///////', label='β')
        ,patches.Rectangle((0, 0), 1, 1, facecolor='white',linewidth=0.1, edgecolor='dimgrey', hatch='|||||||', label='L')
        ,Line2D([0], [0], color='#809a2d', lw=2, label='Chroma')
        ,Line2D([0], [0], color='#ffc73c', lw=2, label='Protpardelle')
        ,Line2D([0], [0], color='#a4837b', lw=2, label='ProteinGenerator')
    ]
    legend = leg_ax.legend(handles=legend_elements, loc='center', ncol=6, fontsize=8)
    _= legend.get_frame().set(linewidth=0, facecolor='white')
    _= leg_ax.axis('off')
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
    leg_ax = fig.add_axes([0.89,0.83,0.1,0.1])
    legend_elements = [
        Line2D([0], [0], color='dimgrey', lw=2, label='PISCES'),
        Line2D([0], [0], color='#005cdf', lw=2, label='RFdiffusion'),
        Line2D([0], [0], color='#b3f4ff', lw=2, label='Genie'),
        Line2D([0], [0], color='#586386', lw=2, label='ProteinSGM'),
        Line2D([0], [0], color='#7373ae', lw=2, label='FoldingDiff'),
        Line2D([0], [0], color='#30aa8f', lw=2, label='FrameDiff'),
        Line2D([0], [0], color='#809a2d', lw=2, label='Chroma'),
        Line2D([0], [0], color='#ffc73c', lw=2, label='Protpardelle'),
        Line2D([0], [0], color='#a4837b', lw=2, label='ProteinGenerator'),
        Line2D([0], [0], color='#93355c', lw=2, label='EvoDiff'),
        Line2D([0], [0], color='#cf4f00', lw=2, label='RITA'),
        Line2D([0], [0], color='#e44fff', lw=2, label='ProGen2'),
        Line2D([0], [0], color='#9e0049', lw=2, label='ProtGPT2'),
        Line2D([0], [0], color='#ff7bb9', lw=2, label='ESM-Design'),
        ]
    legend = leg_ax.legend(handles=legend_elements, loc='center', ncol=1, fontsize=6)
    _= legend.get_frame().set_linewidth(0)
    _= legend.get_frame().set_facecolor('white') 
    _= leg_ax.axis('off')
    gs = GridSpec((len(ref2015_terms)//2),2, figure=fig, left =0.1, right=0.85,top = 0.95, bottom = 0.05, hspace = 0.6)
    for i,term in enumerate(ref2015_terms):
        ax = fig.add_subplot(gs[i%(len(ref2015_terms)//2),i//(len(ref2015_terms)//2)])
        ax = sns.kdeplot(data = energy.loc[energy.term == term,['model','value']], x = 'value', hue = 'model',palette = colors, legend = False,linewidth = 0.1, common_norm = False)
        _= (lambda ax: (
            ax.minorticks_off()
            #,ax.set_title(term,loc='center',fontsize=7)
            ,ax.tick_params(axis='x', direction='out', length=1, width=0.1, labelsize = 4,pad =3)
            #,ax.set_xlim(-1000,5000)
            #,ax.set_xticklabels(['... -1000','0','1000','2000','3000','4000','5000 ...'])
            #,ax.set_yticks([])
            ,ax.set_ylabel('Probability Density')
            ,ax.set_xlabel(term,fontsize = 7)
            #,ax.yaxis.set_label_position('right')
            ,ax.spines['bottom'].set_visible(True)
            ,ax.spines['left'].set_visible(True)
        ))(ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #For interactive display
    #Monitor size = 34.1 inch (diagonal), 3440 x 1440 resolution. DPI = sqrt(width^2 + height^2)/diagonal = 109.36
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
    gs = GridSpec(360,360, figure=fig, left =0.01, right=0.995,top = 0.97, bottom = 0.01)
    #AA enrichments
    heatmap_lambda = lambda data, colname, var: data.loc[(data[colname] == var) & (data.model.isin(colors.keys())), :].pivot(index='model', columns = ['position'], values = 'perc_enrichment').reindex(colors.keys())
    cmap = plt.colormaps['hot'].copy()
    cmap.set_bad(color='grey') #Want to highlight missing data with a different color
    plot_heatmap(fig, gs, enrichment, heatmap_lambda, 0,cmap, 'res', ['% ' + res for res in residues], 'Residue Position')#, (0,100))
    #Plot summary enrichment data
    for i, model in enumerate([list(colors.keys())[i//2 + (7 * (i % 2))] for i in range(14)]):
        sax = fig.add_subplot(gs[260 + (60*(i%2)):300+(60*(i%2)), 17+50*(i//2):55+50*(i//2)])
        bars = sax.bar(summary.loc[summary.model == model].sort_values('res').res, summary.loc[summary.model == model].sort_values('res').perc_enrichment,color=colors[model])
        _= (lambda ax: (
            ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
            ,ax.tick_params(which='major', axis='x', direction='inout', length=0, width=0.1, labelsize = 3)
            ,ax.tick_params(which='minor', axis='x', direction='inout', length=0, width=0, labelsize = 3)
            ,ax.tick_params(which='minor', axis='y', direction='inout', length=0, width=0, labelsize = 3)
            ,ax.tick_params(which='major', axis='y', direction='inout', length=2, width=0.1, labelsize = 3,pad=1)
            ,ax.yaxis.tick_left()
            ,ax.yaxis.set_label_position('left')
            ,ax.set_ylabel('Total Enrichment')
            ,ax.set_title(model.replace('Protein-Generator','ProteinGenerator'),loc='center',color=colors[model],fontsize=8,pad=5)
            #,ax.set_ylim(0,math.ceil(summary.perc_enrichment.max()))
            ,ax.spines['left'].set_visible(True)
            ,ax.spines['bottom'].set_visible(True)
            ,ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ))(sax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
    gs = GridSpec(14,3, figure=fig, left =0.05, right=0.95,top = 0.95, bottom = 0.05,wspace=0.25,hspace=0.8)
    _= fig.text(0.18, 0.98, '\u03C6', fontsize=10, ha='center', va='center')
    _= fig.text(0.5, 0.98, '\u03C8', fontsize=10, ha='center', va='center')
    _= fig.text(0.82, 0.98, '\u03C9', fontsize=10, ha='center', va='center')
    for i, angle in enumerate(['phi_angle','psi_angle','w_angle']):
        for j, model in enumerate(colors.keys()):
            ax = fig.add_subplot(gs[j,i])
            _=ax.hist(data.loc[data.model == model,angle], bins = list(range(-180,181,1)),density=True,color = colors[model])
            _= (lambda ax: (
            ax.spines['left'].set_visible(True)
            ,ax.spines['bottom'].set_visible(True)
            ,ax.set_xlim(-360,360)
            #,ax.set_ylim(0,0.2)
            ,ax.set_xticks(np.arange(-360,361,90))
            ))(ax)
            if i == 0:
                _= (lambda ax: (
                        ax.set_title(f"{model}",loc='left',color=colors[model],fontsize=8,pad=5, x = -0.15,ha='left')
                        ))(ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    #
    fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
    gs = GridSpec(14,4, figure=fig, left =0.05, right=0.95,top = 0.95, bottom = 0.05,wspace=0.25,hspace=0.8)
    _= fig.text(0.14, 0.98, 'N - C\u03B1', fontsize=10, ha='center', va='center')
    _= fig.text(0.38, 0.98, 'C\u03B1 - C', fontsize=10, ha='center', va='center')
    _= fig.text(0.62, 0.98, 'C - N', fontsize=10, ha='center', va='center')
    _= fig.text(0.86, 0.98, 'C - O', fontsize=10, ha='center', va='center')
    for i, bond_length in enumerate(['N_CA','CA_C','C_N','C_O']):
        for j, model in enumerate(colors.keys()):
            ax = fig.add_subplot(gs[j,i])
            _=ax.hist(data.loc[(data.model == model),bond_length], bins = np.arange(0,math.ceil(data['CA_C'].max())+0.001,0.001),density=True,color = colors[model])
            _= (lambda ax: (
            ax.spines['left'].set_visible(True)
            ,ax.spines['bottom'].set_visible(True)
            #,ax.set_xlim(0,math.ceil(data['CA_C'].max()))
            ,ax.set_xlabel('\u212B', fontsize=5)
            ,ax.set_xlim([1.3,1.4,1,1.1][i],[1.6,1.6,2.6,1.4][i])
            #,ax.set_xticks(np.arange(-360,361,90))
            ))(ax)
            if i == 0:
                _= (lambda ax: (
                    ax.set_title(f"{model}",loc='left',color=colors[model],fontsize=8,pad=5, x = -0.2,ha='left')
                    ))(ax)
    pdf.savefig(fig,dpi = 300)
    plt.close(fig)
    plt.clf()
    k = 1
    for res in ONE_CHI:
        print(res)
        fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
        gs = GridSpec(14,1, figure=fig, left =0.05, right=0.95,top = 0.95, bottom = 0.05,wspace=0.25,hspace=0.8)
        _= fig.text(0.05, 0.98, f'{roman.toRoman(k)}) {res}', fontsize=16, ha='center', va='center')
        k = k+1
        _= fig.text(0.5, 0.98, '\u03C7 1', fontsize=10, ha='center', va='center')
        for i, angle in enumerate(['chi_1']):
            for j, model in enumerate(colors.keys()):
                ax = fig.add_subplot(gs[j,i])
                _=ax.hist(data.loc[(data.model == model) & (data.res == res),angle], bins = list(range(-180,181,1)),density=True,color = colors[model])
                _= (lambda ax: (
                ax.spines['left'].set_visible(True)
                ,ax.spines['bottom'].set_visible(True)
                ,ax.set_xlim(-360,360)
                #,ax.set_ylim(0,0.2)
                ,ax.set_xticks(np.arange(-360,361,90))
                ))(ax)
                if i == 0:
                    _= (lambda ax: (
                        ax.set_title(f"{model}",loc='left',color=colors[model],fontsize=8,pad=5, x = -0.04,ha='left')
                        ))(ax)
        pdf.savefig(fig,dpi = 300)
        plt.close(fig)
        plt.clf()
    for res in TWO_CHI:
        print(res)
        fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
        gs = GridSpec(14,2, figure=fig, left =0.05, right=0.95,top = 0.95, bottom = 0.05,wspace=0.25,hspace=0.8)
        _= fig.text(0.05, 0.98, f'{roman.toRoman(k)}) {res}', fontsize=16, ha='center', va='center')
        k = k+1
        _= fig.text(0.25, 0.98, '\u03C7 1', fontsize=10, ha='center', va='center')
        _= fig.text(0.75, 0.98, '\u03C7 2', fontsize=10, ha='center', va='center')
        for i, angle in enumerate(['chi_1','chi_2']):
            for j, model in enumerate(colors.keys()):
                ax = fig.add_subplot(gs[j,i])
                _=ax.hist(data.loc[(data.model == model) & (data.res == res),angle], bins = list(range(-180,181,1)),density=True,color = colors[model])
                _= (lambda ax: (
                ax.spines['left'].set_visible(True)
                ,ax.spines['bottom'].set_visible(True)
                ,ax.set_xlim(-360,360)
                #,ax.set_ylim(0,0.2)
                ,ax.set_xticks(np.arange(-360,361,90))
                ))(ax)
                if i == 0:
                    _= (lambda ax: (
                        ax.set_title(f"{model}",loc='center',color=colors[model],fontsize=8,pad=5, x = -0.01,ha='left')
                        ))(ax)
        pdf.savefig(fig,dpi = 300)
        plt.close(fig)
        plt.clf()
    for res in THREE_CHI:
        print(res)
        fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
        gs = GridSpec(14,3, figure=fig, left =0.05, right=0.95,top = 0.95, bottom = 0.05,wspace=0.25,hspace=0.8)
        _= fig.text(0.05, 0.98, f'{roman.toRoman(k)}) {res}', fontsize=16, ha='center', va='center')
        k = k+1
        _= fig.text(0.18, 0.98, '\u03C7 1', fontsize=10, ha='center', va='center')
        _= fig.text(0.5, 0.98, '\u03C7 2', fontsize=10, ha='center', va='center')
        _= fig.text(0.82, 0.98, '\u03C7 3', fontsize=10, ha='center', va='center')
        for i, angle in enumerate(['chi_1','chi_2','chi_3']):
            for j, model in enumerate(colors.keys()):
                ax = fig.add_subplot(gs[j,i])
                _=ax.hist(data.loc[(data.model == model) & (data.res == res),angle], bins = list(range(-180,181,1)),density=True,color = colors[model])
                _= (lambda ax: (
                ax.spines['left'].set_visible(True)
                ,ax.spines['bottom'].set_visible(True)
                ,ax.set_xlim(-360,360)
                #,ax.set_ylim(0,0.2)
                ,ax.set_xticks(np.arange(-360,361,90))
                ))(ax)
                if i == 0:
                    _= (lambda ax: (
                        ax.set_title(f"{model}",loc='right',color=colors[model],fontsize=8,pad=5, x = -0.15,ha='left')
                        ))(ax)
        pdf.savefig(fig,dpi = 300)
        plt.close(fig)
        plt.clf()
    for res in FOUR_CHI:
        print(res)
        fig = plt.figure(figsize=(aspect_ratio*figure_height,figure_height),dpi=109.36)
        gs = GridSpec(14,4, figure=fig, left =0.05, right=0.95,top = 0.95, bottom = 0.05,wspace=0.25,hspace=0.8)
        _= fig.text(0.05, 0.98, f'{roman.toRoman(k)}) {res}', fontsize=16, ha='center', va='center')
        k = k+1
        _= fig.text(0.14, 0.98, '\u03C7 1', fontsize=10, ha='center', va='center')
        _= fig.text(0.38, 0.98, '\u03C7 2', fontsize=10, ha='center', va='center')
        _= fig.text(0.62, 0.98, '\u03C7 3', fontsize=10, ha='center', va='center')
        _= fig.text(0.86, 0.98, '\u03C7 4', fontsize=10, ha='center', va='center')
        for i, angle in enumerate(['chi_1','chi_2','chi_3','chi_4']):
            for j, model in enumerate(colors.keys()):
                ax = fig.add_subplot(gs[j,i])
                _=ax.hist(data.loc[(data.model == model) & (data.res == res),angle], bins = list(range(-180,181,1)),density=True,color = colors[model])
                _= (lambda ax: (
                ax.spines['left'].set_visible(True)
                ,ax.spines['bottom'].set_visible(True)
                ,ax.set_xlim(-360,360)
                #,ax.set_ylim(0,0.2)
                ,ax.set_xticks(np.arange(-360,361,180))
                ))(ax)
                if i == 0:
                    _= (lambda ax: (
                        ax.set_title(f"{model}",loc='left',color=colors[model],fontsize=8,pad=5, x = -0.2,ha='left')
                        ))(ax)
        pdf.savefig(fig,dpi = 300)
        plt.close(fig)
        plt.clf()
    plt.close('all')

