import pyrosetta; pyrosetta.init()
from pyrosetta import *
init(extra_options="-mute all")

sfxn = get_score_function(True)

import argparse
import os
import sys
from Bio.SeqUtils import IUPACData
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
current_time = datetime.now()
timestamp_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

#Reference for which residue have which chi angles
#http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html#:~:text=SIDE%20CHAIN%20DIHEDRAL%20ANGLES&text=The%20angle%20chi2%20is%20not,is%20calculated%20only%20for%20ARG.
ONE_CHI ="CDEFHIKLMNPQRSTVWY"
TWO_CHI ="DEFHIKLMNPQRWY"
THREE_CHI ="EKMQR"
FOUR_CHI ="KR"

def phi_angle(chain):
    return [None] + [pyrosetta.rosetta.core.pose.get_bb_torsion(1,chain,j)for j in range(2,chain.total_residue()+1)]


def psi_angle(chain):
    return [pyrosetta.rosetta.core.pose.get_bb_torsion(2,chain,j)for j in range(1,chain.total_residue())] + [None]


def w_angle(chain):
    return [pyrosetta.rosetta.core.pose.get_bb_torsion(3,chain,j)for j in range(1,chain.total_residue())] + [None]


def chi_1(chain):
    return [chain.chi(1, j) if chain.sequence()[j - 1] in ONE_CHI else None for j in range(1, chain.total_residue() + 1)]


def chi_2(chain):
    return [chain.chi(2, j) if chain.sequence()[j - 1] in TWO_CHI else None for j in range(1, chain.total_residue() + 1)]


def chi_3(chain):
    return [chain.chi(3, j) if chain.sequence()[j - 1] in THREE_CHI else None for j in range(1, chain.total_residue() + 1)]


def chi_4(chain):
    return [chain.chi(4, j) if chain.sequence()[j - 1] in FOUR_CHI else None for j in range(1, chain.total_residue() + 1)]


def N_CA(chain):
    return [(chain.residue(j).xyz("CA") - chain.residue(j).xyz("N")).norm() for j in range(1,chain.total_residue()+1)]


def CA_C(chain):
    return [(chain.residue(j).xyz("C") - chain.residue(j).xyz("CA")).norm() for j in range(1,chain.total_residue()+1)]


def C_N(chain):
    return [(chain.residue(j+1).xyz("N") - chain.residue(j).xyz("C")).norm() for j in range(1,chain.total_residue())] + [None]


def C_O(chain):
    return [(chain.residue(j).xyz("O") - chain.residue(j).xyz("C")).norm() for j in range(1,chain.total_residue()+1)]


def res(chain):
    return list(chain.sequence())


def res_no(chain):
    return list(range(0,len(chain.sequence())))


def prev_res(chain):
    return [None] + list(chain.sequence())[:-1]


def next_res(chain):
    return list(chain.sequence())[1:] + [None]


def fa_atr(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_atr'))


def fa_rep(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_rep'))


def fa_sol(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_sol'))


def fa_intra_rep(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_intra_rep'))


def fa_intra_sol_xover4(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_intra_sol_xover4'))


def lk_ball_wtd(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'lk_ball_wtd'))


def fa_elec(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_elec'))


def pro_close(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'pro_close'))


def hbond_sr_bb(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'hbond_sr_bb'))


def hbond_lr_bb(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'hbond_lr_bb'))


def hbond_bb_sc(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'hbond_bb_sc'))


def hbond_sc(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'hbond_sc'))


def dslf_fa13(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'dslf_fa13'))


def omega(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'omega'))


def fa_dun(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'fa_dun'))


def p_aa_pp(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'p_aa_pp'))


def yhh_planarity(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'yhh_planarity'))


def ref(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'ref'))


def rama_prepro(chain):
    return sfxn.score_by_scoretype(chain,getattr(rosetta.core.scoring.ScoreType, 'rama_prepro'))

def ss(chain):
    chain.display_secstruct() #Need to run this first otherwise will just return 'L' for everything
    return chain.secstruct()


def main(args):
    res_features = args.res_features.split(',')
    chain_features = args.chain_features.split(',')
    data_dir = args.path_to_dataset
    output_dir = args.output_dir
    output_prefix = args.output_prefix
    chain_out = os.path.join(output_dir, output_prefix + "_per_chain.csv")
    res_out = os.path.join(output_dir, output_prefix + "_per_res.csv")
    per_chain_processed = list(pd.read_csv(chain_out)['filepath'].unique()) if Path(chain_out).exists() else []
    per_res_processed = list(pd.read_csv(res_out)['filepath'].unique()) if Path(res_out).exists() else []

    for file in os.listdir(data_dir):
        if ".pdb" in file:
            filepath = os.path.join(data_dir, file)
            #if filepath.split('/')[-1].split('.')[0] in [f.split('/')[-1].split('.')[0] for f in processed['filepath'].unique()]: continue #check if PDBID has already been added
            print(filepath)
            if not filepath in per_res_processed:
                print("Calculating Per-res features...")
                per_res = {}
                for feature in res_features:
                    per_res[feature] = []
                per_res['filepath'] = []
                per_res['chain_no'] = []
                per_res['chain_sequence'] = []
                try:
                    pose = pose_from_pdb(filepath)
                    num_chains = pose.num_chains()
                    for i in range(1,num_chains+1):
                        chain = pose.split_by_chain(i)
                        if all([res in IUPACData.protein_letters for res in chain.sequence()]):
                            print(f"Chain {i}...")
                            for feature in res_features:
                                method = globals().get(feature)
                                if method:
                                    per_res[feature].extend(method(chain))
                                else:
                                    print(f"Error: Method '{feature}' not found.")
                                    sys.exit()
                            per_res['filepath'].extend([filepath]*(len(chain.sequence())))
                            per_res['chain_no'].extend([i]*(len(chain.sequence())))
                            per_res['chain_sequence'].extend([chain.sequence()]*(len(chain.sequence())))
                        else:
                            print(f"Warning - Chain {str(i)} of {str(num_chains)} in {filepath} is not wholely IUPACData.protein_letters {chain.sequence()}, skipping...")
                except RuntimeError as e:
                    print(f"Error loading PDB file '{filepath}': {e}")
                    continue
                per_res = pd.DataFrame(per_res)
                if not per_res.empty:
                    per_res.to_csv(res_out, mode='w' if not Path(res_out).exists() else 'a', header=not Path(res_out).exists(), index=False)
            
            if not filepath in per_chain_processed:
                print("Calculating per-chain features...")
                per_chain = {}
                for feature in chain_features:
                    per_chain[feature] = []
                per_chain['filepath'] = []
                per_chain['chain_no'] = []
                per_chain['chain_sequence'] = []
                try:
                    pose = pose_from_pdb(filepath)
                    num_chains = pose.num_chains()
                    for i in range(1,num_chains+1):
                        chain = pose.split_by_chain(i)
                        if all([res in IUPACData.protein_letters for res in chain.sequence()]):
                            print(f"Chain {i}...")
                            for feature in chain_features:
                                method = globals().get(feature)
                                if method:
                                    per_chain[feature].append(method(chain))
                                else:
                                    print(f"Error: Method '{feature}' not found.")
                                    sys.exit()
                            per_chain['filepath'].append(filepath)
                            per_chain['chain_no'].append(i)
                            per_chain['chain_sequence'].append(chain.sequence())
                        else:
                            print(f"Warning - Chain {str(i)} of {str(num_chains)} in {filepath} is not wholely IUPACData.protein_letters {chain.sequence()}, skipping...")
                except RuntimeError as e:
                    print(f"Error loading PDB file '{filepath}': {e}")
                    continue
                per_chain = pd.DataFrame(per_chain)
                if not per_chain.empty:
                    per_chain.to_csv(chain_out, mode='w' if not Path(chain_out).exists() else 'a', header=not Path(chain_out).exists(), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_to_dataset',type=str, help='Path to directory containing .pdb or .pdb.gz files')
    parser.add_argument('--output_dir',type=str, help='Path to output directory')
    parser.add_argument('--output_prefix',type=str, help='Prefix of output_files')
    parser.add_argument('--res_features', type=str, help='Comma seperated list of the geometric feature to collect data on. Options include: ...')
    parser.add_argument('--chain_features', type=str, help='Comma seperated list of the geometric feature to collect data on. Options include: ...')
    args = parser.parse_args()
    main(args)


#filepath = "/scratch/alexb2/generative_protein_models/raw_data/pisces_pdb/1A1X.pdb.gz"
#pose = pose_from_pdb(filepath)
#pose = pyrosetta.toolbox.rcsb.pose_from_rcsb("1CA2")
#num_chains = pose.num_chains()
#for i in range(1,num_chains+1):
#    print(i)
#    chain = pose.split_by_chain(i)
#    print(chain.sequence())
#    print(all([res in IUPACData.protein_letters for res in chain.sequence()]))


"""
/scratch/alexb2/generative_protein_models/raw_data/pisces_pdb/6DWD.pdb.gz
core.import_pose.import_pose: File '/scratch/alexb2/generative_protein_models/raw_data/pisces_pdb/6DWD.pdb.gz' automatically determined to be of type PDB
core.chemical.GlobalResidueTypeSet: Loading (but possibly not actually using) 'HDV' from the PDB components dictionary for residue type 'pdb_HDV'

ERROR: Unrecognized residue: GTP
ERROR:: Exit from: /home/benchmark/rosetta/source/src/core/io/pose_from_sfr/PoseFromSFRBuilder.cc line: 1821
Traceback (most recent call last):
  File "/scratch/alexb2/generative_protein_models/analyses/feasibility/fetch_geometric_data.py", line 77, in <module>
    main(args)
  File "/scratch/alexb2/generative_protein_models/analyses/feasibility/fetch_geometric_data.py", line 53, in main
    pose = pose_from_pdb(filepath)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/alexb2/conda_envs/pyrosetta/lib/python3.12/site-packages/pyrosetta/io/__init__.py", line 20, in pose_from_pdb
    return pose_from_file(filename)
           ^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError:

File: /home/benchmark/rosetta/source/src/core/io/pose_from_sfr/PoseFromSFRBuilder.cc:1821
[ ERROR ] UtilityExitException
ERROR: Unrecognized residue: GTP
"""