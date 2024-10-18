import pyrosetta; pyrosetta.init()
from pyrosetta import *
init(extra_options="-mute all")
import gzip
import argparse
import os
import sys
from Bio.SeqUtils import IUPACData
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

for root, dirs, files in os.walk("/scratch/alexb2/generative_protein_models/raw_data/pisces_pdb"):
        for file in files:
            if ".pdb" in file:
                print(file)
                temp_pdb_file = os.path.join("/scratch/alexb2/generative_protein_models/raw_data/pisces_pdb", file.split('.gz')[0])
                with gzip.open(file, 'rt') as f_in:
                    with open(temp_pdb_file, 'wt') as f_out:
                        f_out.write(f_in.read())
                    pyrosetta.toolbox.cleaning.cleanATOM(temp_pdb_file)
                    os.remove(temp_pdb_file)
