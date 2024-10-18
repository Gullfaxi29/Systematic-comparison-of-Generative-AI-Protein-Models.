
#Can't get this working on rosalind (env issues) so I've just been running this in pymol (windows)
import os
from pymol import cmd, stored

input_dir = r"C:\Users\ajbar\Downloads\cifs"
output_dir = r"C:\Users\ajbar\Downloads\cifs\out"

for filename in os.listdir(input_dir):
    if filename.endswith(".cif"):
        cif_file = os.path.join(input_dir, filename)
        pymol.cmd.load(cif_file, "structure")
        chains = cmd.get_chains("structure")
        if "B" in chains:
            pymol.cmd.remove("chain A")
        filename = os.path.splitext(os.path.basename(cif_file))[0]
        pdb_file = os.path.join(output_dir, filename + '.pdb')
        pymol.cmd.save(pdb_file, "structure")
        pymol.cmd.delete("structure")
