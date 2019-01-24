There are 92 structure-files for each concentration.
The files are named with the concentration first followed by a number. E.g. “2M92.pdb” is a box with 2 Molar concentration.


In the Header of each file is the dimensions of the simulated box:
E.g. CRYST1   32.476   28.958   29.770  90.00  90.00  90.00 P 1           1
 Means that the box has the dimension 32x28x29 Ångström***3.

The coordinates are in Angstrom, Clc is chlorine, Csc is ceasium, OW* is oxygen and HW* is hydrogen.

The difference with the files in the folder CsCl-PDB_ed is the added column number 78 containing the fist letter of the molecule. This is required for reading the files with Condor, which uses Spsim package.

To generate edited pdb-files run "fix_all_pdb_for_condor.py"