#!/bin/bash

#SBATCH -o CsCl_simulate_test_noisefree.out
#SBATCH -e CsCl_simulate.err

HOST=`hostname`
echo "Node: $HOST" 
/davinci/Cellar/Python/miniconda3/envs/py2/bin/python test_CsCl_84-X_v6.py -dn None -f test_noisefree -n 5 -pdb 6M90_ed
