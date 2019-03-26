"""
 
  In order to avoid having to type a lot of arguments intot the command line, 
  this script runs the program 'CCA_cxi_84-X_v3.py' with selected ipnut arguments.
  The program uses Condor to simulate a numer of Diffraction Patterns.

  Adapted from 'run_a_script' by Andrew Martin (andrew.martin@rmit.edu.au)
  	... 1 RUN => Loop to multiple? BASH script?
  	2019-02-14         @ Caroline Dahlqvist cldah@kth.se

"""

from os import system

script = 'CCA_cxi_84-X_v3.py' #'test_CsCl_84-119_v6.py'
params = {}

## loop through diff parame? ##
#run_name = ["84-119", "84-150"]
#pdb_files = ['4M0', '6M90']
## nestled loop for both noise types?? ##
#noise = ['poisson', None]
# if params["-dn"] == 'poisson': params["-f"] = 'test_Pnoise'
# if params["-dn"] == 'None: params["-f"] = 'test_noisefree'


## Paramters for Script: ##
#params["-r"] = "84-119" ## also 84-105 (with 6.34M/dm3)
#params["-r"] = "119" ## also 105 (with 6.34M/dm3)
params["-f"] = 'test_Pnoise'
params["-pdb"] = '4M0_ed' ## also 6M0, 6M90
params["-n"] = '5'
params["-dn"] = 'poisson'
#params["-dns"] = 0.005000

## Generate the Command Line String ##
#command = "ipython " + script + " " ##CRITICAL | Bad config encountered during initialization:##
command = "python " + script + " "
for d, e in params.items():
	command += d+ " " + e + " "

## Run the Script ##
print command
system( command )
