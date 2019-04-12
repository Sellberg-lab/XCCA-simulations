"""
 
  In order to avoid having to type a lot of arguments intot the command line, 
  this script runs the program 'simulate_CsCl_84-X_v6.py' with selected ipnut arguments.
  The program uses Condor to simulate a numer of Diffraction Patterns.

  Adapted from 'run_a_script' by Andrew Martin (andrew.martin@rmit.edu.au)
  	... 1 RUN => Loop to multiple? 
  	2019-02-14         @ Caroline Dahlqvist cldah@kth.se

"""

from os import system

script = 'simulate_CsCl_84-X.py' ## Currently beam set to 1nm focus and 0.1 J in Energy ##
params = {}

#####################################################
########## Paramters for Script: ####################
#####################################################

## loop through diff parame? ##
#pdb_files = ['4M0', '6M90']
## nestled loop for both noise types?? ##
#noise = ['poisson', None]
# if params["-dn"] == 'poisson': params["-f"] = 'test_Pnoise'
# if params["-dn"] == 'None: params["-f"] = 'test_noisefree'


## PDB-file simulated on: ##
params["-pdb"] = '4M0_ed'#'4M0_ed' ## simulated '4M0_ed'
#params["-pdb"] = '6M90_ed' #'6M90_ed'#'6M45_ed'

## Nmber of Patterns Simulated (if in CXI-file-name) :##
params["-n"] = '1' #Test-run with '1', '5', '100'

## Output PATH (not required, Default= ./simulation_result)##
#params["-o"] = None


## if no NOISE : ##
params["-f"] = 'test_Fnoise-1-None'
#params["-f"] = 'noisefree_Beam-NarrInt' ##with Narrower Beam and higher Intensity
#params["-f"] = 'Fnoise_Beam-NarrInt' ##with Narrower Beam and higher Intensity
params["-dn"] = "None"

## if Poission NOISE : ##
#params["-f"] =  'test_Pnoise' 
#params["-f"] = 'Pnoise_Beam-NarrInt' ##with Narrower Beam and higher Intensity
#params["-dn"] = 'poisson'

## if Gaussian NOISE : ##
#params["-f"] = 'Gnoise_Beam-NarrInt' ##with Narrower Beam and higher Intensity
#params["-dn"] = 'normal'
#params["-dns"] = 0.005000

#####################################################
#####################################################

## Generate the Command Line String ##
#command = "ipython " + script + " " ## ipython steals arguments
command = "python " + script + " "
for d, e in params.items():
	command += d+ " " + e + " "

## Run the Script ##
print command
system( command )


