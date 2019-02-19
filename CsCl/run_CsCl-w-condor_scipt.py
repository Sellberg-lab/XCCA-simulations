"""
 
  In order to avoid having to type a lot of arguments intot the command line, 
  this script runs the program 'test_CsCl_84-X_v6.py' with selected ipnut arguments.
  The program uses Condor to simulate a numer of Diffraction Patterns.

  Adapted from 'run_a_script' by Andrew Martin (andrew.martin@rmit.edu.au)
  	... 1 RUN => Loop to multiple? 
  	2019-02-14         @ Caroline Dahlqvist cldah@kth.se

"""

from os import system

script = 'test_CsCl_84-X_v6.py' #'test_CsCl_84-119_v6.py'
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
params["-f"] = 'Pnoise_BeamNarrInt' # 'test_Pnoise' with Narrower Beam and higher Intensity
params["-pdb"] = '6M45_ed'#'4M0_ed' ## simulated '4M0_ed', 6M45_ed,
params["-n"] = '5'
#params["-o"] = None 	## Output Path, not required ##
params["-dn"] = 'poisson'
#params["-dns"] = 0.005000

## Generate the Command Line String ##
#command = "ipython " + script + " " ## ipython steals arguments
command = "python " + script + " "
for d, e in params.items():
	command += d+ " " + e + " "

## Run the Script ##
print command
system( command )


