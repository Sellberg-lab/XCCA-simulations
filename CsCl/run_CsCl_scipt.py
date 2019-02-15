"""
 
  In order to avoid having to type a lot of arguments intot the command line, 
  this script runs the program 'test_CsCl_84-X_v6.py' with selected ipnut arguments.
  The program uses Condor to simulate a numer of Diffraction Patterns.

  Adapted from 'run_a_script' by Andrew Martin (andrew.martin@rmit.edu.au)
  	... 1 RUN => Loop to multiple? BASH script?
  	2019-02-14         @ Caroline Dahlqvist cldah@kth.se
"""

from os import system

script = 'test_CsCl_84-X_v6.py' #'test_CsCl_84-119_v6.py'
params = {}

## Paramters for Script: ##
params["-r"] = "84-119" ## also 84-105 (with 6.34M/dm3)
#params["-r"] = "119" ## also 105 (with 6.34M/dm3)
params["-f"] = 'test_Pnoise'
params["-pdb"] = '4M0' ## also 6M0, 6M90
params["-n"] = 5
params["-dn"] = 'poisson'
#params["-dns"] = 0.005000

## Generate the Command Line String ##
command = "ipython " + script + " "
	for d, e in params.items():
		command += d+ " " + e + " "

## Run the Script ##
print command
system( command )