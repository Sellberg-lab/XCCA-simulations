#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import HDF5-files from calculations 
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
# 2019-06-06 v9 @ Caroline Dahlqvist cldah@kth.se
#			CrossCA_84-run_qvsq-bin-file.py
#				Load previously calculated and stored CC-maps (2 halves of data) from 'CrossCA_84-run_qvsq.py' |(HDF5-files)
#					on different ranges of pdb-#(time instances) and Re-Save as '.bin'-files
#			compatable with test_CsCl_84-X_v6- generated cxi-files
#			With argparser for input from Command Line
# Run directory must contain CC-maps *'tot-from-42'*.hdf5
# Prometheus path: /Users/Lucia/Documents/KTH/Simulations_CsCl/
# lynch path: /Users/lynch/Documents/users/caroline/Simulations_CsCl/
# Davinci: /home/cldah/source/XCCA-simulations/CsCl
#*****************************************************************************************************************
import argparse
import h5py 
import numpy as np
from numpy.fft import fftn, fftshift, fft ## no need to use numpy.fft.fftn/fftshift##
import matplotlib.cm as cm
import matplotlib.pyplot as pypl
from matplotlib import ticker
from matplotlib.patches import Circle
# pypl.rcParams["image.cmap"] = "jet" ## Change default cmap from 'viridis to 'jet ##
from loki.RingData import RadialProfile,DiffCorr, InterpSimple ## scripts located in RingData-folder ##
#from pylab import *	# load all Pylab & Numpy
# %pylab	# code as in Matlab
import os, time
import gc
#from struct import *  	## For generating 'bin'-files 
import struct  	## For generating 'bin'-files 
###################### Choose Calculations and Set Global variables: ########################################
this_dir = os.path.dirname(os.path.realpath(__file__)) ## Get path of directory
#this_dir = os.path.dirname(os.path.realpath('AutoCCA_84-run_v3.py')) ##for testing in ipython
if "/home/" in this_dir: #/home/cldah/cldah-scratch/ or /home or 
	os.environ['QT_QPA_PLATFORM']='offscreen'
Ampl_image = False			## The complex images in Amplitude Patterns instead of Intensity Patterns ##
#add_noise = False						## Add Generated Noise to Correlations ##
#random_noise_1quad =False			## Generate random Noise in one quardant ##

cart_diff, pol_all, pair_diff, all_DC_diff, pair_all = False,False,False, False, False ## only one should be true !!

q2_idx, q2_invA = None, None

# ------------------------------------------------------------

# ------------------------------------------------------------
def norm_data(data): 	# from Loki/lcls_scripts/get_polar_data.py
	"""
	Normliaze the numpy array data, 1-D numpy array
	"""
	data2 = data- data.min()
	return data2/ data2.max()

# ------------------------------------------------------------
def polarize(data): 	# from Loki/lcls_scripts/get_polar_data
	"""
	Adjust for polarization  !! MUST SET: args.pol_frac 'fraction of in-plane polarization' (float)!!
	"""
	#global mask_better
	if pol_frac is not None:
		assert (dtc_dist is not None)
		pixsize = 0.00010992
		Y,X = np.indices(mask_better.shape)
		R = np.sqrt( (Y-cntr[1])**2 + (X-cntr[0])**2 )
		PHI = np.arctan2( Y-cntr[1], X-cntr[0] )
		TWOTHETA = np.arctan( R*pixsize/ dtc_dist )
		inplane = pol_frac
		outplane = 1-pol_frac
		 # old Polarize - correct for PHI defined along the positive y-axis OR rotated geometry
		Polarize = inplane*(1-(np.sin(PHI)*np.sin(TWOTHETA) )**2 ) \
			+ outplane*(1-(np.cos(PHI)*np.sin(TWOTHETA) )**2 )
		# correct for PHI defined along the positive x-axis
		#Polarize = inplane*(1-(np.cos(PHI)*np.sin(TWOTHETA) )**2 ) \
		#    + outplane*(1-(np.sin(PHI)*np.sin(TWOTHETA) )**2 )
	else:	## if no fraction given, just divide by 1 ##
		Polarize = np.ones_like( mask_better)
	return data/Polarize
# ------------------------------------------------------------

#def create_out_dir(name,run,pdb,noisy,n_spread, random_noise_1quad=False):
def create_out_dir(out_dir,sim_res_dir=None,name=None,pdb=None,noisy=None, random_noise_1quad=False):
	"""
	Create/ Define the output directory for storing the resulting data.
		=========================
	out_dir 		Storage location of the calculated and plotted result
	sim_res_di 		Location (path) to data files to read in
	"""
	#if args.outpath== args.dir_name:
	#	outdir = args.dir_name +'/%s_%s_%s_(%s-sprd%s)/' %(name,run,pdb,noisy,n_spread)
	#else:	outdir = args.outpath
	
	#global this_dir


	if out_dir== sim_res_dir:
		#if (name is not None) and (pdb is not None) and (noisy is not None):
		if (name is not None) and (noisy is not None):
			#outdir = sim_res_dir+'/%s_%s_%s/' %(name,pdb,noisy)
			outdir = sim_res_dir+'/%s_%s/' %(name,noisy)
		else: outdir = sim_res_dir+ '/output/'
	else:	outdir = out_dir
	if not os.path.exists(outdir):
		os.makedirs(outdir)# os.makedirs(outdir, 0777)
	if random_noise_1quad:
		if (pdb is not None) and (noisy is not None):
			outdir = outdir + '/random_noise_1quad_(%s_%s)/' %(pdb,noisy)
		else:	outdir = outdir + '/random_noise_1quad/'
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	return outdir



def read_and_generate_bin_file(list_names, out_name):
	"""
	Read in CC data from multiple files in 'list_names' list of filenames. 
	Plot the average from all read-in files (saved in 'out_name'-location).
	"""
	#
	# Write a double precision binary file (i.e. correlation function)
	#
	#  for a 3D correlation function data is a 3D array with indices (q,q’,theta)
	# @ Andrew Martin
	def write_dbin( fname, data ):
	       
	    f = open( fname, "wb")
	    fmt='<'+'d'*data.size
	    bin = struct.pack(fmt, *data.flatten()[:] )
	    f.write( bin )
	    f.close()
	    print ".... File %s Closed ...."%f


	## ---- Load Calculations from Files and Calculate the Mean : ---- ##
	t_load =time.time()
	
	#corr_sum = []
	#cross_corr = []
	#mask_corr = []
	file_count= 0	## the number of ACC perfomed, one per loaded file ##
	#tot_shot_sum = 0	## the total number of shots (diffraction pattersn) from simulation ##
	#tot_diff_sum = 0	## the number of diffs or the number of patterns in input to ACC ##
	#q_map= None
	#nphi= None
	#diff_type_str= None
	for file in list_names:
		print "\nFile being read: ", file
		with h5py.File(file, 'r') as f:
			#if file==list_names[0]:
			#	q_map=np.asarray(f['q_mapping'])
			dset_cc = f['cross-correlation_sum']#[q2_idx,:,:] ## Data-set with Cross-Correlations (3D); Select only q2 ##
			#ccsummed =np.asarray(dset_cc[q2_idx,:,:])		## for importing only a specific q-values CC  ##
			ccsummed =np.asarray(dset_cc)	## Read in ALL q_2, OBS, must limit the number of files to read in from se line 1291-1295 ##
			#diff_count = dset_cc.attrs["tot_number_of_diffs"]
			#shot_count = dset_cc.attrs["tot_number_patterns"]
			tot_corr_sum= dset_cc.attrs["tot_number_of_corrs"]
			mask_crosscor = np.asarray(f['mask_cross-correlation'])
			#if file==list_names[0]:
			#	nphi=int(np.asarray(f['num_phi'])) 
			#	diff_type_str =dset_cc.attrs["diff_type"]
			#	dtc_dist_m = dset_cc.attrs["detector_distance_m"]
			#	wl_A = dset_cc.attrs["wavelength_Angstrom"]
			#	ps_m = dset_cc.attrs["pixel_size_m"]
			#	be_eV = dset_cc.attrs["beam_energy_eV"] 
		file_count+= 1
		cross_sum_m = np.divide(ccsummed, mask_crosscor, out=None, where=mask_crosscor!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
		cross_sum_m /= tot_corr_sum 	## Take the mean of the stored sums ## 
		fname = out_name+'_'+str(file_count)+'.bin'
		write_dbin( fname, cross_sum_m )

		#cross_corr.append(ccsummed)  ## (2,Qidx,Q,phi)
		#mask_corr.append(mask_crosscor)
		#tot_corr_sum+= 1 ##corr_count ## calculate the number of auto-correlations loaded ##
		#tot_shot_sum+= shot_count 
		#tot_diff_sum+= diff_count 
	#cross_corr = np.asarray(cross_corr)
	#mask_corr = np.asarray(mask_corr)

	#print "\n Dim of intershell-CC (@ selected q2): ", cross_corr_sum.shape ##(500,360)
	#print "\n Dim of intershell-CC : ", cross_corr.shape		# =(2, 500, 500, 360)
	t = time.time()-t_load
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Loading Time for %i patterns: "%(len(list_names)), t_m, "min, ", t_s, "s " #   0 min,  4.42866182327 s

	exit(0)  ## Finished ! ##



##################################################################################################################
##################################################################################################################
#---------------------------- Parameters and File-Names: --------------------------------------------------------------#
##################################################################################################################
##################################################################################################################

pol_frac = None  ## 'fraction of in-plane polarization' (float)!! ##
#select_name = " "

## ----- Load and Add Mask from Assembly (only 0.0 an 1.0 in data): --- ##
mask_better = np.load("%s/masks/better_mask-assembled.npy" %str(this_dir))
## if data is not NOT binary:  ione =np.where(mask_better < 1.0), mask_better[ione] = 0.0 # Make sure that all data < 1.0 is 0.0 	##  
mask_better = mask_better.astype(int) 	## Convert float to integer ##
print"\nDim of the assembled mask: ", mask_better.shape

# ---- Centre Coordiates Retrieved Directly from File ((X,Y): fs = fast-scan {x}, ss = slow-scan {y}): ----
#cntr = np.load("%s/centers/better_cent_lt14.npy" %str(this_dir))	# from exp-file run 84-119
#print "Centre from file: ", cntr ## [881.43426    863.07597243]
cntr_msk =np.asarray([(mask_better.shape[1]-1)/2.0, (mask_better.shape[0]-1)/2.0 ]) ## (X,Y)
print "  Centre from Mask: ", cntr_msk ##[870, 868]; [869.5, 871.5]; [871.5, 869.5]
cntr= cntr_msk 	## (X,Y) if use center point from the Msak ##
cntr_int=np.around(cntr).astype(int)  ## (rounds to nearest int) for the center coordinates in pixles as integers ##
#print "\n Centre as Integer: ", cntr_int,  "\n"


pttrn = "Int"
if Ampl_image:
	pttrn = "Ampl"
#if add_noise:
#	pttrn = "Int-add-noise-%iprc" %(nlevel*100)



##################################################################################################################
#------------------------------ Mean and Plot ['plots']: --------------------------------------------------------#
##################################################################################################################
def load_and_plot_from_hdf5(args):
	'''
	Load calculated data for the Cross-Correlation, file-by-file, from folder 'outpath' and specify 
	a directory and name for the file containg the total, mean and plots.
	'''
	new_folder = args.outpath + '/w_mean_ip/' ## location of files to import, partly calculated ##
	outdir = create_out_dir(out_dir = new_folder)
	fnames = [ os.path.join(outdir, f) for f in os.listdir(outdir)
		 if (f.find('tot-from-42') != -1) and f.endswith('.hdf5') ] ## find returns '-1' if not found ##
	#	 if (f.find('tot-from-42-files') != -1) and f.endswith('.hdf5') ]
	#str.find(str, beg=0, end=len(string))
	if not fnames:
		print"\n No filenames for directory %s"%(str(outdir))
	#fnames = sorted(fnames, 
	#	key=lambda x: int(x.split('M')[-1][6:].split('_(')[0] )) 	## For sorting simulated-file 0-90 ##
	#	#key=lambda x: int(x.split('84-')[-1][6:].split('_(')[0] )) 	## For sorting simulated-file 0-90 ##
	#fnames = sorted(fnames, key=lambda x: int(x.split('84-')[-1][4:].split('_ed')[0][0] )) 	## For sorting conc 4-6 M ##
	# Fnoise_Beam-NarrNarrInt_6M0-6M41_(none-sprd0)_tot-from-42-files_Cross-corr_Int.hdf5
	# Fnoise_Beam-NarrNarrInt_6M49-6M90_(none-sprd0)_tot-from-42-files_Cross-corr_Int.hdf5

	#######################  !! OBS : Memory limit on Cluster: max 60 GB, file size approx 849 Mb => Max Load 43 files  ######################## 
	#numb = 42 ## 43 works for 4 Molar, 42 works for 6Molar ##
	#fnames = fnames[:numb]	## 30 works, 43 exceeded memory limit (61207376 > 60817408),, 44 NOT! ##
	#fnames = fnames[-numb:]  ## 2nd half, 43 +4 works for 4Molar
	###########################################  !! OBS !!  ########################################## 

	# ----- Parameters unique to Simulation: ---- ##		UNSORTED FILES should be 2 maps
	cncntr = fnames[0].split('_(')[0].split('_')[-1].split('-')[0].split('M')[0] ## '...6M..' => '6'
	## Segment-CC__Pnoise_BeamNarrInt_84-119_4M8_(poisson-sprd0)_Int.hdf5
	#if len(fnames)>1:
	#	start_f = fnames[-1].split('_(')[0].split('_')[-1].split('-')[0].split('M')[-1]
	#	end_f = fnames[-1].split('_(')[0].split('_')[-1].split('-')[-1].split('M')[-1]
	#	start_i = fnames[0].split('_(')[0].split('_')[-1].split('-')[0].split('M')[-1] 	## '...6M0-6M41_(none-sprd0)...'=> '0' ##
	#	end_i = fnames[0].split('_(')[0].split('_')[-1].split('-')[-1].split('M')[-1] ## '...6M0 -6M41_(none-sprd0)...'=> '41' ##
	#	pdb= cncntr +'M' + start_i +'-'+  end_i + '_' + cncntr +'M'+ start_f +'-'+  end_f 
	#else :	pdb = cncntr
	pdb = cncntr
	#run=fnames[0].split('84-')[-1][0:3] 		## 3rd index excluded ##
	#noisy = fnames[0].split('_ed_(')[-1].split('-sprd')[0]
	#noisy = fnames[0].split('_(')[-1].split('-sprd')[0] ## if not '4M90_ed' but '4M90' in name
	#n_spread = fnames[0].split('-sprd')[-1].split(')_')[0] ## for files that ends with '_#N.cxi' ##
	#n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
	name = fnames[0].split('/')[-1].split('_(')[0].split('_')[0:2]
	name = name[0]#+name[1]
	pttrn = fnames[0].split(')_')[-1].split('.hdf5')[0]
	## /.../noisefree_Beam-NarrInt_84-105_6M90_ed_(none-sprd0)_#5.cxi
	# /.../Fnoise_Beam-NarrNarrInt_6M0-6M41_(none-sprd0)_tot-from-42-files_Cross-corr_Int.hdf5
	
	# ---- Generate a Storage File: ---- ##
	#prefix = '%s_%s_(%s-sprd%s)_CC-map_' %(name,pdb,noisy,n_spread) ## Final Result ##
	prefix = '%s_%s_meanCC_q300-800pxls_nphi360_CC-map' %(name,pdb) ## Final Result ##
	out_fname = os.path.join( outdir, prefix) 
	#out_hdf = h5py.File( out_fname + 'tot_Auto-corr_%s.hdf5' %(pttrn), 'a')    # a: append, w: write
	#out_hdf = h5py.File( out_fname + 'tot-from-%i-files_Cross-corr_%s.hdf5' %(numb,pttrn), 'w')    # a: append, w: write
	print "\n Data Analysis with LOKI."
	
	## ---- Read in CC from Files and Plot ----  ##
	read_and_generate_bin_file(fnames, out_fname)
##################################################################################################################
##################################################################################################################

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Auto-Correlations.")

parser.add_argument('-o', '--outpath', dest='outpath', default='this_dir', type=str, help="Path for output, Plots and Data-files")

subparsers = parser.add_subparsers()#title='calculate', help='commands for calculations help')


## ---- For Loading Nx# files, Summarise and Plotting the total result: -----##
subparsers_plt = subparsers.add_parser('plots', help='commands for plotting ')
subparsers_plt.add_argument('-p', '--plot', dest='plt_set', default='single', type=str, choices=['single', 'subplot', 'all'],
      help="Select which plots to execute and save: 'single' each plot separately,'subplot' only a sublpot, 'all' both single plots and a subplot.")

## NOT needed, only left for re-using ash-code for multiple prgms
parser_group = subparsers_plt.add_mutually_exclusive_group(required=True)
parser_group.add_argument('-R', '--R', dest='R', default=None, type=int ,
      help="Which radial pixel to look at.")
parser_group.add_argument('-Q', '--Q', dest='Q', default=None, type=float ,
      help="Which reciprocal space coordinate to look at.")

subparsers_plt.set_defaults(func=load_and_plot_from_hdf5)



args = parser.parse_args()
args.func(args) ## if .set_defaults(func=) SUBPARSERS ##
## ---------------- ARGPARSE  END ----------------- ##