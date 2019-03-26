#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import CXI-files from simulations with Condor (v1.0) 
# and calculate and plot the Radial Profile with LOKI
# cxi-file/s located in the same folder, result saved  in subfolder "/simulation_results_N_X"
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
# 2019-03-06   @ Caroline Dahlqvist cldah@kth.se
#			RadProf_84-run.py
#			compatable with test_CsCl_84-X_v6- generated cxi-files
#			With argparser for input from Command Line
# Run directory must contain the Mask-folder
# Prometheus path: /Users/Lucia/Documents/KTH/Simulations_CsCl/
# lynch path: /Users/lynch/Documents/users/caroline/Simulations_CsCl/
# Davinci: /home/cldah/source/XCCA-simulations/CsCl
#*****************************************************************************************************************
import argparse
import h5py 
import numpy as np
from numpy.fft import fftn, fftshift # no need to use numpy.fft.fftn/fftshift
import matplotlib.pyplot as pypl
from matplotlib import ticker
# pypl.rcParams["image.cmap"] = "jet" ## Change default cmap from 'viridis to 'jet ##
#from pylab import *	# load all Pylab & Numpy
# %pylab	# code as in Matlab
import os, time, sys
import gc
this_dir = os.path.dirname(os.path.realpath(__file__)) ## Get path of directory
#this_dir = os.path.dirname(os.path.realpath('CCA_RadProf_84-run.py')) ##for testing in ipython
if "/home/" in this_dir: #/home/cldah/cldah-scratch/ or /home or 
	os.environ['QT_QPA_PLATFORM']='offscreen'

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Radial Profile.")

parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
      help="The Location of the directory with the cxi-files (the input).")

parser.add_argument('-o', '--outpath', dest='outpath', default='dir_name', type=str, help="Path for output, Plots and Data-files")

parser.add_argument('-p', '--prog', dest='RP_program', required='True', type=str,
      help="The Name of the Radial Profile program. 'Loki', 'Swiss' or 'Martin'.")

parser.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
      help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)

args = parser.parse_args()
## ---------------- ARGPARSE  END ----------------- ##

# ------------------------------------------------------------
def set_program(prg):
	if prg == "Loki" or prg =="loki": 
		global RP_Loki
		RP_Loki = True
	elif  prg == "Swiss" or prg == "swiss" : global RP_Swiss; RP_Swiss = True
	elif  prg == "Martin" or prg == "martin" : global RP_Martin; RP_Martin = True

# ------------------------------------------------------------
def load_Experimental_Parameters(data_file):
	"""
	Reads simulation data from 'data_file' and appends to list-objects

		In:
	data_file           the cxi-file/s from Condor to be read in.

	=====================================================================================
		Out:
	photon_e_eV         Photon Energy of source in electronVolt [eV], type = int, (if get_parameters = True)
	wl_A                Wavelength in Angstrom [A], type = float, (if get_parameters = True)
	ps                  Pixel Size [um], type = int, (if get_parameters = True)
	dtc_dist            Detector Distance [m], type = float, (if get_parameters = True)
	N                   Number of shots stored in data-file.
	"""
	with h5py.File(data_file, 'r') as f:
		photon_e_eV = np.asarray(f["source/incident_energy"] )			# [eV]
		photon_e_eV = int(photon_e_eV[0]) 								# typecast 'Dataset' to int

		photon_wavelength = np.asarray(f["source/incident_wavelength"]) #[m]
		photon_wavelength= float(photon_wavelength[0])					# typecast 'Dataset' to float
		wl_A = photon_wavelength*1E+10 									#[A]

		psa = np.asarray(f["detector/pixel_size_um"])  					#[um]
		ps = int(psa[0]) 												# typecast 'Dataset' to int

		dtc_dist_arr = np.asarray(f["detector/detector_dist_m"])		#[m]
		dtc_dist = float(dtc_dist_arr[0]) 								# typecast 'Dataset' to float

		N = f["entry_1/data_1/data"].shape[0]
	return photon_e_eV, wl_A, ps, dtc_dist, N
# -------------------------------------- 


# ------------------------------------------------------------
def load_cxi_data(data_file, idx):
	"""
	Reads simulation data from 'data_file' and returns the pattern selected by idx.

		In:
	data_file           the cxi-file/s from Condor to be read in.
	idx                  The intensity pattern to load, index from 0 to size-1
	=====================================================================================
		Out:
	intensity_pattern     at index 'idx' from the data loaded from 'data_file' appended.

	"""

	with h5py.File(data_file, 'r') as f:
		assert(f["entry_1/data_1/data"].shape[0] >= idx),("Index out of bounds!")
		intensity_pattern = np.asarray(f["entry_1/data_1/data"][idx])
		#amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"][0])
	return intensity_pattern
# -------------------------------------- 

def plot_rp_ip(rad_pro_mean, radii, pttrn):
	"""
	Plot the Mean Radial Profile of all Diffraction Patterns
	"""

	qmin_pix, qmax_pix =  args.q_range[0], args.q_range[1]
	#qrange_pix = np.arange( qmin_pix, qmax_pix)
	rad_cntr = int((rad_pro_mean.shape[0]-1)/2) 	## Center index of radial profile, OBx -1 since index start at 0 not 1 ##
	print "\n Dim of mean(Radial Profile): ", rad_pro_mean.shape, " , center pixel: ", rad_cntr ## (1800,)  ,  899 ##
	#start_pixel, end_pixel = (rad_cntr + qmin_pix), (rad_cntr + qmax_pix)
	#r_bins = np.linspace(0, (qrange_pix.shape[0]-1), num= 5, dtype=int) ## index for radial binsin pixels ##
	#print"r_bins: ", r_bins, "/n Dim r-bins ", r_bins.shape
	qrange_pix = np.arange(rad_cntr, rad_pro_mean.shape[0])  ## since endpoint not included: no need to subtract shape with 1 ##
	print "q range  pixels dim: ", qrange_pix.shape  	## =901 ##
	rp_bins = np.arange( rad_pro_mean.shape[0]) 	## the bin indeex for the radial profile ##

	## ---- radii from middle to end and converted to inv A but start at 0 in the middle [index, Q[inv Angstrom], Q [pixels]] : ----- ##
	qs_c_map = np.array( [ [indx, pix2invang(r_c), r_c] for indx, r_c in enumerate(radii[ np.arange(qrange_pix.shape[0])]) ] ) 
	print "qs_c_map  dim: ", qs_c_map.shape
	qs_map = np.array( [ [indx, pix2invang(r), r] for indx, r in enumerate(radii) ] ) ## index, Q[inv Angstrom], Q [pixels]##

	#print "\n Qs: ", qrange_pix
	print "\n q min: ", qrange_pix[0], "q max: ", qrange_pix[-1]
	start_pixel, end_pixel = qrange_pix[0], qrange_pix[-1]
	print "start pixel: ", start_pixel, "end pixel: ", end_pixel
	#print "\n RP interval: ", rad_pro_mean[ start_pixel : end_pixel ]

	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14			## Fontsize of Axis labels ##
	tick_fsize = 12			## Size of Tick-labels ##
	sb_size = 16 #18		## Fontsize of Sub-Title ##
	sp_size = 18#20			## Fontsize of Super-Title ##
	l_pad = 10				## Padding for Axis Labels ##
	print "Radial Profile max value for q-range %i to %i: "%(qrange_pix[0],qrange_pix[-1]), rad_pro_mean[start_pixel : end_pixel].max()
	fig1 = pypl.figure('IP-RP', figsize=(22,15))
	## subplot , constrained_layout=True 
	############################## [fig.a whole RP || mean intensity pattern] begin ##############################
	if not IP_mean_plot:			## plot from Edge ##
		if tot_RP: pypl.subplot(131)
		else:  pypl.subplot(121)
		axb = pypl.gca()  ## Bottom Axis ##
		axb.set_xlabel("r (pixels)", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##
		axb.plot(qs_map[:,2],rad_pro_mean, color='b', lw=2) ## for Bottom-Axis in r [pixels] ##
		axb.set_ylabel("Mean ADU", fontsize=axis_fsize)
		axb.tick_params(axis='x')#, labelcolor=color) 	## 'x' in plot
		legend = axb.legend() #pypl.legend()  ##(loc='upper center', shadow=True, fontsize='x-large')
		axt = axb.twiny()  ## Top Axis, Share the same y Axis ##	
		axt.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##
		axt.set_xticklabels(['%.3f'%(qs_map[idx,1]) for idx,X in enumerate(axb.get_xticks())]) ## for Top-Axis in q [inv Angstrom] ##
		axt.set_xticks(axb.get_xticks())
		axt.set_xbound(axb.get_xbound())

		rad_title =pypl.title('Radial Profile [from edge]',  fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
		rad_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
		pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	else:						## PLOT mean of Intensity Patterns: ##
		print "\n Plotting some Diffraction Patterns.\n"
		if tot_RP: pypl.subplot(131)
		else:  pypl.subplot(121)
		ax0 = pypl.gca()  ## Bottom Axis ##
		cbs,cbp =  0.98, 0.02 #0.04, 0.1: left plts cb ocerlap middle fig
		color_map ='jet' #'viridis'
		ax0.set_ylabel( 'y Pixels', fontsize=axis_fsize) 
		ax0.set_xlabel( 'x Pixels', fontsize=axis_fsize) 
		#pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'})
		## ax0.set( ylabel='y Pixels')
		## ax0.set( xlabel='x Pixels')
		# ---- Intensity Pattern : ---- #
		Ip_ma_shot = np.ma.masked_where(mask_better == 0, IP_mean) ## Check the # photons ##
		im_ip = pypl.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
		cb_ip = pypl.colorbar(im_ip, ax=ax0,fraction=0.046)
		cb_ip.set_label(r' Photons (mean) ') #(r'Intensity ')
		#cb_ip.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
		#cb_ip.update_ticks()
		#cmap.set_bad('grey',1.) ## (color='k', alpha=None) Set color to be used for masked values. ##
		#cmap.set_under('white',1.) ## Set color to be used for low out-of-range values. Requires norm.clip = False ##
	
		rad_title =pypl.title('Mean Intensity Pattern %i: ' %(shot_count), fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
		rad_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
		#pypl.gca().tick_params(labelsize=axis_fsize, length=9) 
		pypl.gca().tick_params(labelsize=tick_fsize, length=9)
	############################## [fig.a] End ##############################

	############################## [fig.b from center to right end (increasing)] begin ##############################
	if tot_RP: pypl.subplot(132)
	else:  pypl.subplot(122)
	axb = pypl.gca()  ## Bottom Axis ##
	axb.set_xlabel("r [number of bins]", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##
	#axb.set_xlabel("r (pixels)", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##
	#axb.plot(qrange_pix, rad_pro_mean[ start_pixel : end_pixel ], color='b', lw=2, label='center -> right') ## for Bottom-Axis in r [pixels] ##
	#axb.plot(qs_c_map[:,2], rad_pro_mean[ qrange_pix], color='b', lw=2, label='center -> right') ## for Bottom-Axis in r [pixels] ##
	axb.plot(qrange_pix, rad_pro_mean[ qrange_pix], color='b', lw=2, label='center -> right') ## for Bottom-Axis in r [pixels] ##
	axb.set_ylabel("Mean ADU", fontsize=axis_fsize)
	axb.tick_params(axis='x')#, labelcolor=color) 	## 'x' in plot
	legend = axb.legend() #pypl.legend()  ##(loc='upper center', shadow=True, fontsize='x-large')
	axt = axb.twiny()  ## Top Axis, Share the same y Axis ##	
	#axt.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##
	axt.set_xlabel("r [pixels]]", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##

	#axt.set_xticklabels(['%.3f'%(pix2invang(qs)) for idx,qs in enumerate(qrange_pix)]) #idx,X in enumerate(axb.get_xticks())])
	#axt.set_xticklabels(['%.3f'%(qs_c_map[idx,1]) for idx,qs in enumerate(qrange_pix)]) #idx,X in enumerate(axb.get_xticks())])
	#axt.set_xticklabels( map( str, axb.get_xticks()*radii.shape[0]/float(rad_pro_mean.shape[0] ) ) ) ## or qs_map.shape[0] =radii.shape[0]
	tic_conv =axb.get_xticks()*radii.shape[0]/float(rad_pro_mean.shape[0])
	tic_conv = tic_conv.astype(int)
	axt.set_xticklabels( map( str, tic_conv ) ) ## or qs_map.shape[0] =radii.shape[0]
	#print "pxl-value for q-range: ", qrange_pix*radii.shape[0]/rad_pro_mean.shape[0]
	axt.set_xticks(axb.get_xticks())
	axt.set_xbound(axb.get_xbound())

	#rad_title =pypl.title('Radial Profile [from center]',  fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
	rad_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
	#pypl.gca().tick_params(labelsize=axis_fsize, length=9) 
	pypl.gca().tick_params(labelsize=tick_fsize, length=9)
	############################## [fig.b] End ##############################
	if tot_RP:
		pypl.subplot(133)
		axb = pypl.gca()  ## Bottom Axis ##
		axb.set_xlabel("r [pixels]", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##
		axb.plot(rad_pro_mean, color='b', lw=2) ## for Bottom-Axis in r [pixels] ##
		axb.set_ylabel("Mean ADU", fontsize=axis_fsize)
		#axb.tick_params(axis='x')#, labelcolor=color) 	## 'x' in plot
		## ---- Set the Bottom Axis ticks as radial pixels (not bins) ---- ##
		#xtic= np.arange(0, rad_pro_mean.shape[0], rad_pro_mean.shape[0]/radii.shape[0]) ## TOO MANY ##
		print "rad_pro_mean.shape[0]",rad_pro_mean.shape[0], "  ******  radii.shape[0]" , radii.shape[0]
		#xtic= np.arange(0, rad_pro_mean.shape[0], 200*rad_pro_mean.shape[0]/radii.shape[0]) ## TOO MANY ##
		rp_bin_per_pxl = rad_pro_mean.shape[0]/float(radii.shape[0])
		print "bin_per_pixel", rp_bin_per_pxl
		#rp_bin_per_pxl =np.divide(rad_pro_mean.shape[0], radii.shape[0], out=None, where=radii.shape[0]!=0) ## out: scalar ##
		xtic = np.arange(0, rad_pro_mean.shape[0], 200*rp_bin_per_pxl) ## TOO MANY ##
		xtic = xtic.astype(int)
		axb.set_xticks(xtic) 
		# if req int step: axb.set_xticks(np.array(0, rad_pro_mean.shape[0], int(rad_pro_mean.shape[0]/radii.shape[0])) )
		qp_bins = np.linspace(0, radii.shape[0]-1, num= axb.get_xticks().shape[0], dtype=int)
		qp_label = [ '%i'%(qs_map[x,2]) for x in qp_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		qA_label = [ '%.3f'%(qs_map[x,1]) for x in qp_bins]
		axb.set_xticklabels(qp_label)

		print "X-ticks bottom: ", axb.get_xticks()
		print "X-ticks bottom type: ", type(axb.get_xticks())

		#print "X-ticks as list : ",axb.get_xticks.tolist() ## 'function' object has no attribute 'tolist'  
		#legend = axb.legend() #pypl.legend()  ##(loc='upper center', shadow=True, fontsize='x-large')
		axt = axb.twiny()  ## Top Axis, Share the same y Axis ##	
		axt.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##
		#axt.set_xticklabels(['%.3f'%(qs_map[idx,1]) for idx,X in enumerate(axb.get_xticks())]) ## indx instead of values ## ERR for Top-Axis in q [inv Angstrom] ##
		## a[:,v.T>0.5] -> v.t==X for column 2 ?? qs_map[??,1]
		#axt.set_xticklabels(['%.3f'%(qs_map[X,1]) for X in range(qs_map[:,1].shape[0])])#axb.get_xticks()]) ## for Top-Axis in q [inv Angstrom] ##
		
		#a=axb.get_xticks().tolist()
		#axb.set_xticklabels(a)
		#print [item.get_text() for item in axb.get_xticklabels()]
		
		#axt.set_xticklabels(['%.3f'%(qs_map[X,1]) for X in axb.get_xticks()]) 
		## IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
		axt.set_xticks(axb.get_xticks())
		axt.set_xticklabels(qA_label)
		axt.set_xbound(axb.get_xbound())
		print "X-ticks top: ", axt.get_xticks()

		#rad_title =pypl.title('Radial Profile [from edge]',  fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
		rad_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
		#pypl.gca().tick_params(labelsize=axis_fsize, length=9) 
		pypl.gca().tick_params(labelsize=tick_fsize, length=9)
	############################## [fig.c] End ##############################

	pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)  # hspace=0.2,
	## WORSE left=0.01, right=0.99), BETTER left=0.1, right=0.9, top=0.86...bottom 0.10 r label gone
	#pypl.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9) #top=0.86
	pypl.suptitle("%s_%s_(%s-noise_%s)_[qx-%i_qi-%i]_w_Mask_%s" %(name,pdb,noisy,n_spread,qrange_pix[-1],qrange_pix[0],pttrn), fontsize=sp_size)  # y=1.08, 1.0=too low(overlap radpro title)
	fig_name = "SUBPLOT_mean_Radial-Profile_w_Mask_%s.%s" %(pttrn,frmt)
	#pypl.show()
	pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	del fig_name, rad_pro_mean 	# clear up memory1
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##

# -------------------------------------- 

##################################################################################################################
##################################################################################################################
#---------------------------- Choose Calculations: --------------------------------------------------------------#
##################################################################################################################
##################################################################################################################
#add_noise = False						## Add Generated Noise to Correlations ##
random_noise_1quad =False			## Generate random Noise in one quardant ##

RP_Loki, RP_Swiss, RP_Martin = False, False, False
set_program(args.RP_program)

#plotting = True 				## Plot the Mean Radial Profile
IP_mean_plot = True				## Plot the Mean of the Intensity Paterns, if False: plot the mean Radial Profile from edge
tot_RP = True 					## Plot a 3rd subpolt of the complete Radial Profile (from edge) ##
##################################################################################################################
##################################################################################################################
#---------------------------- Parameters and Data: --------------------------------------------------------------#
##################################################################################################################
##################################################################################################################

## ---- Load File Names in Directory and Sort Numerically after Concentraion: ---- ##
sim_res_dir = args.dir_name
fnames = [ os.path.join(sim_res_dir, f) for f in os.listdir(sim_res_dir)
		if f.endswith('cxi') ]
if not fnames:
	print"\n No filenames for directory %s"%(str(sim_res_dir))
fnames = sorted(fnames, 
	key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0] )) 	## For sorting simulated-file 0-90 ##
#fnames = sorted(fnames, key=lambda x: int(x.split('84-')[-1][4:].split('_ed')[0][0] )) 	## For sorting conc 4-6 M ##

## ----- Fetch Simulation Data: ---- ##
#I_list = []  ## or l = np.array([]) used with l=np.append(a) || np.vstack(exposure_diffs):From List-object to array in vertical stack ##
#t_load = time.time()
#for k in fnames :
#	photon_e_eV, wl_A, ps, dtc_dist = load_Experimental_Parameters(k)
#	i = list(fnames).index(k) 	## indices ##
#	if i==0:		## in the 1st loop: save the simulation parameters (only once sine=ce static) ##
#		photon_e_eV, wl_A, ps, dtc_dist, I_list= load_cxi_data(data_file= k, get_parameters= True, ip= I_list)
#	else: 	 I_list = load_cxi_data(data_file= k, ip= I_list)
#intensity_pattern = np.asarray(I_list)
#t = time.time()-t_load
#t_m =int(t)/60
#t_s=t-t_m*60
#print "\n Loading Time for %i patterns: "%(intensity_pattern.shape[0]), t_m, "min, ", t_s, "s \n" # 5 min, 19.2752921581 s
photon_e_eV, wl_A, ps, dtc_dist, N = load_Experimental_Parameters(fnames[0])
##############################################################################################################################
#N=2 		# testing mode, only run for  2 patterns
##############################################################################################################################

## ----- Load and Add Mask from Assembly (only 0.0 an 1.0 in data): --- ##
mask_better = np.load("%s/masks/better_mask-assembled.npy" %str(this_dir))
## if data is not NOT binary:  ione =np.where(mask_better < 1.0), mask_better[ione] = 0.0 # Make sure that all data < 1.0 is 0.0 	##  
mask_better = mask_better.astype(int) 	## Convert float to integer ##
print"Dim of the assembled mask: ", mask_better.shape
#Ip_w_mb=np.multiply(intensity_pattern,mask_better)	## Intensity Pattern with Mask ##


# ---- Centre Coordiates Retrieved Directly from File ((X,Y): fs = fast-scan {x}, ss = slow-scan {y}): ----
#cntr = np.load("%s/centers/better_cent_lt14.npy" %str(this_dir))	# from exp-file run 84-119
#print "Centre from file: ", cntr ## [881.43426    863.07597243]
cntr_msk =np.asarray([(mask_better.shape[1]-1)/2.0, (mask_better.shape[0]-1)/2.0 ]) ## (X,Y)
print "\n Centre from Mask: ", cntr_msk ##[870, 868]; [869.5, 871.5]; [871.5, 869.5]
cntr= cntr_msk 	## (X,Y) if use center point from the Msak ##
cntr_int=np.around(cntr).astype(int)  ## (rounds to nearest int) for the center coordinates in pixles as integers ##
#print "\n Centre as Integer: ", cntr_int,  "\n"


# ----- Parameters unique to Simulation: ---- ##
cncntr_start = fnames[0].split('_ed')[0].split('84-')[-1][4:] 	## '...84-105_6M90_ed...'=> '6M90' ##
if len(fnames)>1:
	cncntr_end = fnames[-1].split('_ed')[0].split('84-')[-1][4:]
	pdb=cncntr_start +'-'+cncntr_end
else :	pdb = cncntr_start
run=fnames[0].split('84-')[-1][0:3] 		## 3rd index excluded ##
noisy = fnames[0].split('_ed_(')[-1].split('-sprd')[0]
#n_spread = fnames[0].split('-sprd')[-1].split(')_')[0]
n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
name = fnames[0].split('_84-')[0].split('/')[-1]
#N = intensity_pattern.shape[0] 	## the Number of Pattersn loaded ## 


## ---- Make an Output Dir: ----##
frmt = "eps" 	## Format used to save plots ##
if args.outpath == sim_res_dir:
	outdir = sim_res_dir+'/%s_%s_%s_(%s-sprd%s)_#%i/' %(name,run,pdb,noisy,n_spread,N)
else:	outdir = args.outpath
if not os.path.exists(outdir):
	os.makedirs(outdir)# os.makedirs(outdir, 0777)
outdir_raw =outdir ## For storting raw imgs if plot_diffraction=TRUE ##
if random_noise_1quad:
	outdir = outdir + '/random_noise_1quad_(%s_%s_%s)/' %(pdb,noisy,n_spread)
	if not os.path.exists(outdir):
		os.makedirs(outdir)


# ---- Generate a Storage File's Prefix: ---- 
prefix = '%s_%s_' %(name,pdb)
out_fname = os.path.join( outdir, prefix)       # with directory 

# ---- Some Useful Functions [ps in um, det_dist in m]: ----/t/ from Sacla-tutorial
pix2invang = lambda qpix : np.sin(np.arctan(qpix*(ps*1E-6)/dtc_dist )*0.5)*4*np.pi/wl_A
invang2pix = lambda qia : np.tan(2*np.arcsin(qia*wl_A/4/pi))*dtc_dist/(ps*1E-6)


##################################################################################################################
##################################################################################################################
#------------------------------- Loki Radial Profile: -----------------------------------------------------------#
##################################################################################################################
##################################################################################################################
if RP_Loki:
	from loki.RingData import RadialProfile,DiffCorr, InterpSimple ## scripts located in RingData-folder ##
	print "\n Data Analysis with LOKI.\n"

	# ---- Generate a Storage File for Data/Parameters: ----                                                      
	data_hdf = h5py.File( out_fname + 'Radial-Profile_w_Loki.hdf5', 'a')    # a: append   

	print "\n Generating Radial Profile Instance..."
	t_RF1 = time.time()

	# ---- RadialProfile - instance: ----
	r_p = RadialProfile(cntr, mask_better.shape, mask = mask_better )
	#r_p = RadialProfile(cntr, mask_better.shape, mask = mask_better, minlength=21230 ) #dim of img[0]: (1738, 1742)
	## defauly: minlength=1800  yelds a Radial Profile of min length 1800, Radial bin normalized ##
	## Diagonal in pixels: 2460  => Radii : 1230

	pxls_bin = r_p.num_pixels_per_radial_bin
	R_bins= r_p.Rbins
	print "\nnum_pixels_per_radial_bin: ", pxls_bin  ## [0 0 0 ... 0 0 0]
	print "\nnum_pixels_per_radial_bin Max = %f , Min = %f" %(pxls_bin.max(), pxls_bin.min())  ## [0 0 0 ... 0 0 0]
	print "\n R_bins: ", R_bins 		##  [   0    1    2 ... 1227 1228 1229]



	# --- Radial Profile Calculate: ----
	print "\n Starting to Calculate the Radial Profile..."
	#rad_pros = np.array( [ r_p.calculate(img[i]) for i in range( N ) ] ) # Rad profile of all
	RP_list = []
	shot_count = 0
	IP_sum = np.zeros_like(mask_better, dtype=float)
	#for k,i in zip(fnames, range(N)):  ## pairs list objects !! ##
	for k in fnames:
		for i in range(N):
			img = load_cxi_data(data_file= k, idx=i)*mask_better
			IP_sum +=  img ## sum all the diffraction patterns ##
			#sys.stdout.write("\n Loaded image's dim: %i,%i \n" %(img.shape[0],img.shape[1])), sys.stdout.flush()
			#r_p = RadialProfile(cntr, img.shape, mask = mask_better) #dim of img[0]: (1738, 1742)
			radial_profile_i = r_p.calculate(img)
			sys.stdout.write( "\n Calculated #%i RP's dim: " %(i)),sys.stdout.write(str(radial_profile_i.shape)),sys.stdout.flush()
			
			#RP_list.extend(radial_profile_i)  ## adds to end of list : lxnxm => l+1xnxm
			RP_list.append(radial_profile_i)  ## since item is 1D (1800,) :lxnxm => 2xlxnxm
			sys.stdout.write( "\n List of RP's length: %i" %( len(RP_list) )),sys.stdout.flush()
			del radial_profile_i, img 
			gc.collect() ## Free up Memory: ##
			shot_count += 1
		#shot_count += N
	rad_pros = np.asarray(RP_list)
	rad_pro_mean = rad_pros.mean(0)
	print "\n Radial Profiles calculated: ", rad_pros.shape, " ... and dimension of mean is: ", rad_pro_mean.shape
	IP_mean = IP_sum/float(shot_count)	## the mean of all the Diffraction Patterns ##
	del r_p, rad_pros, IP_sum, RP_list
	gc.collect() ## Free up Memory: ##
	
	# ---- Calculate Q's [1/A] for using in plotting: ---- ## self.num_pixels_per_radial_bin; self.Rbins
	#radii = np.arange( rad_pro_mean.shape[0])	# from GDrive: CXI-2018_MArting/scripts/plot_wax2
	radii = R_bins
	print "Radii dim: ", radii.shape
	#qs = pix2invang(radii) # ps[um]= (ps*1E-6)[m]; dtc_dist[m]: wl_A[A]	
	qs_map = np.array( [ [indx, pix2invang(r), r] for indx, r in enumerate(radii) ] ) ## index, Q[inv Angstrom], Q [pixels]##
	#radii_c = np.arange( (rad_pro_mean.shape[0]-1)/2) 	## For plotting from center of detector and not the edge, index start at 0 ##
	radii_c =  radii[: radii.shape[0]/2]  ## convert to pixels, end-point not included ##
	#qs_c = pix2invang(radii_c) 		## For plotting from center of detector and not the edge ##
	qs_c_map = np.array( [ [indx, pix2invang(r_c), r_c] for indx, r_c in enumerate(radii_c) ] ) ## index, Q[inv Angstrom], Q [pixels]##
	#del radii, radii_c 	# clear up memory
	#gc.collect() ## Free up Memory: ##
	# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
	if 'mean_radial_profiles' not in data_hdf.keys(): data_hdf.create_dataset( 'mean_radial_profiles', data = rad_pro_mean)
	else: 
		del data_hdf['mean_radial_profiles']
		dset = data_hdf.create_dataset('mean_radial_profiles', data=rad_pro_mean)
	if 'qs_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_for_profile', data = qs_map)
	else: 
		del data_hdf['qs_for_profile']
		dset = data_hdf.create_dataset('qs_for_profile', data=qs_map)
	if 'qs_c_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_c_for_profile', data = qs_c_map)
	else: 
		del data_hdf['qs_c_for_profile']
		dset = data_hdf.create_dataset('qs_c_for_profile', data=qs_c_map)
	if 'mean intensity pattern' not in data_hdf.keys(): 
		dset = data_hdf.create_dataset( 'mean intensity pattern', data = IP_mean)
		dset.attrs["number_patterns"] = shot_count
	else: 
		del data_hdf['mean intensity pattern']
		dset = data_hdf.create_dataset('mean intensity pattern', data=IP_mean)
		dset.attrs["number_patterns"] = shot_count
	data_hdf.close()
	print "\n File Closed! \n"
	del qs_map, qs_c_map 	# clear up memory
	gc.collect() ## Free up Memory: ##
	t = time.time()-t_RF1
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Radial Profile & Calculation Time in LOKI: ", t_m, "min, ", t_s, " s \n" 

	pttrn= "Loki"
	plot_rp_ip(rad_pro_mean, radii, pttrn)
	#plot_rp_ip(rad_pro_mean, qs_map,qs_c_map, pttrn)

##################################################################################################################
##################################################################################################################
#------------------------------ Swiss-FEEL Radial Profile: ----------------------------------------------------------#
##################################################################################################################
##################################################################################################################
if RP_Swiss:
	from SwissFEL_src import integrators as igs ## no module SwissFEL_src 
	print "\n Data Analysis with Swiss-FEL.\n"
		# ---- Generate a Storage File for Data/Parameters: ----                                                      
	data_hdf = h5py.File( out_fname + 'Radial-Profile_w_SwissFEL.hdf5', 'a')    # a: append   
	
	print "\n Generating Radial Profile Instance..."
	t_RF1 = time.time()

	## ----- Setup the  integrator class : ----- ##
	img0 = load_cxi_data(data_file= fnames[0], idx=0)*mask_better
	rad_dist = igs.radial_distances(img0, center=cntr_int)
	del img0
	#ra = igs.RadialAverager(rad_dist, mask_better, n_bins=1000)
	ra = igs.RadialAverager(rad_dist, mask_better, n_bins=1230)
	## Diagonal in pixels: 2460  => Radii : 1230
	q_pxls = ra.bin_centers   ## [pixles] ## The q center of each bin. 
	print "\n bin_centers: ", q_pxls
	b_factor =ra._bin_factor
	print "\n bin_factor: ", b_factor
	#del rad_dist, bin_values 
	#del rad_dist

	# --- Radial Profile Calculate: ----
	print "\n Starting to Calculate the Radial Profile..."
	RP_list = []
	shot_count = 0
	bin_c = None
	IP_sum = np.zeros_like(mask_better, dtype=float)
	#for k,i in zip(fnames, range(N)):
	for k in fnames:
		for i in range(N):
			img= load_cxi_data(data_file= k, idx=i)*mask_better
			IP_sum +=  img
			## ----- Use RA-instance to estimate average :----- ##
			bin_values = ra(img)  ## The average intensity in the bin (int) ##
			sys.stdout.write( "\n Calculated #%i RP's dim: " %(i)),sys.stdout.write(str(bin_values.shape)),sys.stdout.flush()
			RP_list.append(bin_values)
			del bin_values

			## ----- Use function to estimate Radial Profile (CRASH: different shape / loop):----- ##
			#rad_dist = igs.radial_distances(img, center=cntr_int)
			#bin_c, r_p = igs.angular_average(image=img, mask=mask_better,rad=rad_dist, center=cntr_int, nx=1
            #             pixel_size=ps*1E-6, photon_energy=photon_e_eV, detector_distance=dtc_dist) 	## if detector properties: conv to inv A & if nx is None:  nx = 1000 pixels/bin
			#sys.stdout.write( "\n Calculated #%i RP's dim: " %(i)),sys.stdout.write(str(r_p.shape)),sys.stdout.flush()
			#del bin_c
			#bin_c, r_p = igs.angular_average(image=img, mask=mask_better,rad=rad_dist, center=cntr_int, nx=1)
			#sys.stdout.write( "\n Calculated #%i RP's dim: " %(i)),sys.stdout.write(str(r_p.shape)),sys.stdout.flush()
			#RP_list.append(r_p) ## list of 1D objecs ##
			#del r_p
			
			sys.stdout.write( "\n List of RP's length: %i" %( len(RP_list) )),sys.stdout.flush()
			#del rad_dist, bin_c, r_p 
			del img 
			gc.collect() ## Free up Memory: ##
			shot_count += 1
		#shot_count += N
	rad_pros = np.asarray(RP_list)
	rad_pro_mean = rad_pros.mean(0)
	print "\n Radial Profiles calculated: ", rad_pros.shape, " ... and dimension of mean is: ", rad_pro_mean.shape
	IP_mean = IP_sum/float(shot_count)	## the mean of all the Diffraction Patterns ##
	del rad_pros, IP_sum, RP_list
	gc.collect() ## Free up Memory: ##
	
	# ---- Calculate Q's [1/A] for using in plotting: ----
	if bin_c is None: radii = q_pxls 	## The q center of each bin from  ra.bin_centers  ##
	else: radii = bin_c 	## The center of each bin in R. shape is (nx, ) from Radial_Average##
	#np.arange( rad_pro_mean.shape[0])	# from GDrive: CXI-2018_MArting/scripts/plot_wax2
	qs_map = np.array( [ [indx, pix2invang(r), r] for indx, r in enumerate(radii) ] ) ## index, Q[inv Angstrom], Q [pixels]##
	#qs = pix2invang(radii) # ps[um]= (ps*1E-6)[m]; dtc_dist[m]: wl_A[A]	
	radii_c =  radii[: (radii.shape[0])/2] 	## For plotting from center of detector and not the edge, index start at 0 ends at 'stop'-1 ##
	#qs_c = pix2invang(radii_c) 		## For plotting from center of detector and not the edge ##
	qs_c_map = np.array( [ [indx, pix2invang(r_c), r_c] for indx, r_c in enumerate(radii_c) ] ) ## index, Q[inv Angstrom], Q [pixels]##
	#del radii, radii_c 	# clear up memory
	#gc.collect() ## Free up Memory: ##
	# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
	if 'mean_radial_profiles' not in data_hdf.keys(): data_hdf.create_dataset( 'mean_radial_profiles', data = rad_pro_mean)
	else: 
		del data_hdf['mean_radial_profiles']
		dset = data_hdf.create_dataset('mean_radial_profiles', data=rad_pro_mean)
	if 'qs_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_for_profile', data = qs_map )
	else: 
		del data_hdf['qs_for_profile']
		dset = data_hdf.create_dataset('qs_for_profile', data=qs_map )
	if 'qs_c_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_c_for_profile', data = qs_c_map)
	else: 
		del data_hdf['qs_c_for_profile']
		dset = data_hdf.create_dataset('qs_c_for_profile', data=qs_c_map)
	if 'mean intensity pattern' not in data_hdf.keys(): 
		dset = data_hdf.create_dataset( 'mean intensity pattern', data = IP_mean)
		dset.attrs["number_patterns"] = shot_count
	else: 
		del data_hdf['mean intensity pattern']
		dset = data_hdf.create_dataset('mean intensity pattern', data=IP_mean)
		dset.attrs["number_patterns"] = shot_count
	data_hdf.close()
	print "\n File Closed! \n"
	del qs_map, qs_c_map 	# clear up memory
	gc.collect() ## Free up Memory: ##
	t = time.time()-t_RF1
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Radial Profile & Calculation Time wiht Swiss-FEL-script: ", t_m, "min, ", t_s, " s \n" 

	pttrn= "Swis-FEL"
	plot_rp_ip(rad_pro_mean, radii, pttrn)
	#plot_rp_ip(rad_pro_mean, qs_map,qs_c_map, pttrn)

##################################################################################################################
##################################################################################################################
#------------------------------ Martin Radial Profile: ----------------------------------------------------------#
##################################################################################################################
##################################################################################################################
if RP_Martin:
	print "\n Data Analysis with Martin.\n"

	def div_nonzero(a, b):
	    m = b != 0
	    c = np.zeros_like(a)
	    c[m] = a[m] / b[m].astype(a.dtype)
	    return c

	def make_radial_profile(image, x, y, mask = None, rs = None):
	    """
	    """
	    if rs is None :
	        rs = np.round(np.sqrt(x**2 + y**2), 0).astype(np.uint16).ravel()
	        print "\n rs = ",  rs,
	    if mask is None :
	        mask = np.ones_like(image, dtype=np.bool) 
	    m = mask.ravel().astype(np.bool)
	    print "\n rs.shape = ",  rs.shape, " & m.shape = ",m.shape, "\n & mask.ravel.astype(np.bool)  = ", mask.ravel().astype(np.bool)
	    ## rs.shape =  (3027596,)  & m.shape = (3027596,)
	    
	    r_count = np.bincount(rs[m], minlength=rs.max()+1)
	    #IndexError: boolean index did not match indexed array along dimension 0; dimension is 1 but corresponding boolean dimension is 3027596
	    r_int   = np.bincount(rs[m], image.ravel()[m].astype(np.float),  minlength=rs.max()+1)
	    #print('r_count.shape, r_int.shape', r_count.shape, r_int.shape, m.shape, rs.shape)
	    return div_nonzero(r_int, r_count), rs
	 
	# ------------------------------------------------------------
	# ---- Generate a Storage File for Data/Parameters: ----                                                      
	data_hdf = h5py.File( out_fname + 'Radial-Profile_w_Martin.hdf5', 'a')    # a: append   
	
	print "\n Generating Radial Profiles..."
	t_RF1 = time.time()
	RP_list = []
	shot_count = 0
	IP_sum = np.zeros_like(mask_better, dtype=float)
	rs = None
	## ---- Added code from 'integrators.py'-script because no psana is used ---- ##
	pxls_x, pxls_y = np.arange(0,mask_better.shape[1]),  np.arange(0,mask_better.shape[0]) ## size of det, or vector with all pixles ?? ##
	xx, yy = np.meshgrid(pxls_x, pxls_y)
	xx -= cntr_int[1]
	yy -= cntr_int[0] #rad = np.sqrt(xx*xx + yy*yy)

	for k in fnames:
		for i in range(N): #N):
			img = load_cxi_data(data_file= k, idx=i)*mask_better
			IP_sum +=  img
			rad_av, rs = make_radial_profile(img, x=xx, y=yy, mask=mask_better, rs=rs)
			#rad_av, rs = make_radial_profile(img, x=mask_better.shape[1], y=mask_better.shape[0], mask=None, rs=rs)
			sys.stdout.write( "\n Calculated #%i RP's dim: " %(i)),sys.stdout.write(str(rad_av.shape)),sys.stdout.flush()

			#RP_list.extend(rad_av)
			RP_list.append(rad_av) ## list of 1D objecs ##
			sys.stdout.write( "\n List of RP's length: %i" %( len(RP_list) )),sys.stdout.flush()
			del img, rad_av
			gc.collect() ## Free up Memory: ##
			shot_count += 1
		#shot_count += N
	rad_pros = np.asarray(RP_list)
	rad_pro_mean = rad_pros.mean(0)
	print "\n Radial Profiles calculated: ", rad_pros.shape, " ... and dimension of mean is: ", rad_pro_mean.shape
	IP_mean = IP_sum/float(shot_count)	## the mean of all the Diffraction Patterns ##
	del rad_pros, IP_sum, RP_list
	gc.collect() ## Free up Memory: ##

	# ---- Calculate Q's [1/A] for using in plotting: ----
	radii = np.arange( rad_pro_mean.shape[0])	# from GDrive: CXI-2018_MArting/scripts/plot_wax2
	#radii = rs
	print "\n dim 0 of radii sent to plot: ", radii.shape[0]
	qs = pix2invang(rs) #radii) # ps[um]= (ps*1E-6)[m]; dtc_dist[m]: wl_A[A]	
	#radii_c = np.arange( (rad_pro_mean.shape[0]-1)/2) 	## For plotting from center of detector and not the edge, index start at 0 ##
	radii_c = radii[: (radii.shape[0])/2] 	## For plotting from center of detector and not the edge, index start at 0 ends at 'stop'-1 ##
	qs_c = pix2invang(radii_c) 		## For plotting from center of detector and not the edge ##
	#del radii, radii_c 	# clear up memory
	#gc.collect() ## Free up Memory: ##
	# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
	if 'mean_radial_profiles' not in data_hdf.keys(): data_hdf.create_dataset( 'mean_radial_profiles', data = rad_pro_mean)
	else: 
		del data_hdf['mean_radial_profiles']
		dset = data_hdf.create_dataset('mean_radial_profiles', data=rad_pro_mean)
	if 'qs_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_for_profile', data = qs)
	else: 
		del data_hdf['qs_for_profile']
		dset = data_hdf.create_dataset('qs_for_profile', data=qs)
	if 'qs_c_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_c_for_profile', data = qs_c)
	else: 
		del data_hdf['qs_c_for_profile']
		dset = data_hdf.create_dataset('qs_c_for_profile', data=qs_c)
	if 'mean intensity pattern' not in data_hdf.keys(): 
		dset = data_hdf.create_dataset( 'mean intensity pattern', data = IP_mean)
		dset.attrs["number_patterns"] = shot_count
	else: 
		del data_hdf['mean intensity pattern']
		dset = data_hdf.create_dataset('mean intensity pattern', data=IP_mean)
		dset.attrs["number_patterns"] = shot_count
	data_hdf.close()
	print "\n File Closed! \n"
	t = time.time()-t_RF1
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Radial Profile & Calculation Time with Martin-script: ", t_m, "min, ", t_s, " s \n" 

	pttrn= "Martin"
	plot_rp_ip(rad_pro_mean, radii, pttrn)
	#plot_rp_ip(rad_pro_mean, qs_map,qs_c_map, pttrn)
