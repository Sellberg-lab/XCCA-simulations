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
# Prometheus path: /Users/Lucia/Documents/KTH/Ex-job_Docs/Simulations_CsCl/
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
from loki.RingData import RadialProfile,DiffCorr, InterpSimple ## scripts located in RingData-folder ##
#from pylab import *	# load all Pylab & Numpy
# %pylab	# code as in Matlab
import os, time
this_dir = os.path.dirname(os.path.realpath(__file__)) ## Get path of directory
#this_dir = os.path.dirname(os.path.realpath('CCA_RadProf_84-run.py')) ##for testing in ipython
if "/home/" in this_dir: #/home/cldah/cldah-scratch/ or /home or 
	os.environ['QT_QPA_PLATFORM']='offscreen'

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Radial Profile and Auto-Correlations.")

parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
      help="The Location of the directory with the cxi-files (the input).")

parser.add_argument('-o', '--outpath', dest='outpath', default='dir_name', type=str, help="Path for output, Plots and Data-files")

#parser.add_argument('-f', '--fname', dest='sim_name', default='test', type=str,
#      help="The Name of the Simulation. Default name is 'test'.")

parser.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
      help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)

args = parser.parse_args()
## ---------------- ARGPARSE  END ----------------- ##



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
	Reads simulation data from 'data_file' and appends to list-objects

		In:
	data_file           the cxi-file/s from Condor to be read in.
	ip                  The intensity pattern, amplitude patterns and patterson image respectively as list, the 
						     loaded vaules from 'data_file' are appended to these lists.
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

###################### Choose Calculations: ########################################
add_noise = False						## Add Generated Noise to Correlations ##
random_noise_1quad =False			## Generate random Noise in one quardant ##

plotting = True 				## Plot the Mean Radial Profile

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
	key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0][0] )) 	## For sorting simulated-file 0-90 ##
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
n_spread = fnames[0].split('-sprd')[-1].split(')_')[0]
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
##################################################################################################################
##################################################################################################################
#------------------------------------ Loki XCCA: ----------------------------------------------------------------#
##################################################################################################################
##################################################################################################################
print "\n Data Analysis with LOKI.\n"

# ---- Generate a Storage File's Prefix: ---- 
prefix = '%s_%s_' %(name,pdb)
out_fname = os.path.join( outdir, prefix)       # with directory 


# ---- Some Useful Functions : ----/t/ from Sacla-tutorial
pix2invang = lambda qpix : np.sin(np.arctan(qpix*(ps*1E-6)/dtc_dist )*0.5)*4*np.pi/wl_A
invang2pix = lambda qia : np.tan(2*np.arcsin(qia*wl_A/4/pi))*dtc_dist/(ps*1E-6)


# ---- Set Working parameter 'img' to selected data: ----                                                     
m_b = mask_better       # if converting to int directly after load()                                          
#print "MAX(int): ", m_b.max(), " Min(int): ", m_b.min()                                                      
if run_with_MASK: img = Ip_w_mb #or amplitudes:  np.abs(Ap_w_mb) ## All Patterns with Better-Mask-Assembled #\
#  ######### INT.PATT. OR AMPL.PATT                                                                           
else: img = intensity_pattern   ## All Patterns saved, give shorter name for simplicity ##                    

# ---- Generate a Storage File for Data/Parameters: ----                                                      
data_hdf = h5py.File( out_fname + 'Radial-Profile_w_Loki.hdf5', 'a')    # a: append   

##################################################################################################################
##################################################################################################################

#
# ---- Calculate the Radial Profile for all Diffraction Patterns: ----##
#
print "\n Generating Radial Profile Instance..."
t_RF1 = time.time()

# ---- RadialProfile - instance: ----
r_p = RadialProfile(cntr, m_b.shape, mask = m_b) #dim of img[0]: (1738, 1742)

# --- Radial Profile Calculate: ----
print "\n Starting to Calculate the Radial Profile..."
#rad_pros = np.array( [ r_p.calculate(img[i]) for i in range( N ) ] ) # Rad profile of all
RP_list = []
for k,i in zip(fnames, range(N))
	r_p.calculate(load_cxi_data(data_file= k, idx=i)*mask_better)
 	RP_list.extend(r_p)
rad_pros = np.asarray(RP_list)
rad_pro_mean = rad_pros.mean(0)

# ---- Calculate Q's [1/A] for using in plotting: ----
#radii = np.arange( rad_pros[0].shape[0])	# from GDrive: CXI-2018_MArting/scripts/plot_wax2
#qs = pix2invang(radii) # ps[um]= (ps*1E-6)[m]; dtc_dist[m]: wl_A[A]	
radii_c = np.arange( (rad_pros[0].shape[0]-1)/2) 	## For plotting from center of detector and not the edge, index start at 0 ##
qs_c = pix2invang(radii_c) 		## For plotting from center of detector and not the edge ##

# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
if 'radial_profiles' not in data_hdf.keys(): data_hdf.create_dataset( 'radial_profiles', data = rad_pros)
else: 
	del data_hdf['radial_profiles']
	dset = data_hdf.create_dataset('radial_profiles', data=rad_pros)
#if 'qs_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_for_profile', data = qs)
#else: del data_hdf['qs_for_profile'], dset = data_hdf.create_dataset('qs_for_profile', data=qs)
if 'qs_c_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_c_for_profile', data = qs_c)
else: 
	del data_hdf['qs_c_for_profile']
	dset = data_hdf.create_dataset('qs_c_for_profile', data=qs_c)
del r_p, radii_c 	# clear up memory

# ---- If NOT saving the polar paramters, save only the Radial Profiles ---- #
data_hdf.close()
print "\n File Closed! \n"
t = time.time()-t_RF1
t_m =int(t)/60
t_s=t-t_m*60
print "\n Radial Profile & Calculation Time in LOKI: ", t_m, "min, ", t_s, " s \n" # @Lynch: 38 min, @Prometheus:
##################################################################################################################
##################################################################################################################

#
# ---- Plot the Mean Radial Profile and Auto-Correlation of all Diffraction Patterns: ----##
if plotting:
	rad_cntr = int((rad_pro_mean.shape[0]-1)/2) 	## Center index of radial profile, OBx -1 since index start at 0 not 1 ##
	print "\n Dim of mean(Radial Profile): ", rad_pro_mean.shape, " , center pixel: ", rad_cntr ## (1800,)  ,  899 ##
	start_pixel, end_pixel = (rad_cntr + qmin_pix), (rad_cntr + qmax_pix)
	r_bins = np.linspace(0, (qrange_pix.shape[0]-1), num= 5, dtype=int) ## index for radial binsin pixels ##
	print"r_bins: ", r_bins, "/n Dim r-bins ", r_bins.shape

	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14
	sb_size = 16 #18
	sp_size = 18#20
	l_pad = 10
	print "Radial Profile max value for q-range %i -%i: "%(qmin_pix,qmax_pix), rad_pro_mean[start_pixel : end_pixel].max()
	fig1 = pypl.figure('RP', figsize=(22,15))
	## subplot , constrained_layout=True 
	############################## [fig.a] begin ##############################
	pypl.subplot(121)
	axb = pypl.gca()  ## Bottom Axis ##
	#axb.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad) 
	#axb.plot(qs_c[qmin_pix:qmax_pix], rad_pro_mean[start_pixel : end_pixel], lw=2 , color='b', label='center -> right')
	#axb.plot(qs_c[qmin_pix:qmax_pix], rad_pro_mean[(rad_cntr - qmax_pix):(rad_cntr - qmin_pix)], lw=2, color='r', label='left <- center')
	axb.set_xlabel("r (pixels)", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##
	axb.plot(rad_pro_mean, color='b', lw=2) ## for Bottom-Axis in r [pixels] ##
	#axb.plot(rad_pro_mean[(rad_cntr - qmax_pix):(rad_cntr - qmin_pix)], color='r', lw=2, label='left <- center')
	axb.set_ylabel("Mean ADU", fontsize=axis_fsize)
	axb.tick_params(axis='x')#, labelcolor=color) 	## 'x' in plot
	#axb.set_xticklabels(qrange_pix[r_bins]) 
	#axb.set_xticks(np.linspace(0.0, 1.0, num= r_bins.shape[0], dtype=float)) # xticks(np.arange(0, 1, step=0.2))
	legend = axb.legend() #pypl.legend()  ##(loc='upper center', shadow=True, fontsize='x-large')
	axt = axb.twiny()  ## Top Axis, Share the same y Axis ##	
	#axt.set_xlabel("r (pixels)", fontsize=axis_fsize, labelpad=l_pad)   ## for Bottom-Axis in r [pixels] ##
	axt.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##
	#axt.plot(qs_c[qmin_pix:qmax_pix], rad_pro_mean[start_pixel : end_pixel]),ax2.tick_params(axis='x')#, labelcolor=color)
	axt.set_xticklabels(['%.3f'%(pix2invang(X)) for X in axb.get_xticks()]) ## for Top-Axis in q [inv Angstrom] ##
	axt.set_xticks(axb.get_xticks())
	axt.set_xbound(axb.get_xbound())
	#axt.set_xticklabels(np.arange(qmin_pix,qmax_pix))

	rad_title =pypl.title('Radial Profile [from edge]',  fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
	rad_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	############################## [fig.a] End ##############################

		############################## [fig.b] begin ##############################
	pypl.subplot(122)
	axb = pypl.gca()  ## Bottom Axis ##
	#axb.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad) 
	#axb.plot(qs_c[qmin_pix:qmax_pix], rad_pro_mean[start_pixel : end_pixel], lw=2 , color='b', label='center -> right')
	#axb.plot(qs_c[qmin_pix:qmax_pix], rad_pro_mean[(rad_cntr - qmax_pix):(rad_cntr - qmin_pix)], lw=2, color='r', label='left <- center')
	axb.set_xlabel("r (pixels)", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##
	axb.plot(qrange_pix, rad_pro_mean[ start_pixel : end_pixel ], color='b', lw=2, label='center -> right') ## for Bottom-Axis in r [pixels] ##
	#axb.plot(rad_pro_mean[(rad_cntr - qmax_pix):(rad_cntr - qmin_pix)], color='r', lw=2, label='left <- center')
	axb.set_ylabel("Mean ADU", fontsize=axis_fsize)
	axb.set_ylim(bottom=-2.E-8, top=2.E-8) ## lowest value in tics of RAdial Profile 1-250 of 100 Patterns ##
	#axb.set_xlim(qmin_pix,qmax_pix)
	axb.tick_params(axis='x')#, labelcolor=color) 	## 'x' in plot
	#axb.set_xticklabels(qrange_pix[r_bins]) 
	#axb.set_xticks(np.linspace(0.0, 1.0, num= r_bins.shape[0], dtype=float)) # xticks(np.arange(0, 1, step=0.2))
	legend = axb.legend() #pypl.legend()  ##(loc='upper center', shadow=True, fontsize='x-large')
	axt = axb.twiny()  ## Top Axis, Share the same y Axis ##	
	#axt.set_xlabel("r (pixels)", fontsize=axis_fsize, labelpad=l_pad)   ## for Bottom-Axis in r [pixels] ##
	axt.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##
	#axt.plot(qs_c[qmin_pix:qmax_pix], rad_pro_mean[start_pixel : end_pixel]),ax2.tick_params(axis='x')#, labelcolor=color)
	axt.set_xticklabels(['%.3f'%(pix2invang(X)) for X in axb.get_xticks()]) ## for Top-Axis in q [inv Angstrom] ##
	axt.set_xticks(axb.get_xticks())
	axt.set_xbound(axb.get_xbound())
	#axt.set_xticklabels(np.arange(qmin_pix,qmax_pix))

	rad_title =pypl.title('Radial Profile [from center]',  fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
	rad_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	############################## [fig.b] End ##############################

	pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)  # hspace=0.2,
	## WORSE left=0.01, right=0.99), BETTER left=0.1, right=0.9, top=0.86...bottom 0.10 r label gone
	#pypl.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9) #top=0.86
	pypl.suptitle("%s_%s_(%s-%s)_[qx-%i_qi-%i_nphi-%i]_w_Mask_%s" %(name,pdb,noisy,n_spread,qmax_pix,qmin_pix, nphi,pttrn), fontsize=sp_size)  # y=1.08, 1.0=too low(overlap radpro title)
	fig_name = "SUBPLOT_mean_Radial-Profile_w_Mask_%s.%s" %(pttrn,frmt)
	#pypl.show()
	pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	del fig_name, rad_pro_mean 	# clear up memory1
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##
