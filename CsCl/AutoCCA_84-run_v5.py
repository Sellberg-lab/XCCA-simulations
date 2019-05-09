#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import CXI-files from simulations with Condor (v1.0) and Cross-Correlate Data
# and plot with LOKI
# cxi-file/s located in the same folder, result saved  in subfolder "/simulation_results_N_X"
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#   Patterson_Image '..._patterson_image_...' are from FFTshift-Fast Fourier transforms-FFTshift
#           of Intensity Patterns =  AutoCorrelated image (can be used as initial guess for phae retrieval)
# 2019-04-25 v5 @ Caroline Dahlqvist cldah@kth.se
#			CrossCA_84-run_v5.py
#			v4 with -s input: choose which simulated file to process
#				without -s: process hdf5-files with stored ACC-cacluations
#				 select if calculate with mask, select if ACC of pairs of differences, differences or all.
#				With experiment properties saved in part-calculation-files and in summed (final) file.
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
###################### Choose Calculations and Set Global variables: ########################################
this_dir = os.path.dirname(os.path.realpath(__file__)) ## Get path of directory
#this_dir = os.path.dirname(os.path.realpath('AutoCCA_84-run_v3.py')) ##for testing in ipython
if "/home/" in this_dir: #/home/cldah/cldah-scratch/ or /home or 
	os.environ['QT_QPA_PLATFORM']='offscreen'
Ampl_image = False			## The complex images in Amplitude Patterns instead of Intensity Patterns ##
#add_noise = False						## Add Generated Noise to Correlations ##
#random_noise_1quad =False			## Generate random Noise in one quardant ##
#ACC_wo_MASK = False  			## if True: Calculate tha ACC without the Mask ##
cart_diff, pol_all, pair_diff, all_DC_diff = False,False,False, False ## only one should be true !!

# ## --------------- ARGPARSE START ---------------- ##
# parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Auto-Correlations.")

# # parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
# #       help="The Location of the directory with the cxi-files (the input).")

# parser.add_argument('-o', '--outpath', dest='outpath', default='this_dir', type=str, help="Path for output, Plots and Data-files")

# parser.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
#       help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)

# ## if not subparser with groups (use if True/false of 'calculate'): ##
# # parser.add_argument('-c', '--calculate', dest='calculate', default=False, action='store_true', 
# #       help="If given as a command, then run calculation, if not given then plot") ##type=float)
# # parser_group_c = parser.add_argument_group('calculate')
# # parser_group_c.add_argument('-d', '--dir-name', dest='dir_name', default='this_dir', type=str,
# #       help="The Location of the directory with the cxi-files (the input).")
# # parser_group_c.add_argument('-s', '--simulation-number', dest='sim_n', default=None, type=int, 
# # 	   help="The number of the pdb-file, which simulation is to be loaded from 0 to 90, e.g. '20' for the file <name>_4M20_<properties>l.cxi")
# # parser_group_c.add_argument('-e', '--exposure', dest='exp_set', required=True, type=str, choices=['pair', 'diffs', 'all'],
# #       help="Select how to auto-correalte the data: 'pair' (pair-wise difference between shots),'diffs', 'all' (all shot without difference).")
# # parser_group_c.add_argument('--no-mask', dest='w_MASK', action='store_false, default=True,
# #       help="Select if the mask is included in the auto-correalte calculations.")
# # parser_group_p = parser.add_argument_group('plot')
# # parser_group_p.add_argument('-p', '--plot', dest='plt_set', default='single', type=str, choices=['single', 'subplot', 'all'],
# #      help="Select which plots to execute and save: 'single' each plot separately,'subplot' only a sublpot, 'all' both single plots and a subplot.")

# args = parser.parse_args()
# ## ---------------- ARGPARSE  END ----------------- ##

# ------------------------------------------------------------
def set_ACC_input(exposure):
	"""
	Set how to do the Auto-correlation: 
		on difference of 'pairs' of data (image1-image2, image3-image4, ...)
		on difference of all indices, 'diffs', of data (image1-image2, image2-image3, ...)
		on all indices, 'all', of data without any differences (image1, image2, image3 ...)
	"""
	if exposure.lower() == "pair" : global pair_diff; pair_diff = True
	elif  exposure.lower() == "diffs" : global cart_diff; cart_diff = True
	elif  exposure.lower() == "all" : global pol_all; pol_all = True
	elif  exposure.lower() == "all-dc" : global all_DC_diff; all_DC_diff = True


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

# ------------------------------------------------------------
#def load_cxi_data(data_file, ip, ap, get_parameters=False):
def load_cxi_data(data_file, get_parameters=False, Amplitudes = False):
	"""
	Reads simulation data from 'data_file'
	 If Amplitudes is true, then the absolut value of the amplitudes (complex) is returned instead.

		In:
	data_file           the cxi-file/s from Condor to be read in.
	get_parameters      Determines if the static simulation parameters should be retured, Default = False.
	##ip, ap              The intensity pattern, amplitude patterns and patterson image respectively as list, the 
						     loaded vaules from 'data_file' are appended to these lists.
	=====================================================================================
		Out:
	intensity/amplitude_patterns   the loaded intensity/amplitude patterns from the cxi.file.
	#ip, ap              list-objects with the data loaded from 'data_file' appended.
	photon_e_eV         Photon Energy of source in electronVolt [eV], type = int, (if get_parameters = True)
	wl_A                Wavelength in Angstrom [A], type = float, (if get_parameters = True)
	ps                  Pixel Size [um], type = int, (if get_parameters = True)
	dtc_dist            Detector Distance [m], type = float, (if get_parameters = True)

	"""
	with h5py.File(data_file, 'r') as f:
		if get_parameters:
			photon_e_eV = np.asarray(f["source/incident_energy"] )			# [eV]
			photon_e_eV = int(photon_e_eV[0]) 								# typecast 'Dataset' to int

			photon_wavelength = np.asarray(f["source/incident_wavelength"]) #[m]
			photon_wavelength= float(photon_wavelength[0])					# typecast 'Dataset' to float
			wl_A = photon_wavelength*1E+10 									#[A]

			psa = np.asarray(f["detector/pixel_size_um"])  					#[um]
			ps = int(psa[0]) 												# typecast 'Dataset' to int

			dtc_dist_arr = np.asarray(f["detector/detector_dist_m"])		#[m]
			dtc_dist = float(dtc_dist_arr[0]) 								# typecast 'Dataset' to float

			shots_recorded = f["entry_1/data_1/data"].shape[0]

			if Amplitudes:
				amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"])
			else: 
				intensity_pattern = np.asarray(f["entry_1/data_1/data"])

		else:
			if Amplitudes:
				amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"])
			else: 
				intensity_pattern = np.asarray(f["entry_1/data_1/data"])

	if get_parameters:
		if Amplitudes:
			return photon_e_eV, wl_A, ps, dtc_dist, shots_recorded, np.abs(amplitudes_pattern)
		else:
			return photon_e_eV, wl_A, ps, dtc_dist, shots_recorded,intensity_pattern
	else:
		if Amplitudes:
			#ap.extend(amplitudes_pattern), return ap
			return np.abs(amplitudes_pattern)
		else:
			#ip.extend(intensity_pattern),  return ip
			return intensity_pattern
# ------------------------------------------------------------

#def calc_acc(img, cntr, q_map, mask, data_hdf):
def calc_ac_and_cc(img, cntr, q_map, mask, data_hdf, 
		beam_eV, wavelength_A, pixel_size_m, detector_distance_m):
	"""
	Caclulate the Auto-Correlation and the Cross-Correlation for the imported 'data' with LOKI.

	In:
	================
	img 			the shots loaded from the specific CXI-file. dim (N,Y,X)
	cntr 			the beam center (Y,X)
	q_map 			the Qs stored in inv Angstrom and in pixels. dim (k,3),
						with  column 0 = index, column 1 = q [inv A], column 2: q [pixels]
	mask 			Pixel Maske with dead or none pixels masked out. no pixel = 0
						pixel = 1. dim (Y,X) must be the same X,Y as in images.
	data_hdf 		the hdf5-file handler for storing the data.
	beam_eV 		the beam energy [eV] from the simulation
	wavelength_A  	the beam wavelength [A] from the simulation
	pixel_size_m 	the pixel size [m] for the detector in the simulation
	detector_distance_m 	the distance [m] between the sample and the detctor in the simulation
	"""
	## ---- Some Useful Functions : ----/t/ from Sacla-tutorial                                                    
	#pix2invang = lambda qpix : np.sin(np.arctan(qpix*(ps*1E-6)/dtc_dist )*0.5)*4*np.pi/wl_A
	#img= images
	IP_sum =  np.sum(img, 0)
	print "\n dim of IP_sum : ", IP_sum.shape
	t_AC = time.time()
	#cart_diff, pol_all, pair_diff = False,True,False ## only one should be true !!


	shot_count = img.shape[0]
	# ---- Single Pixel Resolution at Maximum q [1/A]: -----
	nphi = 360#180#
	#### alt. (not tested) Try other version from Loki/lcls_scripts/get_polar_data_and_mask: ####
	#nphi_flt = 2*np.pi*qmax_pix #[ qmax_pix = interp_rmax in pixels, OBS adjusted edges in original ]
	#phibins = qmax_pix - qmin_pix 		# Choose how many, e.g. same as # q_pixls
	#phibin_fct = np.ceil( int(nphi_flt)/float(phibins) )
	#nphi = int( np.ceil( nphi_flt/phibin_fct )*phibin_fct )
	#print "\n nphi: ", nphi 	

	# ---- Save g-Mapping (the q- values [qmin_pix,qmin_pix]): ----
	#q_map = np.array( [ [ind, pix2invang(q)] for ind,q in enumerate( np.arange( qmin_pix, qmax_pix))])
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2] ## first and last values in cloumn 2 = q [pixels] ##

	# ---- Interpolater initiate : ----	# fs = fast-scan {x} cntr[0]; Define a polar image with dimensions: (qRmax-qRmin) x nphi
	Interp = InterpSimple( cntr[0], cntr[1], qmax_pix, qmin_pix, nphi, mask.shape)

	# ---- Make a Polar Mask (if ACC_wo_MASK=True then just an array of 1s) : ----
	#if ACC_wo_MASK:	 mask = np.ones_like(mask)
	if not bool(args.w_MASK):	 mask = np.ones_like(mask)
	polar_mask = Interp.nearest(mask, dtype=bool).round() ## .round() returns a floating point number that is a rounded version of the specified number ##
	

	# ---- Calculate the Difference in Intensities and store in a List (N-1 ACC): ----
	if cart_diff:
		#exposure_diffs_cart =img[:-1]-img[1:]				# Intensity Patterns in Carteesian cor
		img_mean = np.array([ img[i][img[i]>0.0].mean() for i in range(img.shape[0]) ]) ## Calculate the Mean of each pattern(shot) ##
		#exposure_diffs_cart= img[:-1,:,:]/img_mean[:-1]-img[1:,:,:]/img_mean[1:]  ## Normalize each pattern(shot) before subtraction ##
		exposure_diffs_cart = np.array([  img[i][:,:]/img_mean[i]-img[i+1][:,:]/img_mean[i+1]    for i in range(img.shape[0]-1) ])
		#exposure_diffs_cart = np.divide(img[:-1,:,:], img_mean[:-1], out=img.shape)#, where=corr_mask!=0) 
		del img,img_mean 	## Do not need the mean-values anymore ##
		gc.collect() ## Free up Memory: ##
		exposure_diffs_cart = np.asarray(exposure_diffs_cart) 	# = (4, 1738, 1742)
		exposure_diffs_cart = polarize(exposure_diffs_cart)  ## Polarize (if no fraction given => just 1s) ##
		#exposure_diffs_cart = norm_data(exposure_diffs_cart):
		## ---- Conv to polar of diff-data : ---- ##
		print "\n Starting to Calculate the Polar Images..."
		exposure_diffs_cart = np.array( [ polar_mask* Interp.nearest(exposure_diffs_cart[i]) for i in range(exposure_diffs_cart.shape[0]) ] ) 
		pttrn = "cart-diff"
		# if 'exposure_diffs_cart' not in data_hdf.keys(): data_hdf.create_dataset( 'exposure_diffs_cart', data = np.asarray(exposure_diffs_cart))
		# else: 
		# 	del data_hdf['exposure_diffs_cart']
		# 	dset = data_hdf.create_dataset('exposure_diffs_cart', data=np.asarray(exposure_diffs_cart))
		
		exposures = exposure_diffs_cart 
		#DC=DiffCorr( exposure_diffs_cart  )
		del  exposure_diffs_cart 
		gc.collect() ## Free up Memory: ##

		## ---- Calc from polar Images, ALL shots: ---- ##
	elif pol_all or all_DC_diff:
		# ---- Generate Polar Image/Images (N Diffracton Patterns) : ----
		print "\n Starting to Calculate the Polar Images..."
		polar_imgs = np.array( [ polar_mask* Interp.nearest(img[i]) for i in range( img.shape[0]) ] )
		## ---- Normalize - with function: ---- ##
		## alt try: polar_imgs_norm = norm_data(polar_imgs)

		exposures =  polar_imgs # polar_imgs_norm
		#DC=DiffCorr( polar_imgs  )
		del img, polar_imgs #,polar_imgs_norm
		gc.collect() ## Free up Memory: ##
		#exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)
		#pttrn = "polar-diff"
		pttrn = "polar-all"
		if all_DC_diff: pttrn = "all-DC-diff"
		

	# 	# ---- Let the DC-class perform the difference calculation ----#
	# elif all_DC_diff:
	# 	# ---- Generate Polar Image/Images (N Diffracton Patterns) : ----
	# 	print "\n Starting to Calculate the Polar Images..."
	# 	polar_imgs = np.array( [ polar_mask* Interp.nearest(img[i]) for i in range( img.shape[0]) ] )
	# 	## ---- Normalize - with function: ---- ##
	# 	## alt try: polar_imgs_norm = norm_data(polar_imgs)
	# 	#polar_imgs_norm = polar_imgs

	# 	exposures =  polar_imgs	# Polar Imgs
	# 	#DC=DiffCorr( polar_imgs, q_values=None, k=None ,delta_shot=None,pre_dif=False)
	# 	del img, polar_imgs #,polar_imgs_norm
	# 	gc.collect() ## Free up Memory: ##
	# 	#exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)
	# 	pttrn = "all-DC-diff"

		## ---- Calc difference of images from Pairs of Images and convert to polar diff-pairs-images: ---- ##
	elif pair_diff :
		exp_diff_pairs = img[:-1:2]-img[1::2]
		del img
		gc.collect() ## Free up Memory: ##
		exp_diff_pairs = np.asarray(exp_diff_pairs)
		print "\n Starting to Calculate the Polar Images..."
		exp_diff_pairs = np.array( [ polar_mask* Interp.nearest(exp_diff_pairs[i]) for i in range(exp_diff_pairs.shape[0]) ] ) 
		pttrn = "pairs"

		exposures = exp_diff_pairs #exposure_diffs_cart #exposure_diffs_pol
		#DC=DiffCorr( exp_diff_pairs  )
		del exp_diff_pairs 
		gc.collect() ## Free up Memory: ##

	print "exposures vector's shape", exposures.shape ## ( Nshots x Nq x Nphi) ##
	del Interp

	diff_count = exposures.shape[0]  ## The number off differenses used in the correlation cacluations ##

	# ---- DiffCorr instance: ----##
	if all_DC_diff: DC=DiffCorr( exposures , delta_shot=None, pre_dif=False)
	else: DC=DiffCorr( exposures  ) # pre_dif=True

	# ---- Autocorrelation of each exposure: ---- ##
	print "\n Starting to Auto-Correlate the Images... "
	#acorr = [RingData.DiffCorr( exposure_diffs_cart).autocorr(), RingData.DiffCorr( exposure_diffs_pol ).autocorr()]
	#cor_mean = [RingData.DiffCorr( exposure_diffs_cart).autocorr().mean(0), RingData.DiffCorr( exposure_diffs_pol ).autocorr().mean(0)]
	#acorr = DiffCorr( exposures ).autocorr()
	acorr = DC.autocorr()
	print "\n Dim of ACC: ", acorr.shape

	# ---- Crosscorrelation of each Pair (? intershell-CC): ---- ##
	ccorr_q_sum_of_shots = []
	#ccorr_q_mean_of_shots = []
	#ccorr_all = np.zeros( (exposures.shape[1],exposures.shape[0],exposures.shape[1], exposures.shape[2]) )
	for qindex in range(exposures.shape[1]): ## Nshots x (Nq x Nphi), where shots=exposures (all or differences); # = 'diff_count'  ##
		cross_corr = DC.crosscorr(qindex)  ##  Nshots x (Nq x Nphi) ##
		ccorr_sum = np.sum(cross_corr, 0)  		## sum over all the shots: (Nq x Nphi) ##
		ccorr_q_sum_of_shots.append(ccorr_sum) ## in last loop: Qx(Q,phi)
		#ccorr_mean = cross_corr.mean(0) 		## mean over all the shots ##
		#ccorr_q_mean_of_shots.extend(ccorr_mean) ## (N,Q,phi)
		#ccorr_all[qindex, :, :, :]= cross_corr
	ccorr_q_sum_of_shots = np.asarray(ccorr_q_sum_of_shots) ## ? (qindex x Nq x Nphi) ##
	#ccorr_q_mean_of_shots = np.asarray(ccorr_q_mean_of_shots) ## ? (qindex x Nq x Nphi) ##
	print "\n Dim of <crosscorr> of all qindex: ", ccorr_q_sum_of_shots.shape
	del exposures

	# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
	#if ACC_wo_MASK: pttrn = "wo_Mask_"+ pttrn
	if not bool(args.w_MASK): pttrn = "wo_Mask_"+ pttrn
	else: pttrn = "w_Mask_"+ pttrn
	#if 'auto-correlation_mean' not in data_hdf.keys(): data_hdf.create_dataset( 'auto-correlation_mean', data = np.asarray(cor_mean))
	#else: 
	#	del data_hdf['auto-correlation_mean']
	#	dset = data_hdf.create_dataset('auto-correlation_mean', data=np.asarray(cor_mean))
	if 'auto-correlation' not in data_hdf.keys(): 
		dset =data_hdf.create_dataset( 'auto-correlation', data = np.asarray(acorr))
		dset.attrs["diff_type"] = np.string_(pttrn)
		dset.attrs["number_patterns"] = shot_count
		dset.attrs["number_of_diffs"] = diff_count
	else: 
		del data_hdf['auto-correlation']
		dset = data_hdf.create_dataset('auto-correlation', data=np.asarray(acorr))
		dset.attrs["diff_type"] = np.string_(pttrn)
		dset.attrs["number_patterns"] = shot_count
		dset.attrs["number_of_diffs"] = diff_count

	if 'cross-correlation_sum' not in data_hdf.keys(): 
		dset_cc =data_hdf.create_dataset( 'cross-correlation_sum', data = ccorr_q_sum_of_shots)
		dset_cc.attrs["diff_type"] = np.string_(pttrn)
		dset_cc.attrs["number_of_diffs"] = diff_count
	else: 
		del data_hdf['cross-correlation_sum']
		dset_cc = data_hdf.create_dataset('cross-correlation_sum', data=ccorr_q_sum_of_shots)
		dset_cc.attrs["diff_type"] = np.string_(pttrn)
		dset_cc.attrs["number_of_diffs"] = diff_count

	if 'polar_mask' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_mask', data = polar_mask.astype(int)) 
	else: 
		del data_hdf['polar_mask']
		dset = data_hdf.create_dataset('polar_mask', data = polar_mask.astype(int))
	
	if 'num_phi' not in data_hdf.keys():	data_hdf.create_dataset( 'num_phi', data = nphi, dtype='i4') #INT, dtype='i1', 'i8'f16', i4=32 bit
	else: 
		del data_hdf['num_phi']
		dset = data_hdf.create_dataset('num_phi',  data = nphi, dtype='i4')
	
	if 'q_mapping' not in data_hdf.keys():	data_hdf.create_dataset( 'q_mapping', data = q_map)
	else: 
		del data_hdf['q_mapping']
		dset = data_hdf.create_dataset('q_mapping',data = q_map)
	
	if 'beam_center' not in data_hdf.keys():	data_hdf.create_dataset( 'beam_center', data = cntr)
	else: 
		del data_hdf['beam_center']
		dset = data_hdf.create_dataset('beam_center',data = cntr)
	
	if 'sum_intensity_pattern' not in data_hdf.keys(): 
		dset_ip = data_hdf.create_dataset( 'sum_intensity_pattern', data = np.asarray(IP_sum))
		dset_ip.attrs["number_patterns"] = shot_count
		dset_ip.attrs["detector_distance_m"]= detector_distance_m 
		dset_ip.attrs["wavelength_Angstrom"]= wavelength_A
		dset_ip.attrs["pixel_size_m"]= pixel_size_m 
		dset_ip.attrs["beam_energy_eV"]= beam_eV
	else: 
		del data_hdf['sum_intensity_pattern']
		dset_ip = data_hdf.create_dataset('sum_intensity_pattern', data=np.asarray(IP_sum))
		dset_ip.attrs["number_patterns"] = shot_count
		dset_ip.attrs["detector_distance_m"]= detector_distance_m 
		dset_ip.attrs["wavelength_Angstrom"]= wavelength_A 
		dset_ip.attrs["pixel_size_m"]= pixel_size_m 
		dset_ip.attrs["beam_energy_eV"]= beam_eV

	# ---- Save by closing file: ----
	data_hdf.close()
	print "\n File Closed! \n"

	t = time.time()-t_AC
	t_m =int(t)/60
	t_s=t-t_m*60
	print "AutoCorrelation Time: ", t_m, "min, ", t_s, " s "
	print " --------------------------------------------------------------------- "
	exit(0)
# ------------------------------------------------------------
#read_and_plot_acc(fnames, out_hdf, out_fname) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]
#def read_and_plot_acc(list_names, data_hdf, out_name):
def read_and_plot_acc(list_names, data_hdf, out_name, mask):
	"""
	Read in ACC data from multiple files in 'list_names' list of filenames. 
	Store the resut in 'data_hdf' and then 
	Plot the average from all read-in files (saved in 'out_name'-location).
	"""

	## ---- Load Calculations from Files and Calculate the Mean : ---- ##
	t_load =time.time()
	ip_mean = []
	corr_sum = []
	cross_corr_sum = []
	tot_corr_sum = 0	## the number of ACC perfomed, one per loaded file ##
	tot_shot_sum = 0	## the total number of shots (diffraction pattersn) from simulation ##
	tot_diff_sum = 0	## the number of diffs or the number of patterns in input to ACC ##
	q_map= None
	nphi= None
	diff_type_str= None
	for file in list_names:
		with h5py.File(file, 'r') as f:
			dset_ac = f['auto-correlation'] ## Data-set with ACCs ##
			acc = np.asarray(dset_ac)		#np.asarray(f['auto-correlation'])
			diff_count = dset_ac.attrs["number_of_diffs"]
			shot_count = dset_ac.attrs["number_patterns"] 
			dset_cc = f['cross-correlation_sum'] ## Data-set with Cross-Correlations (3D) ##
			ccsummed =np.asarray(dset_cc)
			#ip_sum=np.asarray(f['sum_intensity_pattern']) ## (Y,X) ##
			dset_ip_sum = f['sum_intensity_pattern'] ## Data-set with sum diffraction patterns (intensity) ##
			ip_sum=np.asarray(dset_ip_sum)  ## (Y,X) ##
			#print "\n Dim of Sum of intensity patterns: ", ip_sum.shape
			if file==list_names[0]:
				q_map=np.asarray(f['q_mapping'])
				nphi=int(np.asarray(f['num_phi'])) 
				diff_type_str =dset_ac.attrs["diff_type"]
				dtc_dist_m = dset_ip_sum.attrs["detector_distance_m"]
				wl_A = dset_ip_sum.attrs["wavelength_Angstrom"]
				ps_m = dset_ip_sum.attrs["pixel_size_m"]
				be_eV = dset_ip_sum.attrs["beam_energy_eV"]
		corr_sum.extend(acc) ## (N,Q,phi)
		ip_mean.append(ip_sum) ## (Y,X)
		cross_corr_sum.append(ccsummed)  ## (Qidx,Q,phi)
		tot_corr_sum+= 1 ##corr_count ## calculate the number of auto-correlations loaded ##
		tot_shot_sum+= shot_count 
		tot_diff_sum+= diff_count 
	#tot_shot_sum=100*91 ## !OBS:Temp fix for forgotten to store data ##
	corr_sum = np.asarray(corr_sum) ## 91xN(Q,phi)
	corr_sum = np.sum(corr_sum, 0) ## Sum all the ACC (Q,phi) ##
	cross_corr_sum  = np.sum(np.asarray( cross_corr_sum ), 0) ## sum of all the cc-summs  sum(91x(Y,X))##
	#ip_mean = np.asarray(ip_mean )
	print "\n length of array of Sum of intensity patterns: ", len(ip_mean)
	ip_mean  = np.sum(np.asarray(ip_mean ), 0) ## sum of all the intensity patterns  sum(91x(Y,X))##
	print "\n Dim of Sum of intensity patterns: ", ip_mean.shape
	#ip_mean=ip_mean.astype(float) 	## convert to float else porblem with 'same_kind' when dividing with float  ##
	#ip_mean /= float(tot_shot_sum)	 ## mean of all the intensity patterns in float ##
	#ip_mean = np.around(ip_mean).astype(int)	 ## Round up/down ##
	ip_mean /= tot_shot_sum		## Alt. work only in int ##
	print "\n Dim of Mean of intensity patterns: ", ip_mean.shape
	print "\n Dim of ACC: ", corr_sum.shape
	print "\n Dim of intershell-CC: ", cross_corr_sum.shape
	t = time.time()-t_load
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Loading Time for %i patterns: "%(len(list_names)), t_m, "min, ", t_s, "s " # 5 min, 19.2752921581 s


	## ---- Load the Mask : ---- ##
	#nphi = 360 ## Temp fix, stored nphi is 127 (with dtype='i4') ##
	print "\n nphi = %i, " %(nphi)
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2]
	cntr =np.asarray([(mask.shape[1]-1)/2.0, (mask.shape[0]-1)/2.0 ]) ## (X,Y)
	Interp = InterpSimple( cntr[0], cntr[1], qmax_pix, qmin_pix, nphi, mask.shape) 
	mask_polar = Interp.nearest(mask, dtype=bool)
	print "\n dim of mask converted to polar space: ", mask_polar.shape
	## ---- Auto-Correlate the Mask : ---- ##
	mask_DC = DiffCorr(mask_polar.astype(int), pre_dif=True)
	mask_acc = mask_DC.autocorr()
	print "\n dim of ACC(Mask): ", mask_acc.shape

	# ---- Crosscorrelation of each Pair (? intershell-CC => add 2 mask in series (2,Q,phi)): ---- ##
	mask_crosscor = []
	mask_polar_w_eXdim = np.zeros([1,mask_polar.shape[0],mask_polar.shape[1]])
	mask_polar_w_eXdim[0] =mask_polar ## New Array with the polar mask but with a 3rd dim: (1,Q,phi) ##
	mask_DC_w_Xdim = DiffCorr(mask_polar_w_eXdim.astype(int), pre_dif=True)
	for qindex in range(corr_sum.shape[1]): ## Nshots x (Nq x Nphi), where shots=exposures (all or differences); # = 'diff_count'  ##
		mask_crosscor_qindex = mask_DC_w_Xdim.crosscorr(qindex)  ## require len=3 #
		ccorr_sum = np.sum(mask_crosscor_qindex, 0) 
		mask_crosscor.append(ccorr_sum) ## in last loop: Qx(Q,phi)
		del ccorr_sum
	mask_crosscor = np.asarray(mask_crosscor) ##  (qindex x Nq x Nphi) ##


	## ---- Store the Total Results : ---- ##
	dset_acc = data_hdf.create_dataset( 'auto-correlation-sum', data = np.asarray(corr_sum))
	dset_acc.attrs["diff_type"]= np.string_(diff_type_str)
	dset_acc.attrs["tot_number_of_diffs"]= tot_diff_sum
	dset_acc.attrs["tot_number_patterns"]= tot_shot_sum
	dset_acc.attrs["tot_number_of_corrs"]= tot_corr_sum

	data_hdf.create_dataset( 'mask_auto-correlation', data = np.asarray(mask_acc))
	data_hdf.create_dataset( 'mask_cross-correlation', data = np.asarray(mask_crosscor))
	data_hdf.create_dataset( 'q_mapping', data = q_map)
	#data_hdf.create_dataset( 'tot-corr', data = tot_corr_sum, dtype='i4' )
	#data_hdf.create_dataset( 'tot-shots', data = tot_shot_sum, dtype='i4' )
	data_hdf.create_dataset( 'num_phi', data = nphi, dtype='i4') ## i4 = 32 -bit ##
	#data_hdf.create_dataset( 'diff_type', data = diff_type_str, dtype=h5py.special_dtype(vlen=str))

	dset_ip = data_hdf.create_dataset( 'mean_intensity_pattern', data = np.asarray(ip_mean))
	dset_ip.attrs["detector_distance_m"]= dtc_dist_m 
	dset_ip.attrs["wavelength_Angstrom"]= wl_A 
	dset_ip.attrs["pixel_size_m"]= ps_m 
	dset_ip.attrs["beam_energy_eV"]= be_eV

	dset_cc = data_hdf.create_dataset( 'cross-correlation_sum', data = np.asarray(cross_corr_sum))
	dset_cc.attrs["diff_type"]= np.string_(diff_type_str)
	dset_cc.attrs["tot_number_of_diffs"]= tot_diff_sum
	dset_cc.attrs["tot_number_patterns"]= tot_shot_sum
	dset_cc.attrs["tot_number_of_corrs"]= tot_corr_sum
	# ---- Save by closing file: ----
	data_hdf.close()
	print "\n File Closed! \n"

	if str(args.plt_set).lower() in ('single', 'all'):
		plot_acc(corr_sum,tot_corr_sum,q_map,nphi,diff_type_str,mask_acc,ip_mean,tot_shot_sum,dtc_dist_m,wl_A,ps_m,be_eV, cross_corr_sum,mask_crosscor,  out_name, mask=mask) ## tot_shot_sum
	if str(args.plt_set).lower() in ('subplot','all'):
		subplot_acc(corr_sum,tot_corr_sum,q_map,nphi,diff_type_str,mask_acc,ip_mean,tot_shot_sum, out_name, mask=mask) ## tot_shot_sum
		#subplot_acc(corr_sum,tot_corr_sum,q_map,nphi,diff_type_str,mask_acc,ip_mean,tot_shot_sum,dtc_dist_m,wl_A,ps_m,be_eV, out_name, mask=mask) ## tot_shot_sum


	exit(0)  ## Finished ! ##

# ------------------------------------------------------------

def plot_acc(corrsum,corr_count, q_map, nphi, pttrn,corr_mask, IP_mean, shot_count,dtc_dist_m,wl_A,ps_m,be_eV ,cross_corrsum,mask_crosscor,  out_fname, mask=None):
	"""
	Read in ACC data from multiple files in '' list of filenames. 
	Plot the average from all read-in files.

	In:
	corrsum 			The total sum of all the auto-correlations (Qs, phi)
	corr_count 			 The total number of corrlations calculated in the 'corrsum',
							e.g. the total number of differeneses that was used to calculate the correlations.
	q_map 				The momentum transfer vaules, Qs', in pixels for which the correlations were calculated.
	nphi 				The angles for which the correlations were calcucated (e.g. 180 or 360).
	pttrn 				A string informing which differences were used in the correlation calcutaion, such as 
							if the correlatsion were calculated from pairwise diffreence or N-1 diffrens for N shots.
	corr_mask			The Auto-correlation of the mask.
	IP_mean 			The Mean of all Intensity Patterns (or Amplitude Patterns).
	mask 				The Detector Mask (for plotting the intensity Pattern wth mask)
	shot_count 			The total number of shots summed in 'IP_mean' and loaded.
	cross_corrsum 		 ... (Qindex, Qs, phi)
	out_fname 			The pathname and prefix of the filename of the plot.

	Out:				Plot of the auto-correlation.
	"""
	print "\n Plotting..."

	## ---- Local Parameters for all plots: ---- ##
	frmt = "eps" 	## Format used to save plots ##
	## columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2] ## first and last values in cloumn 2 = q [pixels] ##
	radii = q_map[:,2] 	 ## column 2: q [pixels]. ##

	## ---- Divide the Auto-Correlation(data) with the Auto-correlation(Mask): ---- ##
	corrsum_m = np.divide(corrsum, corr_mask, out=None, where=corr_mask!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
	
	## ---- Calculate the average intensity in Q (row-bby-row): ---- ##
	qs_mean= np.asarray([	corrsum_m[i,:].mean() for i in range(corrsum_m.shape[0])	])  ## (Q,phi) where row: Q; columns: phi ##
	# qs_mean = corrsum_m.mean(axis=0) ## Row-wise mean ##
	print "\n Dim of Qs mean: ", qs_mean.shape

	## interesting pixels/invA:  400 - 487/1.643 invAA : ##
	r_pixel=425 	## approx 1.45 invAA ##
	idx = np.argmin( np.abs(   radii-r_pixel  ) ) ## the index for the pixel ##

	## ---- Divide the Cross-Correlation(data) at Q=idx with the Cross-Correlation(Mask): ---- ##
	cross_sum_m = np.divide(cross_corrsum[idx], mask_crosscor[0], out=None, where=mask_crosscor!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##


	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14		## Fontsize of axis labels ##
	tick_fsize = 12		## Size of Tick-labels ##
	sb_size = 16 #18 	## Fontsize of Sub-titles ##
	sp_size = 18#20  	## Fontsize of Super-titles ##
	l_pad = 10 			## Padding for Axis Labels ##

	cb_shrink, cb_padd = 1.0, 0.2  ## Colorbar padding ##

	padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181,  ave_corrs-script has 50 ##

	if 'all' in pttrn : frac = 1#0.8 		## Limit the vmax with this percentage, for 2M need 1.0 ##
	else: frac = 1



	def plot_2std_to_file(fig, data, norm_name, fig_name, title=None):
		"""
		Plot a 2D image with limits mean-2*std to mean + 2*std
		in polar coordinates.

		fig			the figure to plot in, figure handle
		data 		the calculated FFT-coefficients to plot, 2D array
		norm_name 	the normalisations except for the mask, string.
		filename	part of figure name unique to plot, string
		"""

		m = data[:,padC:-padC].mean() 	# MEAN
		s = data[:,padC:-padC].std() 	# Standard Deviation
		vmin = m-2*s
		vmax = m+2*s
		
		ax = pypl.gca()
		polar_axis = True 	## Plot with axis in polar coord and with polar labels, else pixels ##

		if polar_axis:
			im = ax.imshow( data,
		                 extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
		                    vmin=vmin, vmax=vmax*frac, aspect='auto')
		else : im =ax.imshow(data, vmin=vmin, vmax=vmax*frac, aspect='auto' )
		cb = pypl.colorbar(im, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)
		cb.set_label(r'Auto-correlation of Intensity Difference [a.u.]',  size=12) # normed intensity
		cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
		cb.update_ticks()

		if polar_axis:
			# #---- Adjust the X-axis: ----  		##
			#xtic = ax.get_xticks()  	#[nphi/4, nphi/2, 3*nphi/4]
			phi=2*np.pi  ## extent of sigma in angu;ar direction ##
			xtic = [phi/4, phi/2, 3*phi/4]
			xlab =[ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
			ax.set_xticks(xtic), ax.set_xticklabels(xlab)

			# #---- Adjust the Y-axis: ----
			nq = q_map.shape[0]-1#q_step 	# Number of q radial polar bins, where q_map is an array q_min-q_max
			#print "q-map : ", q_map
			q_bins = np.linspace(0, nq, num= ax.get_yticks().shape[0], dtype=int)  ## index of axis tick vector ##
			#print "Dim q-bins: ", q_bins.shape[0], " max:", q_bins.max(), " min:", q_bins.min()
			#print "q-bins array: ", q_bins, "/n Dim q-bins ", q_bins.shape
			q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
			print "q_label: ", q_label

			ytic=ax.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
			#ylab= q_label
			ax.set_yticklabels(q_label)
			ax.set_yticks(ytic)

			# #---- Label xy-Axis: ----
			ax.set_xlabel(r'$\phi$', fontsize = axis_fsize)
			ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)
		pypl.gca().tick_params(labelsize=axis_fsize, length=9)
		if title is None:
			Norm_ave_corr_title = ax.set_title(r"Average of %d corrs [%s] %s with limits $\mu \pm 2\sigma$"%(corr_count, pttrn, norm_name),  fontsize=sb_size)
		else: 
			Norm_ave_corr_title = ax.set_title(title)
		Norm_ave_corr_title.set_y(1.08) # 1.1)
		############################## [fig.] End ##############################
		fig_name = fig_name + "_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
		#pypl.show()
		#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
		fig3.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
		print "\n Plot saved as %s " %fig_name
		del fig_name 	# clear up memory
		pypl.cla() ## clears axis ##
		pypl.clf() ## clears figure ##



	def plot_even_odd(fig, data, norm_name, fig_name):
		"""
		Plot the even and odd coefficients separately in 
		side-by-side plots

		fig			the figure to plot in, figure handle
		data 		the calculated FFT-coefficients to plot, 2D array
		norm_name 	the normalisations except for the mask, string.
		filename	part of figure name unique to plot, string
		"""
		
		## ---- Phi even-odd = [1,20]: ---- ##
		lim=21#13 	## the number of coefficients to plot. 51 is too many !
		ax_even=fig.add_subplot(121)
		ax_odd=fig.add_subplot(122, sharex=ax_even, sharey=ax_even)
		pypl.subplots_adjust(wspace=0.2, hspace=0.2, left=0.08, right=0.95, top=0.85, bottom=0.15)
		cmap = pypl.get_cmap('jet')
		for i in range(1,lim):  
			color = cmap(float(i)/lim)
			if i % 2 == 0:		## only even i:s ##
				ax_even.plot(data[:,i].real, c=color, lw=2, label= 'n=%i' %i)#'Angle $ \phi=%i $' %i)
				ax_even.set_title("even n:s")
				ax_even.legend()
				q_bins = np.linspace(0, (q_map.shape[0]-1), num= ax_even.get_xticks().shape[0], dtype=int)  ## index of axis tick vector ##
				q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
				ax_even.set_xticklabels(q_label), ax_even.set_xticks(q_bins)
				ax_even.set_xlabel(r'q', fontsize = axis_fsize)

			else:				## only odd ##
				ax_odd.plot(data[:,i].real, c=color, lw=2, label= 'n=%i' %i)#'Angle $ \phi=%i $' %i)
				ax_odd.set_title("odd n:s")
				ax_odd.legend()
				ax_odd.set_xticklabels(ax_even.get_xticklabels()), ax_odd.set_xticks(ax_even.get_xticks())
				ax_odd.set_xlabel(r'q', fontsize = axis_fsize)
		pypl.subplots_adjust(wspace=0.3, hspace=0.2, left=0.08, right=0.95, top=0.85, bottom=0.15)
			############################### [fig.] End ##############################
		#pypl.title('Re{FFT} $\phi [1,%i] $ of Average of %d corrs [%s] \n(Normalized with Mask)'%(lim,corr_count, pttrn),  fontsize=sb_size)
		pypl.suptitle("Re{FFT} $\phi [1,%i]$ of Average of %d corrs [%s] \n (Normalized with Mask %d)"%(lim,corr_count, pttrn, norm_name),  fontsize=sb_size)
		# #pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)
		#pypl.show()
		fig_name = fig_name + 'phi-1-%i_(qx-%i_qi-%i_nphi-%i)_%s.%s'%(lim,qmax_pix,qmin_pix, nphi ,pttrn,frmt)
		fig.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
		print "\n Subplot saved as %s " %fig_name
		del fig_name, data, norm_name 	# clear up memory
		gc.collect() 
		pypl.cla() ## clears axis ##
		pypl.clf() ## clears figure ##



	def plot_radial_profile(fig, data, norm_name, fig_name):
		"""
		Plot the square root 0th Fourier Coefficient, a.k.a the 'Radial Profile',
		'Azimuthal Average'

		fig			the figure to plot in, figure handle
		data 		the calculated FFT-coefficients to plot, 2D array
		norm_name 	the normalisations except for the mask, string.
		filename	part of figure name unique to plot, string
		"""

		## ---- Bottom x-Axis: ---- ##
		ax_bttm = pypl.gca()
		ax_bttm.set_xlabel("r [pixels]", fontsize=axis_fsize, labelpad=l_pad) ## for Bottom-Axis in r [pixels] ##

		## ---- Phi = 0 : ---- ##
		#ax_bttm.plot(data[:,0].real, lw=2)	## plot the 0th coefficient ##
		ax_bttm.plot(np.sqrt(data[:,0].real), lw=2) ## sqrt of 0th F-coeffient = radial profile ##
		ax_bttm.set_ylabel("Radial Intensity (ADU)", fontsize=axis_fsize)

		qp_bins = np.linspace(0, q_map.shape[0], endpoint=False, num= ax_bttm.get_xticks().shape[0], dtype=int) #  endpoint=True,
		qp_label = [ '%i'%(q_map[x,2]) for x in qp_bins] 	## Columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		ax_bttm.set_xticks(qp_bins)	## ? ticks in wrong place, 300-800 in pixles ## bins as index along the array of data[:,0].real
		ax_bttm.set_xticklabels(qp_label)
		ax_bttm.grid(b=True, linestyle=':')
		## grid(b=None, which='major', axis='both', **kwargs) **kwargs:(color='r', linestyle='-', linewidth=2)
		## linestyle	{'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
		
		## ---- Top x-Axis: ---- ##
		ax_top = ax_bttm.twiny()  ## Top Axis, Share the same y Axis ##	
		ax_top.set_xlabel("Q $(\AA^{-1})$", fontsize=axis_fsize, labelpad=l_pad)  ## for Top-Axis in q [inv Angstrom] ##	
		qA_label = [ '%.3f'%(q_map[x,1]) for x in qp_bins] ## Columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		ax_top.set_xticks(ax_bttm.get_xticks())
		ax_top.set_xticklabels(qA_label)
		ax_top.set_xbound(ax_bttm.get_xbound())
		pypl.gca().tick_params(labelsize=tick_fsize, length=9)

		#main_title=pypl.title('Re{FFT} $\phi = 0 $ of Average of %d corrs [%s] (Normalized with Mask %s )'%(corr_count, pttrn, norm_name),  fontsize=sb_size)
		main_title=pypl.title('$ \sqrt{Re\{\textrm{FFT}(\phi)\}}|_{n=0}$  of Average of %d corrs [%s] \n(Normalized with Mask %s )'%(corr_count, pttrn, norm_name),  fontsize=sb_size)
	
		main_title.set_y(1.1) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
		############################### [fig.13] End ##############################
		fig_name = fig_name +  "FFT-angle-phi-0_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
		fig.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
		print "\n Subplot saved as %s " %fig_name
		del fig_name, data, norm_name 	# clear up memory
		gc.collect() 
		pypl.cla() ## clears axis ##
		pypl.clf() ## clears figure ##


	##################################################################################################################
	##################################################################################################################
	#---------------------- Fig.1 ACC Normed with Variance ( PHI=0 ): -----------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC."
	fig1 = pypl.figure('AC', figsize=(22,15))
	sig = corrsum/corrsum[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	#sig = corrsum/corrsum[:,0][:,None] / corr_count
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape #carteesian = (1738, 1742): polar =(190, 5)
	
	fig_name = 'Diff-Auto-Corr'
	plot_2std_to_file(fig=fig1, data=sig, norm_name='', fig_name=fig_name)

	#del sig
	#gc.collect()

	##################################################################################################################
	##################################################################################################################
	#--------------- Fig.2 ACC Normalised with correlated Mask & PHI=0 (variance): ----------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC Normalied wit ACC of the Mask."
	fig2 = pypl.figure('AC_N_Msk', figsize=(22,15))
	sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	## withtNomred 0th angle ##
	## every q : row vise normalization with Mean of q : norm each q  bin by average of that q
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape 

	fig_name="Diff-Auto-Corr_Normed_w_Mask"
	norm_name = '(Normalized with Mask)'
	plot_2std_to_file(fig=fig2, data=sig, norm_name=norm_name, fig_name=fig_name)

	#del sig
	#gc.collect()


	##################################################################################################################
	##################################################################################################################
	#--------------------- Fig.3 ACC Normalised with correlated Mask & Qs: ------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC Normalied wit ACC of the Mask."
	fig3 = pypl.figure('AC_N_Qs_Msk', figsize=(22,15))

	#sig = np.array([  corrsum[i,:]/qs_mean[i]    for i in range(corrsum.shape[0]) ])
	sig = corrsum_m/qs_mean[:,None]/corr_count
	sig  =np.nan_to_num(sig)

	fig_name = "Auto-Corr_Q-normed_Normed_w_Mask"
	norm_name = '(Normalized with Mask and Q)'
	plot_2std_to_file(fig=fig3, data=sig, norm_name=norm_name, fig_name=fig_name)

	#del sig
	#gc.collect()


	##################################################################################################################
	##################################################################################################################
	#--------------------- Fig.4 ACC of the Mask: -----------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC of the Mask."
	fig4A = pypl.figure('AC_of_Msk', figsize=(22,15))

	sig_m = corr_mask/corr_mask[:,0][:,None]#/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig_m  =np.nan_to_num(sig_m)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	fig_name="ACC_Mask"
	mask_title = "AC of Mask with limits $\mu \pm 2\sigma$"
	plot_2std_to_file(fig=fig4A, data=sig_m, norm_name='', fig_name=fig_name, title=mask_title)

	#del sig_m
	#gc.collect()


	fig4B = pypl.figure('CC_of_Msk', figsize=(22,15))
	sig_m = mask_crosscor[0]/mask_crosscor[0][:,0][:,None]
	sig_m  =np.nan_to_num(sig_m)
	fig_name="CC_Mask"
	mask_title = "CC of Mask with limits $\mu \pm 2\sigma$"
	plot_2std_to_file(fig=fig4B, data=sig_m, norm_name='', fig_name=fig_name, title=mask_title)

	#del sig_m
	#gc.collect()


	##################################################################################################################
	##################################################################################################################
	#-------------------- Fig.5 Plotting  Mean Intensiyt Pattern (log10) --------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	color_map ='jet' #'viridis'
	print "\n Plotting some Diffraction Patterns."
	fig5 = pypl.figure('Mean_Intensity', figsize=(15,12)) 
	ax_tr = pypl.gca()  ## Bottom Axis ##
	#cbs,cbp =  0.98, 0.02 #0.04, 0.1: left plts cb ocerlap middle fig
	ax_tr.set_ylabel( 'y Pixels', fontsize=axis_fsize) 
	ax_tr.set_xlabel( 'x Pixels', fontsize=axis_fsize) 
	#pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'})
	## ax0.set( ylabel='y Pixels')
	## ax0.set( xlabel='x Pixels')

	# ---- Intensity Pattern : ---- #
	if mask is not None:
		Ip_ma_shot = np.ma.masked_where(mask == 0, np.log10(IP_mean)) ## look at the log10##
	else: Ip_ma_shot = IP_mean
	im_ip =ax_tr.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
	## ADD CIRCLES AT MAC AND MIN Q ##
	cntr =np.asarray([(mask.shape[1]-1)/2.0, (mask.shape[0]-1)/2.0 ]) ## (X,Y)
	x_center, y_center = cntr[0], cntr[1]
	#q_ring_1 ,q_ring_2 = q_map[0,2], q_map[-1,2] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
	q_ring_1 ,q_ring_2 =qmin_pix, qmax_pix
	#from matplotlib.patches import Circle
	pypl.gca().add_patch(Circle( xy=(x_center,y_center), radius=q_ring_1 , ec = 'w', fc ='none', lw= 2,) )
	pypl.gca().add_patch(Circle( xy=(x_center,y_center), radius=q_ring_2 , ec = 'w', fc ='none', lw= 2,) )
	
	#########################################################
	cb_ip = pypl.colorbar(im_ip, ax=ax_tr,fraction=0.046)
	cb_ip.set_label(r' Photons (mean) ') #(r'Intensity ')
	#cb_ip.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	#cb_ip.update_ticks()
	#cmap.set_bad('grey',1.) ## default: (color='k', alpha=None) Set color to be used for masked values. ##
	#cmap.set_under('white',1.) ## Set color to be used for low out-of-range values. Requires norm.clip = False ##
	
	rad_title =pypl.title('Mean Intensity of %i Patterns (Log10): ' %(shot_count), fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
	rad_title.set_y(1.08) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	############################## [fig.] End ##############################
	fig_name = "Mean_Intensity_log10.%s" %(frmt)
	#pypl.show()
	#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	fig5.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Plot saved as %s " %fig_name
	del fig_name 	# clear up memory1
	gc.collect() 
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##

	##################################################################################################################
	##################################################################################################################
	#---- Fig.6 FFTn-odd||even of ACC Normalised with correlated Mask (! NOT with variance (phi=0)): ----------------#
	##################################################################################################################
	##################################################################################################################

	fig6 = pypl.figure('AC_FFT_series', figsize=(22,15))
	#sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count 
	#corrsum_m = np.divide(corrsum, corr_mask, out=None, where=corr_mask!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
	sig = corrsum_m/corr_count 
	sig  =np.nan_to_num(sig)
	sig_fft =  fftshift(fftn(sig))
	del sig
	gc.collect()

	sig_fft_sets = [ np.real(sig_fft), np.imag(sig_fft) ]
	#sig_fft_sets = [ np.real(sig_fft), np.imag(sig_fft), 
						# np.log10(np.real(sig_fft)), np.log10(np.imag(sig_fft))  ]
	titles = ['even', 'odd']
	#titles = ['even', 'odd', 'Log10 of even', 'Log10 of odd']
	axs = [fig6.add_subplot(121), fig6.add_subplot(122)]
	#axs = [fig6.add_subplot(221), fig6.add_subplot(222), fig6.add_subplot(223), fig6.add_subplot(224)]
	pypl.subplots_adjust(wspace=0.2, hspace=0.4, left=0.04, right=0.92) #, OK=left=0.01(but cb ERR)
	## --------------- SUBPLOTs -------------- ##
	for i in range(len(sig_fft_sets)):
		ax = axs[i]
		m = sig_fft_sets[i].mean() 	# MEAN
		s = sig_fft_sets[i].std() 	# standard Deeviation
		vmin = m-2*s #sig.max()#sig.min()#m-2*s
		vmax = m+2*s #sig.min()#sig.max()#m+2*s
		
		polar_axis = False# True 	## Plot with axis in polar coord and with polar labels, else pixels ##
		if polar_axis:
			im = ax.imshow( sig_fft_sets[i],
		                 extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
		                    vmin=vmin, vmax=vmax*frac, aspect='auto')
		else : im =ax.imshow(sig_fft_sets[i], 
						extent=[-sig_fft_sets[i].shape[1]/2,sig_fft_sets[i].shape[1]/2 , -sig_fft_sets[i].shape[0]/2, sig_fft_sets[i].shape[0]/2], 
						 vmin=vmin, vmax=vmax*frac, aspect='auto' )
		cb = pypl.colorbar(im, ax=ax, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)
		cb.set_label(r'Auto-correlation of Intensity  [a.u.]',  size=12) # normed intensity
		cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
		cb.update_ticks()
		pypl.draw()

		# if polar_axis:
		# 	# #---- Adjust the X-axis: ----  		##
		# 	#xtic = ax.get_xticks()  	#[nphi/4, nphi/2, 3*nphi/4]
		# 	phi=2*np.pi  ## extent of sigma in angu;ar direction ##
		# 	xtic = [phi/4, phi/2, 3*phi/4]
		# 	xlab =[ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
		# 	ax.set_xticks(xtic), ax.set_xticklabels(xlab)

		# 		# #---- Adjust the Y-axis: ----
		# 	nq = q_map.shape[0]-1#q_step 	# Number of q radial polar bins, where q_map is an array q_min-q_max
		# 	#print "q-map : ", q_map
		# 	q_bins = np.linspace(0, nq, num= ax.get_yticks().shape[0], dtype=int)  ## index of axis tick vector ##
		# 	#print "Dim q-bins: ", q_bins.shape[0], " max:", q_bins.max(), " min:", q_bins.min()
		# 	#print "q-bins array: ", q_bins, "/n Dim q-bins ", q_bins.shape
		# 	q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		# 	#print "q_label: ", q_label
		# 	ytic=ax.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		# 	#ylab= q_label
		# 	ax.set_yticklabels(q_label)
		# 	ax.set_yticks(ytic)

		# 	# #---- Label xy-Axis: ----
		# 	ax.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		# 	ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)

		FFT_ave_corr_title = ax.set_title("FFT %s-part with limits $\mu \pm 2\sigma$"%(titles[i]),  fontsize=sb_size)
		FFT_ave_corr_title.set_y(1.08) # 1.1)
	############################## [fig.6] End ##############################
	pypl.suptitle("FFT of Average of %d corrs [%s] \n (Normalized with Mask) with limits $\mu \pm 2\sigma$"%(corr_count, pttrn),  fontsize=sb_size)
	#pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)
	fig_name = "Auto-Corr_Normed_w_Mask_fftshift-FFT_even-odd_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#fig_name = "Auto-Corr_Normed_w_Mask_FFT_even-odd_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#pypl.show()
	fig6.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	del fig_name 	# clear up memory
	gc.collect() 
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##



	##################################################################################################################
	##################################################################################################################
	#------- Fig.7 FFTn{ACC}-rowvise (by Qs) of ACC Normalised with correlated Mask and variance: -------------------#
	##################################################################################################################
	##################################################################################################################

	fig7 = pypl.figure('AC_FFT_by_row', figsize=(22,15))
	## Fourier Coefficients: plot F-coefficients (q)  row by row {changes per row}, change  per row [0:th is the Radial Profile] 
	#corrsum_m = np.divide(corrsum, corr_mask, out=None, where=corr_mask!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
	sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig  =np.nan_to_num(sig)
	sig_fft =  fftn(sig)
	del sig
	gc.collect() 
	## /usr/local/lib/python2.7/site-packages/numpy/core/numeric.py:538: ComplexWarning: Casting complex values to real discards the imaginary part
	##		return array(a, dtype, copy=False, order=order)
	sig_fft_R = np.real(sig_fft)	## The Real part ##
	#sig_fft =  fftshift(fftn(sig))
	##

	# sig_fft[i,:]
	# for i in range(sig_fft.shape[1]):  pypl.plot (sig_fft[i,:], lw=2, label='Row #%i' %i)
	end_fcc=21
	cmap = pypl.get_cmap('jet')

	subplots_phi_q = True
	modulus = True #False#True	## plot the modulus = absolute value ##
	half_angle_int = True
	if subplots_phi_q:	ax_phi=fig7.add_subplot(121); ax_q=fig7.add_subplot(122); 

	if modulus:		## plot the Absolute values ##
		if subplots_phi_q:
			for i in range(end_fcc):  ax_phi.plot(np.abs(sig_fft_R[i,:]),c=cmap(float(i)/end_fcc), lw=2, label='Row #%i' %i)
			for i in range(end_fcc):  ax_q.plot(np.abs(sig_fft_R[:,i]),c=cmap(float(i)/end_fcc), lw=2, label='Column #%i' %i)
		else: 		## only plot the Absolute values  of Row-wise##
			for i in range(end_fcc):  pypl.plot(np.abs(sig_fft_R[i,:]),c=cmap(float(i)/end_fcc), lw=2, label='Row #%i' %i)
			#pypl.plot(np.abs(np.transpose(sig_fft_R[0:21,:]), lw=2) 
	else:	## or plot the true values ##
		if subplots_phi_q:
			for i in range(end_fcc):  ax_phi.plot(sig_fft_R[i,:], c=cmap(float(i)/end_fcc),lw=2, label='Row #%i' %i)
			for i in range(end_fcc):  ax_q.plot(sig_fft_R[:,i], c=cmap(float(i)/end_fcc),lw=2, label='Column #%i' %i)
		else:
			for i in range(end_fcc):  pypl.plot (sig_fft_R[i,:],c=cmap(float(i)/end_fcc), lw=2, label='Row #%i' %i)
	if subplots_phi_q: ax_phi.legend(); ax_q.legend()
	else:	pypl.legend()
	# if modulus:
	# 	#for i in range(end_fcc):  pypl.plot(np.abs(sig_fft_R[i,:]), c=cmap(float(i)/end_fcc),lw=2, label='Row #%i' %i)
	# 	for i in range(end_fcc):  pypl.plot(np.abs(sig_fft_R[i,:]), c=cmap(float(i)/end_fcc),lw=2, label='Row #%i' %i)
	# else:
	# 	for i in range(end_fcc):  pypl.plot(sig_fft_R[i,:],c=cmap(float(i)/end_fcc), lw=2, label='Row #%i' %i)
	# pypl.legend()

	#pypl.plot(np.transpose(sig_fft)[:,0:20],  lw=2)
	ylim = 2	#6.0 #0.7 #6.0  #10.0
	polar_axis = False #True 
	#ax = pypl.gca()
	if subplots_phi_q: 
		ax = ax_phi
		#ax_q.set_xlabel(r'1/q', fontsize = axis_fsize)
		ylim_q = sig_fft_R[:,1:end_fcc].max(axis=1).max()
		ylim_q_low = sig_fft_R[:,1:end_fcc].min(axis=1).min()
		print "y-limit for columns: ", ylim_q
		if modulus:
			FFT_ave_corr_title2 = ax_q.set_title("|FFT Real|",  fontsize=sb_size)
			ax_q.set_ylim(-0.5,ylim_q)
			#ax_q.set_xlim(right=50) ## very low at 100 ##
			ax_q.set_xlim(-2,25) ## very low at 100 ##
		else:	
			FFT_ave_corr_title2 = ax_q.set_title("FFT Real",  fontsize=sb_size)
			ax_q.set_ylim(ylim_q_low,ylim_q)
			ax_q.set_xlim(right=100)
	else:	
		ax = pypl.gca()
	if modulus:
		ax.set_ylim(0,ylim)
		FFT_ave_corr_title = ax.set_title("|FFT Real|",  fontsize=sb_size)
	else:	
		ax.set_ylim(-ylim,ylim)
		FFT_ave_corr_title = ax.set_title("FFT Real",  fontsize=sb_size)
	
	if polar_axis:
		#ax = pypl.gca()
		# #---- Adjust the X-axis: ----  		##
		xtic = ax.get_xticks()  	#[nphi/4, nphi/2, 3*nphi/4]
		phi= 360 #2*np.pi  ## extent of sigma in angu;ar direction ##
		xtic = [phi/4, phi/2, 3*phi/4]
		xlab =[ r'$2\pi$', r'$1/\pi$', r'$2/3\pi$'] #nphi_bins or phi_bins_5
		ax.set_xticks(xtic), ax.set_xticklabels(xlab)

		# #---- Label xy-Axis: ----
		ax.set_xlabel(r'$1/\phi$', fontsize = axis_fsize)
		#ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)
		#ax.set_xlabel(r'n:th coefficient', fontsize = axis_fsize)

	if half_angle_int:  ax.set_xlim(0,90) #ax.set_xlim(0,180)
		
	FFT_ave_corr_title.set_y(1.04) # mostly work 1.08, 1.1)
	FFT_ave_corr_title2.set_y(1.04) # mostly work 1.08, 1.1)
	############################## [fig.] End ##############################
	pypl.suptitle("FFT of Average of %d corrs [%s] \n (Normalized with Mask) with limits %.2f"%(corr_count, pttrn, ylim),  fontsize=sb_size)
	#pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)
	if modulus:	
		version = "_abs_lim-%.2f"%ylim
		if half_angle_int: version +="_half-half"
	else: 	
		version = "_lim-%.2f"%ylim
		if half_angle_int: version +="_half-half"
	if subplots_phi_q:	version += "_fft_q"
	fig_name = "FFT-byrow_(qx-%i_qi-%i_nphi-%i)_%s_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,version,frmt)
	fig7.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	del fig_name 	# clear up memory
	gc.collect() 
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##


	##################################################################################################################
	##################################################################################################################
	#---- Fig.8 FFT{ACC}(phi)-rowvise (by Qs) of ACC Normalised with correlated Mask and variance: ------------------#
	
	#---- Fig.9 FFT{ACC}(phi=0)-rowvise (by Qs) of ACC Normalised with correlated Mask and variance: ----------------#
	##################################################################################################################
	##################################################################################################################

	sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig  =np.nan_to_num(sig)	## (Qs, phis) = (200, 360) ##
	print "\n Dim of <ACC> normed with mask: ", sig.shape
	sig_Aft =  fft(sig, axis =1)
	print "\n Dim of Angular FFT: ", sig_Aft.shape # = (200, 360) ##
	del sig
	gc.collect()

	## ---- Phi even-odd = [1,20]: ---- ##
	fig8 = pypl.figure('AC_FFT_by_row_Nvar', figsize=(22,8))
	fig_name = "Nrmd-var_FFT-angle-"
	plot_even_odd(fig=fig8, data=sig_Aft, norm_name='and variance', fig_name=fig_name)
	
	## ---- Phi = [0]: ---- ##
	fig9 = pypl.figure('AC_FFT_0_Nvar', figsize=(22,12))#(30,12)) ## width, height in inches ##
	fig_name = 'Nrmd-var'
	plot_radial_profile(fig=fig9, data=sig_Aft, norm_name='and var', fig_name=fig_name)

	#del sig_Aft
	#gc.collect()


	##################################################################################################################
	##################################################################################################################
	#------- Fig.10 FFT{ACC}(phi)-rowvise (by Qs) of ACC Normalised with correlated Mask and Qs: --------------------#
	
	#------- Fig.11 FFT{ACC}(phi=0)-rowvise (by Qs) of ACC Normalised with correlated Mask and Qs: ------------------#
	##################################################################################################################
	##################################################################################################################

	#sig = np.array([  corrsum[i,:]/qs_mean[i]    for i in range(corrsum.shape[0]) ])
	sig_q = corrsum_m/qs_mean[:,None]/corr_count
	sig  =np.nan_to_num(sig_q)	## (Qs, phis) = (200, 360) ##
	print "\n Dim of <ACC> normed with mask: ", sig.shape
	sig_Aft =  fft(sig, axis =1)
	print "\n Dim of Angular FFT: ", sig_Aft.shape # = (200, 360) ##
	del sig, sig_q
	gc.collect()

	

	## ---- Phi even-odd = [1,20]: ---- ##
	fig10 = pypl.figure('AC_FFT_by_row_Nq', figsize=(22,8))
	fig_name = "Nrmd-qs_FFT-angle-"
	plot_even_odd(fig=fig10, data=sig_Aft, norm_name='and Q', fig_name=fig_name)
	
	## ---- Phi = [0]: ---- ##
	fig11 = pypl.figure('AC_FFT_0_Nq', figsize=(22,12))#(30,12)), alt try (22,14)
	fig_name = 'Nrmd-q'
	plot_radial_profile(fig=fig11, data=sig_Aft, norm_name='and Q', fig_name=fig_name)

	#del sig_Aft
	#gc.collect()

	##################################################################################################################
	##################################################################################################################
	#------ Fig.12 FFT{ACC}(phi)-rowvise (by Qs) of ACC Normalised with correlated Mask only: ----------------------#
	
	#------ Fig.13 FFT{ACC}(phi=0)-rowvise (by Qs) of ACC Normalised with correlated Mask only: ---------------------#
	##################################################################################################################
	##################################################################################################################
 
	sig = corrsum_m/corr_count
	sig  =np.nan_to_num(sig)	## (Qs, phis) = (200, 360) ##
	print "\n Dim of <ACC> normed with mask: ", sig.shape
	sig_Aft =  fft(sig, axis =1)
	print "\n Dim of Angular FFT: ", sig_Aft.shape # = (200, 360) ##
	del sig
	gc.collect()

	## ---- Phi even-odd = [1,20]: ---- ##
	fig12 = pypl.figure('AC_FFT_by_row_N', figsize=(22,8))
	fig_name = "FFT-angle-"
	plot_even_odd(fig=fig12, data=sig_Aft,norm_name='', fig_name=fig_name)

	## ---- Phi = [0]: ---- ##
	fig13 = pypl.figure('AC_FFT_0_N', figsize=(22,12))#(30,12)), (22,10) if not 2 rows in title
	fig_name = ''
	plot_radial_profile(fig=fig13, data=sig_Aft, norm_name='', fig_name=fig_name)
	#del sig_Aft
	#gc.collect()

	##################################################################################################################
	##################################################################################################################
	#----------- Fig.14 FFT{CC}(phi=0)-rowvise (by Qs) of CC Normalised with correlated Mask only : ----------------#
	
	#----------- Fig.15 of CC Normalised with correlated Mask only & Mask and Variance: --------------------------#
	##################################################################################################################
	##################################################################################################################
	
	## plot/ analyse ' cross_corrsum (Nq,Nq,Nphi), mask_crosscor (1,Nq,Nphi),

	sig_cc = cross_sum_m/corr_count 
	sig_cc  =np.nan_to_num(sig_cc)	## (Qs, phis) = (200, 360) ##
	print "\n Dim of <CC> normed with mask: ", sig_cc.shape
	## Fourier Coefficients: plot F-coefficients (q)  row by row {changes per row}, change  per row [0:th is the Radial Profile] 
	sig_cc_Aft =  fft(sig_cc, axis =-1)	## the last axis is the angles ##
	del sig_cc
	gc.collect()
	print "\n Dim of C-Angular FFT: ", sig_cc_Aft.shape #  ##


	fig14 = pypl.figure('AC_FFT_0_N', figsize=(22,12))#(30,12)), (22,10) if not 2 rows in title
	fig_name = 'CC'
	plot_radial_profile(fig=fig14, data=sig_cc_Aft, norm_name='', fig_name=fig_name)

	fig15 = pypl.figure('CC_FFT_by_row_N', figsize=(22,8))#(30,12))
	fig_name = "CC-FFT-angle-"
	plot_even_odd(fig=fig15 , data=sig_cc_Aft, norm_name='',  fig_name=fig_name)
	#del sig_cc_Aft
	#gc.collect()

	##################################################################################################################
	##################################################################################################################
	#------------------ Fig.16 of CC Normalised with correlated  Mask and Variance: ---------------------------------#
	##################################################################################################################
	##################################################################################################################
	
	fig16 = pypl.figure('CC_FFT_by_row_Nvar', figsize=(22,8))#(30,12))
	sig_cc_var = cross_sum_m/cross_sum_m[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig_cc_var  =np.nan_to_num(sig_cc_var)	## (Qs, phis) = (200, 360) ##
	print "\n Dim of <CC> normed with mask: ", sig_cc_var.shape
	## Fourier Coefficients: plot F-coefficients (q)  row by row {changes per row}, change  per row [0:th is the Radial Profile] 
	sig_cc_var_Aft =  fft(sig_cc_var, axis =-1)	## the last axis is the angles ##	#print "\n Dim of C-Angular FFT: ", sig_cc_Aft.shape # = ( ##
	del sig_cc_var
	gc.collect()
	fig_name = "CC-Nrmd-var_FFT-angle-"
	plot_even_odd(fig=fig16, data=sig_cc_var_Aft, norm_name='and variance', fig_name=fig_name)
	#del sig_cc_var_Aft
	#gc.collect()


	##################################################################################################################
	#---------------- Fig.17 COSINUS(psi) of ACC Normalised with correlated Mask & variance: -------------------------#
	##################################################################################################################
	##################################################################################################################
	fig17 = pypl.figure('cos-psi', figsize=(14,8)) ## in e.g. plt.figure(figsize=(12,6))
	cmap = pypl.get_cmap('jet')
	include_mean = True
	plot_multiple_pixels = False #True
	## Fourier Coefficients: plot F-coefficients (q)  row by row {changes per row}, change  per row [0:th is the Radial Profile] 
	sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count #normed with variance
	sig  =np.nan_to_num(sig)

	## ----- For scirpt with parser arguments ----- ##
	#parser.add_argument('-bins', dest='bins',
	#	type=float,default=None,
	#	help='Number of radial bins to average over (should be odd number)')
	# parser.add_argument('-Q', dest='Q',type=float, default=None,
	#     help='Which reciprocal space coordinate to look at')
	# parser.add_argument('-R', dest='R', type=float,default=None,
	#     help='Which radial pixel to look at')
	#if Q is not None:
	#	r_pixel = np.argmin( np.abs( qs - Q) ) ## "Returns the indices of the minimum values along an axis"
	#else:
	#	assert( R is not None)
	#	r_pixel = np.argmin( np.abs( radii-R) )
	#	Q = qs[idx]

	## Experimental Parameters are included in v.5: dtc_dist_m,wl_A,ps_m,be_eV  ##
	th = np.arctan(radii*ps_m/dtc_dist_m)*0.5 	## th(radii) ##
	qs =4*np.pi*np.sin(th)/wl_A 			## qs(th) ##
	phis = np.arange( sig.shape[1] )*2*np.pi/sig.shape[1] 		## sig.shape[1] = the number of angles ##
	#print"\n Dim of 'theta': ", th.shape
	#print"\n Dim of 'qs': ", qs.shape

	#rs = np.linspace(r_pixel-50,r_pixel+50, num=100)
	rs = np.arange(r_pixel-50,r_pixel+51, step=10, dtype=int)
	print "Dim of radial selection of pixels 'rs': ", rs.shape
	rs_length = rs.shape[0]
	i=0
	Q_idx=[]
	if plot_multiple_pixels:
		for R in rs:
			idx = np.argmin( np.abs(   radii-R  ) )
			cos_psi = np.cos( phis ) * np.cos( th[idx] )**2 + np.sin( th[idx] )**2
			## c=cmap(float(i)/end_fcc),
			
			pypl.plot(cos_psi, sig[idx],  c=cmap(float(i)/rs_length),lw= 2, label='pixel=%d'%radii[idx])
			i +=1 	## Counter for the colours ##
			Q_idx.append(idx) 	## the indices plotted, stored for printing Qs later ##
			#pypl.title("Correlation at Q=%.3f $\AA^{-1}$ (R=%d pixels)" % (Q, R), fontsize=18)
			pypl.title("Correlation at Q=%.3f - %.3f $\AA^{-1}$ (R=%d - %d pixels)" % ( qs[Q_idx[0]],qs[Q_idx[-1]],rs[0],rs[-1]), fontsize=18)
			#pypl.title("Correlation at Q=%.3f - %.3f $\AA^{-1}$ (R=%d - %d pixels)" % ( q_map[Q_idx[0],1],q_map[Q_idx[-1],1],q_map[Q_idx[0],2],q_map[Q_idx[-1],2]), fontsize=18)
		plt_bins='multi'
	else:
		idx = np.argmin( np.abs(   radii-r_pixel  ) )
		cos_psi = np.cos( phis ) * np.cos( th[idx] )**2 + np.sin( th[idx] )**2
		pypl.plot(cos_psi, sig[idx], 'b',lw= 2, label='pixel=%d'%r_pixel)
		pypl.title("Correlation at Q=%.3f  $\AA^{-1}$ (R=%d  pixels)" % ( qs[idx],r_pixel), fontsize=18)
		plt_bins='single'
	if include_mean:
		bins=rs.shape[0]
		cos_psi_r_pixel = np.cos( phis ) * np.cos( th[r_pixel] )**2 + np.sin( th[r_pixel] )**2
		pypl.plot( cos_psi, sig[r_pixel-np.int(bins/2):r_pixel+np.int(bins/2)+1].mean(axis=0), 'k', linestyle='dashed', lw= 3, label='mean of pixel %d - %d'%(rs[0],rs[-1]) )
	#pypl.xlim(right=0.994)
	#pypl.ylim(top=0.0057)
	#pypl.ylim(sig[idx].min()-0.01*sig[idx].min(),sig[idx].mean()+0.01*sig[idx].mean())
	std= sig[idx].std()
	mu= sig[idx].mean()
	#pypl.ylim(mu-0.25*std, mu+0.01*std)#(mu-0.25*std, mu+0.25*std)#(mu-2*std, mu+2*std)
	pypl.ylim(mu-0.3*std, mu+0.01*std)#(mu-0.25*std, mu+0.25*std)#(mu-2*std, mu+2*std)


	pypl.xlabel(r"$\cos \,\psi$", fontsize=18)
	#pypl.gca().tick_params(labelsize=15, length=9)
	pypl.gca().tick_params(labelsize=tick_fsize, length=9)
	pypl.legend(loc=0) ## 2= úpper left, 0='best0
	pypl.show()

	############################## [fig.17] End ##############################
	#pypl.suptitle("$\cos(\psi)$ around R=%d of Average of %d corrs [%s] \n (Normalized with Mask) with limits %.2f"%(R, corr_count, pttrn),  fontsize=sb_size)
	# #pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)
	
	fig_name = "cos-psi_R-%d__(qx-%i_qi-%i_nphi-%i)_%s.%s" %(r_pixel,plt_bins,qmax_pix,qmin_pix, nphi,pttrn,frmt)
	fig17.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	del fig_name, sig 	# clear up memory
	gc.collect() 
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##
# -------------------------------------- 


def subplot_acc(corrsum,corr_count, q_map, nphi, pttrn,corr_mask, IP_mean, shot_count, out_fname, mask=None ): ## tot_shot_sum
	"""
	Read in ACC data from multiple files in '' list of filenames. 
	Plot the average from all read-in files together in a subplot.

	In:
	corrsum 			The total sum of all the auto-correlations
	corr_count 			 The total number of corrlations calculated in the 'corrsum',
							e.g. the total number of differeneses that was used to calculate the correlations.
	q_map 				The momentum transfer vaules, Qs', in pixels for which the correlations were calculated.
	nphi 				The angles for which the correlations were calcucated (e.g. 180 or 360).
	pttrn 				A string informing which differences were used in the correlation calcutaion, such as 
							if the correlatsion were calculated from pairwise diffreence or N-1 diffrens for N shots.
	corr_mask			The Auto-correlation of the mask.
	IP_mean 			The Mean of all Intensity Patterns (or Amplitude Patterns).
	mask 				The Detector Mask (for plotting the intensity Pattern wth mask)
	shot_count 			The total number of shots summed in 'IP_mean' and loaded.
	out_fname 			The pathname and prefix of the filename of the plot.

	Out:				Plot of the auto-correlation.
	"""
	print "\n Plotting some Subplots..."
	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14		## Fontsize of axis labels ##
	sb_size = 16 #18 	## Fontsize of Sub-titles ##
	sp_size = 18#20  	## Fontsize of Super-titles ##

	cb_shrink, cb_padd = 1.0, 0.2  ## Colorbar padding works fine for horizontal orientation ##
	cbs,cbp =  0.98, 0.02 #0.04, 0.1: left plts cb ocerlap middle fig
	padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181

	frmt = "eps" 	## Format used to save plots ##
	## columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2] ## first and last values in cloumn 2 = q [pixels] ##
	#qs_c = qs_c_map[:,1]  ## index, Q[inv Angstrom], Q [pixels]##
	#r_bins = qs_c_map[:,2]  ## index, Q[inv Angstrom], Q [pixels]##
	#rad_cntr = int((rad_pro_mean.shape[0]-1)/2) 	## Center index of radial profile, OBx -1 since index start at 0 not 1 ##
	#print "\n Dim of mean(Radial Profile): ", rad_pro_mean.shape, " , center pixel: ", rad_cntr ## (1800,)  ,  899 ##

	
	fig5 = pypl.figure('IP-ACC-MaskACC', figsize=(18,18)) ## width, height in inches ##
	#fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(22,15))
	## subplot , constrained_layout=True 
	ax_tr=pypl.subplot(221)		## Mean of the Intensity Patterns (shots) [top-left] ##
	ax_tl=pypl.subplot(222) 	## ACC of Mask [top-right] ##
	ax_b =pypl.subplot(212) 		## Mean of the  Auto-Correlations Normed with Mask [bottom] ##
	##################################################################################################################
	##################################################################################################################
	#------------------------- 5.A) Plotting  Mean Intensiyt Pattern [ax_tr]-------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	color_map ='jet' #'viridis'
	ax_tr.set_ylabel( 'y Pixels', fontsize=axis_fsize) 
	ax_tr.set_xlabel( 'x Pixels', fontsize=axis_fsize) 
	#pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'})
	## ax0.set( ylabel='y Pixels')
	## ax0.set( xlabel='x Pixels')
	# ---- Intensity Pattern : ---- #
	if mask is not None:
		Ip_ma_shot = np.ma.masked_where(mask == 0, IP_mean) ## Check the # photons ##
	else: Ip_ma_shot = IP_mean
	#im_ip = pypl.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
	im_ip =ax_tr.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
	cb_ip = pypl.colorbar(im_ip, ax=ax_tr,fraction=0.046)
	cb_ip.set_label(r' Photons (mean) ') #(r'Intensity ')
	#cb_ip.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	#cb_ip.update_ticks()
	#cmap.set_bad('grey',1.) ## default: (color='k', alpha=None) Set color to be used for masked values. ##
	#cmap.set_under('white',1.) ## Set color to be used for low out-of-range values. Requires norm.clip = False ##
	
	rad_title =ax_tr.set_title('Mean Intensity of %i Pattersn: ' %(shot_count), fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
	rad_title.set_y(1.08) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)

	##################################################################################################################
	##################################################################################################################
	#---------------------------------- 5.B) ACC of the Mask [ax_tl] --------------------------------------------------#
	##################################################################################################################
	##################################################################################################################

	sig_m = corr_mask/corr_mask[:,0][:,None]#/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig_m  =np.nan_to_num(sig_m)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...

	#padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181
	m = sig_m[:,padC:-padC].mean() 	# MEAN
	s = sig_m[:,padC:-padC].std() 	# standard Deeviation
	vmin = m-2*s
	vmax = m+2*s
	
	#ax = pypl.gca()
	polar_axis = True 	## Plot with axis in polar coord and with polar labels, else pixels ##
	if polar_axis:
		im_m = ax_tl.imshow( sig_m,
                     extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
                        vmin=vmin, vmax=vmax, aspect='auto')#equal')
	else : im_m =ax_tl.imshow(sig, vmin=vmin, vmax=vmax, aspect='equal' )
	#cb = pypl.colorbar(im, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)
	cb = pypl.colorbar(im_m, ax=ax_tl, shrink=cbs, pad= cbp) #fraction=0.046)
	cb.set_label(r'Auto-correlation of Intensity Difference [a.u.]',  size=12) # normed intensity
	cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	cb.update_ticks()

	if polar_axis:
		# #---- Adjust the X-axis: ----  		##
		phi=2*np.pi  ## extent of sigma in angu;ar direction ##
		xtic = [phi/4, phi/2, 3*phi/4]
		xlab =[ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
		ax_tl.set_xticks(xtic), ax_tl.set_xticklabels(xlab)

		# #---- Adjust the Y-axis: ----
		nq = q_map.shape[0]-1#q_step 	# Number of q radial polar bins, where q_map is an array q_min-q_max
		#print "q-map : ", q_map
		q_bins = np.linspace(0, nq, num= ax_tl.get_yticks().shape[0], dtype=int)  ## index of axis tick vector ##
		#print "Dim q-bins: ", q_bins.shape[0], " max:", q_bins.max(), " min:", q_bins.min()
		#print "q-bins array: ", q_bins, "/n Dim q-bins ", q_bins.shape
		q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		#print "q_label: ", q_label

		ytic=ax_tl.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		#ylab= q_label
		ax_tl.set_yticklabels(q_label)
		ax_tl.set_yticks(ytic)

		# #---- Label xy-Axis: ----
		ax_tl.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		ax_tl.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)

	ACC_mask_title = ax_tl.set_title(r"ACC of Mask with limits $\mu \pm 2\sigma$",  fontsize=sb_size)
	ACC_mask_title.set_y(1.08) # 1.1)

	##################################################################################################################
	##################################################################################################################
	#-------------------------- 5.C) Plotting  ACC Normed with Mask [ax_b]---------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	## ---- Normalize (avoide division with 0) ----- ##
	corrsum_m = np.divide(corrsum, corr_mask, out=None, where=corr_mask!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
	sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape 

	#padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181
	m = sig[:,padC:-padC].mean() 	# MEAN
	s = sig[:,padC:-padC].std() 	# standard Deeviation
	vmin = m-2*s
	vmax = m+2*s

	polar_axis = True 	## Plot with axis in polar coord and with polar labels, else pixels ##
	if polar_axis:
		im = ax_b.imshow( sig,
                     extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
                        vmin=vmin, vmax=vmax, aspect='auto')
	else : im =ax_b.imshow(sig, vmin=vmin, vmax=vmax, aspect='auto' )
	cb = pypl.colorbar(im, ax=ax_b, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)
	cb.set_label(r'Auto-correlation of Intensity Difference [a.u.]',  size=12) # normed intensity
	cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	cb.update_ticks()

	if polar_axis:
		# #---- Adjust the X-axis: ----  		##
		#xtic = ax.get_xticks()  	#[nphi/4, nphi/2, 3*nphi/4]
		phi=2*np.pi  ## extent of sigma in angu;ar direction ##
		xtic = [phi/4, phi/2, 3*phi/4]
		xlab =[ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
		ax_b.set_xticks(xtic), ax_b.set_xticklabels(xlab)

		# #---- Adjust the Y-axis: ----
		nq = q_map.shape[0]-1#q_step 	# Number of q radial polar bins, where q_map is an array q_min-q_max
		#print "q-map : ", q_map
		q_bins = np.linspace(0, nq, num= ax_b.get_yticks().shape[0], dtype=int)  ## index of axis tick vector ##
		#print "Dim q-bins: ", q_bins.shape[0], " max:", q_bins.max(), " min:", q_bins.min()
		#print "q-bins array: ", q_bins, "/n Dim q-bins ", q_bins.shape
		q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		#print "q_label: ", q_label

		ytic=ax_b.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		ylab= q_label
		ax_b.set_yticklabels(q_label)
		ax_b.set_yticks(ytic)

		# #---- Label xy-Axis: ----
		ax_b.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		ax_b.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)

	Norm_ave_corr_title = ax_b.set_title(r"Average of %d corrs [%s] (Normalized with Mask) with limits $\mu \pm 2\sigma$"%(corr_count, pttrn),  fontsize=sb_size)
	Norm_ave_corr_title.set_y(1.08) # 1.1)
	############################## [fig.ACC] End ##############################
	
	#pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)  # hspace=0.2,
	#pypl.subplots_adjust(wspace=0.3, hspace=0.4, left=0.08, right=0.95, top=0.85, bottom=0.15)  # hspace=0.2,
	pypl.subplots_adjust(wspace=0.3, hspace=0.4, left=0.125, right=0.95, top=0.9, bottom=0.15)  # hspace=0.2,
	
	#pypl.suptitle("%s_%s_(%s-noise_%s)_[qx-%i_qi-%i]_w_Mask_%s" %(name,pdb,noisy,n_spread,qrange_pix[-1],qrange_pix[0],pttrn), fontsize=sp_size)  # y=1.08, 1.0=too low(overlap radpro title)

	fig_name = "SUBPLOT_IP-ACC-MaskACC_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	
	#pypl.show()
	#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	fig5.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	print "\n Subplot saved in %s " %(out_fname+fig_name)
	del fig_name 	# clear up memory1
	gc.collect() 
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##

# -------------------------------------- 


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
#------------------------------------ Loki XCCA ['calculate']: --------------------------------------------------#
##################################################################################################################
#if args.calculate:
def load_and_calculate_from_cxi(args):
	'''
	Load simulation data, file-by-file, from folder 'dir_name' and specify 
	a directory and name for the file containg the calculatsions.
	'''
	fnames = [ os.path.join(args.dir_name, f) for f in os.listdir(args.dir_name)
		 if f.endswith('.cxi') ]
	if not fnames:
		print"\n No filenames for directory %s"%(str(args.dir_name))
	fnames = sorted(fnames, 
		key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0])) 	## For sorting simulated-file 0-90 ##
	assert (len(fnames)>args.sim_n),("No such File Exists! Simulation number must be less than the number of simulations.")
	fname = fnames[args.sim_n] ## Choose only this pdb-files simulation ##

	# ----- Parameters unique to Simulation(e.g. 'noisefree_Beam-NarrInt_84-105_6M90_ed_(none-sprd0)_#5.cxi'): ---- ##
	pdb = fname.split('_ed')[0].split('84-')[-1][4:] 	## '...84-105_6M90_ed...'=> '6M90' ##
	run=fname.split('84-')[-1][0:3] 		## 3rd index excluded ## 119 or 105
	noisy = fname.split('_ed_(')[-1].split('-sprd')[0]
	#n_spread = fnames[0].split('-sprd')[-1].split(')_')[0] ## for files that ends with '_#N.cxi' ##
	n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
	name = fname.split('_84-')[0].split('/')[-1]

	## ----- Fetch Simulation Parameters: ---- ##
	photon_e_eV, wl_A, ps, dtc_dist, N, shots= load_cxi_data(data_file= fname, get_parameters= True, Amplitudes=Ampl_image)

	# ---- Save g-Mapping (the q- values [qmin_pix to qmin_pix]): ----
	pix2invang = lambda qpix : np.sin(np.arctan(qpix*(ps*1E-6)/dtc_dist )*0.5)*4*np.pi/wl_A
	# invang2pix = lambda qia : np.tan(2*np.arcsin(qia*wl_A/4/pi))*dtc_dist/(ps*1E-6)
	qrange_pix = np.arange( args.q_range[0], args.q_range[1]+1)
	## ---- Q-Mapping with  0: indices, 1: r [inv Angstrom], 2: q[pixels] ---- ##
	q_mapping = np.array( [ [ind, pix2invang(q), q] for ind,q in enumerate( qrange_pix)]) 
	
	# ---- Generate a Storage File: ---- ##
	#prefix = 'Parts-ACC_%s_%s_%i-shots' %(name,pdb,N) ## Partial Result ##
	prefix = 'Parts-ACC_%s_84-%s_%s_(%s-sprd%s)' %(name,run,pdb,noisy,n_spread)
	outdir = create_out_dir(out_dir = args.outpath, sim_res_dir=args.dir_name,name=name,pdb=pdb,noisy=noisy)
	out_fname = os.path.join( outdir, prefix) 
	# ---- Generate a Storage File for Data/Parameters: ----
	out_hdf = h5py.File( out_fname + '_%s.hdf5' %(pttrn), 'w')    # a: append, w:Write


	print "\n Data Analysis with LOKI.\n"
	set_ACC_input(args.exp_set)
	#ACC_wo_MASK = not bool(args.w_MASK)  			## if True: Calculate tha ACC without the Mask ##
	#if not ACC_wo_MASK: 	shots *= mask_better ## Add the MASK, else if ÁCC_wo_MASK =True do not  implement the mask ##
	if bool(args.w_MASK):  	shots *= mask_better ## Add the MASK ##
	calc_ac_and_cc(img=shots, cntr=cntr, q_map= q_mapping, mask = mask_better, data_hdf=out_hdf, 
				beam_eV=photon_e_eV, wavelength_A=wl_A, pixel_size_m=ps*1E-6, detector_distance_m=dtc_dist)

##################################################################################################################
#------------------------------ Mean and Plot ['plots']: --------------------------------------------------------#
##################################################################################################################
#else:
def load_and_plot_from_hdf5(args):
	'''
	Load calculated data, file-by-file, from folder 'outpath' and specify 
	a directory and name for the file containg the total, mean and plots.
	'''
	fnames = [ os.path.join(args.outpath, f) for f in os.listdir(args.outpath)
		 if f.startswith('Parts-ACC_') and f.endswith('.hdf5') ]
	if not fnames:
		print"\n No filenames for directory %s"%(str(args.outpath))
	fnames = sorted(fnames, 
		key=lambda x: int(x.split('84-')[-1][6:].split('_(')[0] )) 	## For sorting simulated-file 0-90 ##
	#fnames = sorted(fnames, key=lambda x: int(x.split('84-')[-1][4:].split('_ed')[0][0] )) 	## For sorting conc 4-6 M ##

	# ######################  !! OBS !!  ###################### 
	# ## OBS! Memory limit on Cluster: max 60 GB, file size approx 849 Mb => only load 60 of 91 files
	# fnames = fnames[:60]
	# ######################  !! OBS !!  ###################### 

	# ----- Parameters unique to Simulation: ---- ##
	cncntr_start = fnames[0].split('84-')[-1][4:].split('_(')[0] 	## '...84-105_6M90_ed...'=> '6M90' ##
	## Parts-ACC_Pnoise_BeamNarrInt_84-119_4M8_(poisson-sprd0)_Int.hdf5
	if len(fnames)>1:
		cncntr_end = fnames[-1].split('84-')[-1][4:].split('_(')[0]
		pdb=cncntr_start +'-'+cncntr_end
	else :	pdb = cncntr_start
	run=fnames[0].split('84-')[-1][0:3] 		## 3rd index excluded ##
	#noisy = fnames[0].split('_ed_(')[-1].split('-sprd')[0]
	noisy = fnames[0].split('84-')[-1].split('_(')[-1].split('-sprd')[0] ## if not '4M90_ed' but '4M90' in name
	#n_spread = fnames[0].split('-sprd')[-1].split(')_')[0] ## for files that ends with '_#N.cxi' ##
	n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
	name = fnames[0].split('_84-')[0].split('Parts-ACC_')[-1]
	pttrn
	## /.../noisefree_Beam-NarrInt_84-105_6M90_ed_(none-sprd0)_#5.cxi
	# ---- Generate a Storage File: ---- ##
	prefix = '%s_%s_(%s-sprd%s)_' %(name,pdb,noisy,n_spread) ## Final Result ##
	new_folder = args.outpath + '/w_mean_ip/'
	outdir = create_out_dir(out_dir = new_folder,name=name,pdb=pdb,noisy=noisy)
	out_fname = os.path.join( outdir, prefix) 
	#out_hdf = h5py.File( out_fname + 'tot_Auto-corr_%s.hdf5' %(pttrn), 'a')    # a: append, w: write
	out_hdf = h5py.File( out_fname + 'tot_Auto-corr_%s.hdf5' %(pttrn), 'w')    # a: append, w: write
	print "\n Data Analysis with LOKI."
	## ---- Read in ACC from Files and Plot ----  ##
	#read_and_plot_acc(fnames, out_hdf, out_fname) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]
	read_and_plot_acc(fnames, out_hdf, out_fname, mask_better) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]

##################################################################################################################
##################################################################################################################

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Auto-Correlations.")

# parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
#       help="The Location of the directory with the cxi-files (the input).")

parser.add_argument('-o', '--outpath', dest='outpath', default='this_dir', type=str, help="Path for output, Plots and Data-files")

subparsers = parser.add_subparsers()#title='calculate', help='commands for calculations help')

## ---- For Calculating the ACC of sile #X and storing in separat hdf5-file: -----##
subparsers_calc = subparsers.add_parser('calculate', help='commands for calculations ')
subparsers_calc.add_argument('-d', '--dir-name', dest='dir_name', default='this_dir', type=str,
      help="The Location of the directory with the cxi-files (the input).")
subparsers_calc.add_argument('-s', '--simulation-number', dest='sim_n', default=None, type=int, 
      help="The number of the pdb-file, which simulation is to be loaded from 0 to 90, e.g. '20' for the file <name>_4M20_<properties>l.cxi")
#subparsers_e = subparsers.add_parser('e', help='a choices')
subparsers_calc.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
      help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)
subparsers_calc.add_argument('-e', '--exposure', dest='exp_set', default='pair', type=str, choices=['pair', 'diffs', 'all', 'all-dc'],
      help="Select how to auto-correalte the data: 'pair' (pair-wise difference between shots),'diffs', 'all' (all shot without difference), 'all-dc' (all shot without difference, difference done by correlation script).")
# subparsers_calc.add_argument('-m', '--masked', dest='w_MASK', default=True, type=lambda s: (str(s).lower() in ['false', 'f', 'no', 'n', '0']),
#       help="Select if the mask is included in the auto-correalte calculations.")
# subparsers_calc.add_argument('-m', '--unmasked', dest='w_MASK', action='store_false',
#       help="Select if the mask is included in the auto-correalte calculations.")
subparsers_calc.add_argument('-m', dest='w_MASK', action='store_true')
subparsers_calc.add_argument('--no-mask', dest='w_MASK', action='store_false')
subparsers_calc.set_defaults(w_MASK=True)
subparsers_calc.set_defaults(func=load_and_calculate_from_cxi)

## ---- For Loading Nx# files, Summarise and Plotting the total result: -----##
subparsers_plt = subparsers.add_parser('plots', help='commands for plotting ')
subparsers_plt.add_argument('-p', '--plot', dest='plt_set', default='single', type=str, choices=['single', 'subplot', 'all'],
      help="Select which plots to execute and save: 'single' each plot separately,'subplot' only a sublpot, 'all' both single plots and a subplot.")
subparsers_plt.set_defaults(func=load_and_plot_from_hdf5)

args = parser.parse_args()
args.func(args) ## if .set_defaults(func=) SUBPARSERS ##
## ---------------- ARGPARSE  END ----------------- ##