#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import CXI-files from simulations with Condor (v1.0) and Auto-Correlate Data
# and plot with LOKI
# cxi-file/s located in the same folder, result saved  in subfolder "/simulation_results_N_X"
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#   Patterson_Image '..._patterson_image_...' are from FFTshift-Fast Fourier transforms-FFTshift
#           of Intensity Patterns =  AutoCorrelated image (can be used as initial guess for phae retrieval)
# 2019-03-25 v3 @ Caroline Dahlqvist cldah@kth.se
#			AutoCCA_84-run_v3.py
#			v3 with -s input: choose which simulated file to process
#				without -s: process hdf5-files with stored ACC-cacluations
#				 select if calculate with mask, select if ACC of pairs of differences, differences or all.
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
import matplotlib.cm as cm
import matplotlib.pyplot as pypl
from matplotlib import ticker
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
cart_diff, pol_all, pair_diff = False,False,False ## only one should be true !!

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
		if (name is not None) and (pdb is not None) and (noisy is not None):
			outdir = sim_res_dir+'/%s_%s_%s/' %(name,pdb,noisy)
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
#def calc_acc(images,  cntr, qmax_pix, qmin_pix, q_map, mask, data_hdf):
def calc_acc(img,  cntr, q_map, mask, data_hdf):
	"""
	Caclulate the Auto-Correlation for the imported 'data' with LOKI.

	In:
	================
	img 			the shots loaded from the specific CXI-file. dim (N,Y,X)
	cntr 			the beam center (Y,X)
	q_map 			the Qs stored in inv Angstrom and in pixels. dim (k,3),
						with  column 0 = index, column 1 = q [inv A], column 2: q [pixels]
	mask 			Pixel Maske with dead or none pixels masked out. no pixel = 0
						pixel = 1. dim (Y,X) must be the same X,Y as in images.
	data_hdf 		the hdf5-file handler for storing the data.
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
	
	if cart_diff:
		# ---- Calculate the Difference in Intensities and store in a List: ----
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
		del  exposure_diffs_cart 
		gc.collect() ## Free up Memory: ##
	#elif pol_diff:
	elif pol_all:
		## ---- Calc from polar Images: ---- ##
		# ---- Generate Polar Image/Images (N Diffracton Patterns) : ----
		print "\n Starting to Calculate the Polar Images..."
		polar_imgs = np.array( [ polar_mask* Interp.nearest(img[i]) for i in range( img.shape[0]) ] )
		## ---- Normalize - with function: ---- ##
		## alt try: polar_imgs_norm = norm_data(polar_imgs)
		#polar_imgs_norm = polar_imgs
		#exposure_diffs_pol =polar_imgs_norm[:-1]-polar_imgs_norm[1:]	# Polar Imgs
		exposures =  polar_imgs # polar_imgs_norm
		del img, polar_imgs #,polar_imgs_norm
		gc.collect() ## Free up Memory: ##
		#exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)
		#pttrn = "polar-diff"
		pttrn = "polar-all"
		# if 'exposure_pol' not in data_hdf.keys(): data_hdf.create_dataset( 'exposure_pol', data = np.asarray(exposure_diffs))
		# else: 
		# 	del data_hdf['exposure_pol']
		# 	dset = data_hdf.create_dataset('exposure_pol', data=np.asarray(exposure_diffs))
	elif pair_diff :
		## ---- Calc difference of images from Pairs of Images and convert to polar diff-pairs-images: ---- ##
		exp_diff_pairs = img[:-1:2]-img[1::2]
		del img
		gc.collect() ## Free up Memory: ##
		exp_diff_pairs = np.asarray(exp_diff_pairs)
		print "\n Starting to Calculate the Polar Images..."
		exp_diff_pairs = np.array( [ polar_mask* Interp.nearest(exp_diff_pairs[i]) for i in range(exp_diff_pairs.shape[0]) ] ) 
		pttrn = "pairs"
		# if 'exp_diff_pairs' not in data_hdf.keys(): data_hdf.create_dataset( 'exp_diff_pairs', data = np.asarray(exp_diff_pairs))
		# else: 
		# 	del data_hdf['exp_diff_pairs']
		# 	dset = data_hdf.create_dataset('exp_diff_pairs', data=np.asarray(exp_diff_pairs))
		exposures = exp_diff_pairs #exposure_diffs_cart #exposure_diffs_pol
		del exp_diff_pairs 
		gc.collect() ## Free up Memory: ##
	print "exposures vector's shape", exposures.shape
	del Interp

	diff_count = exposures.shape[0]  ## The number off differenses used in the correlation cacluations ##
	# ---- Autocorrelation of each Pair: ----
	print "\n Starting to Auto-Correlate the Images... "
	#acorr = [RingData.DiffCorr( exposure_diffs_cart).autocorr(), RingData.DiffCorr( exposure_diffs_pol ).autocorr()]
	#cor_mean = [RingData.DiffCorr( exposure_diffs_cart).autocorr().mean(0), RingData.DiffCorr( exposure_diffs_pol ).autocorr().mean(0)]
	acorr = DiffCorr( exposures ).autocorr()
	#cor_mean =acorr.mean(0)
	#cor_mean = np.asarray(cor_mean)



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
		dset = data_hdf.create_dataset( 'sum_intensity_pattern', data = np.asarray(IP_sum))
		dset.attrs["number_patterns"] = shot_count
	else: 
		del data_hdf['sum_intensity_pattern']
		dset = data_hdf.create_dataset('sum_intensity_pattern', data=np.asarray(IP_sum))
		dset.attrs["number_patterns"] = shot_count
	#if 'diff_type' not in data_hdf.keys():	data_hdf.create_dataset( 'diff_type', data = pttrn, dtype=h5py.special_dtype(vlen=str)) ## Py3 ## 
	#if 'diff_type' not in data_hdf.keys():	data_hdf.create_dataset( 'diff_type', data = pttrn, dtype=h5py.special_dtype(vlen=unicode)) ## Py2 ##
	#else: 
	#	del data_hdf['diff_type']
	#	#dset = data_hdf.create_dataset('diff_type', data = pttrn, dtype=h5py.special_dtype(vlen=str)) ## Py3 ## 
	#	dset = data_hdf.create_dataset('diff_type', data = pttrn, dtype=h5py.special_dtype(vlen=unicode)) ## Py2 ##
	#	#dset.attrs["diff_type"] = numpy.string_(pttrn)
	#if 'number_patterns' not in data_hdf.keys(): data_hdf.create_dataset( 'number_patterns', data = shot_count, dtype='i4')
	#else: 
	#	del data_hdf['number_patterns']
	#	dset = data_hdf.create_dataset('number_patterns', data=shot_count, dtype='i4' )
	#if 'number_of_diffs' not in data_hdf.keys(): data_hdf.create_dataset( 'number_of_diffs', data = diff_count, dtype='i4')
	#else: 
	#	del data_hdf['number_of_diffs']
	#	dset = data_hdf.create_dataset('number_of_diffs', data=diff_count, dtype='i4' )

	#del  cntr # Free up memory
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
	tot_corr_sum = 0	## the number of ACC perfomed, one per loaded file ##
	tot_shot_sum = 0	## the total number of shots (diffraction pattersn) from simulation ##
	tot_diff_sum = 0	## the number of diffs or the number of patterns in input to ACC ##
	q_map= None
	nphi= None
	diff_type_str= None
	for file in list_names:
		with h5py.File(file, 'r') as f:
			dset_ac = f['auto-correlation']
			acc=np.asarray(dset_ac)		#np.asarray(f['auto-correlation'])
			diff_count= dset_ac.attrs["number_of_diffs"]
			shot_count= dset_ac.attrs["number_patterns"] 
			ip_sum=np.asarray(f['sum_intensity_pattern']) ## (Y,X) ##
			#print "\n Dim of Sum of intensity patterns: ", ip_sum.shape
			if file==list_names[0]:
				q_map=np.asarray(f['q_mapping'])
				#cntr=np.asarray(f['beam_center']) ## Not Saved Yet ##
				#nphi=int(np.asarray(f['num_phi'])[0]) ## IndexError: too many indices for array
				nphi=int(np.asarray(f['num_phi'])) 
				#diff_type_str =f['diff_type'][:] ## ValueError: Illegal slicing argument for scalar dataspace
				#diff_type_str =f['diff_type'] ## ValueError: Not a dataset (not a dataset) when store l.193
				#diff_type_str =str(f['diff_type'])
				diff_type_str =dset_ac.attrs["diff_type"]
		corr_sum.extend(acc)
		ip_mean.append(ip_sum) ## (Y,X)
		tot_corr_sum+= 1 ##corr_count ## calculate the number of auto-correlations loaded ##
		tot_shot_sum+= shot_count 
		tot_diff_sum+= diff_count 
	#tot_shot_sum=100*91 ## !OBS:Temp fix for forgotten to store data ##
	corr_sum = np.asarray(corr_sum)
	corr_sum = np.sum(corr_sum, 0) ## Sum all the ACC  ##
	#ip_mean = np.asarray(ip_mean )
	print "\n length of array of Sum of intensity patterns: ", len(ip_mean)
	ip_mean  = np.sum(np.asarray(ip_mean ), 0) ## sum of all the intensity patterns  ##
	print "\n Dim of Sum of intensity patterns: ", ip_mean.shape
	#ip_mean=ip_mean.astype(float) 	## convert to float else porblem with 'same_kind' when dividing with float  ##
	#ip_mean /= float(tot_shot_sum)	 ## mean of all the intensity patterns in float ##
	#ip_mean = np.around(ip_mean).astype(int)	 ## Round up/down ##
	ip_mean /= tot_shot_sum		## Alt. work only in int ##
	print "\n Dim of Mean of intensity patterns: ", ip_mean.shape
	print "\n dim of ACC: ", corr_sum.shape
	t = time.time()-t_load
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Loading Time for %i patterns: "%(len(list_names)), t_m, "min, ", t_s, "s " # 5 min, 19.2752921581 s


	## ---- Correlate the Mask : ---- ##
	#nphi = 360 ## Temp fix, stored nphi is 127 (with dtype='i4') ##
	print "\n nphi = %i, " %(nphi)
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2]
	cntr =np.asarray([(mask.shape[1]-1)/2.0, (mask.shape[0]-1)/2.0 ]) ## (X,Y)
	Interp = InterpSimple( cntr[0], cntr[1], qmax_pix, qmin_pix, nphi, mask.shape) 
	mask_polar = Interp.nearest(mask, dtype=bool)
	print "\n dim of mask converted to polar space: ", mask_polar.shape
	mask_DC = DiffCorr(mask_polar.astype(int), pre_dif=True)
	mask_acc = mask_DC.autocorr()
	print "\n dim of ACC(Mask): ", mask_acc.shape


	## ---- Store the Total Results : ---- ##
	dset =data_hdf.create_dataset( 'auto-correlation-sum', data = np.asarray(corr_sum))
	dset.attrs["diff_type"]= np.string_(diff_type_str)
	dset.attrs["tot_number_of_diffs"]= tot_diff_sum
	dset.attrs["tot_number_patterns"]= tot_shot_sum
	dset.attrs["tot_number_of_corrs"]= tot_corr_sum
	data_hdf.create_dataset( 'mask_auto-correlation', data = np.asarray(mask_acc))
	data_hdf.create_dataset( 'q_mapping', data = q_map)
	#data_hdf.create_dataset( 'tot-corr', data = tot_corr_sum, dtype='i4' )
	#data_hdf.create_dataset( 'tot-shots', data = tot_shot_sum, dtype='i4' )
	data_hdf.create_dataset( 'num_phi', data = nphi, dtype='i4') ## i4 = 32 -bit ##
	#data_hdf.create_dataset( 'diff_type', data = diff_type_str, dtype=h5py.special_dtype(vlen=str))
	data_hdf.create_dataset( 'mean_intensity_pattern', data = np.asarray(ip_mean))
	# ---- Save by closing file: ----
	data_hdf.close()
	print "\n File Closed! \n"

	if str(args.plt_set).lower() in ('single', 'all'):
		plot_acc(corr_sum,tot_corr_sum,q_map,nphi,diff_type_str,mask_acc,ip_mean,tot_shot_sum, out_name, mask=mask) ## tot_shot_sum
	if str(args.plt_set).lower() in ('subplot','all'):
		subplot_acc(corr_sum,tot_corr_sum,q_map,nphi,diff_type_str,mask_acc,ip_mean,tot_shot_sum, out_name, mask=mask) ## tot_shot_sum

	exit(0)  ## Finished ! ##

# ------------------------------------------------------------

def plot_acc(corrsum,corr_count, q_map, nphi, pttrn,corr_mask, IP_mean, shot_count,  out_fname, mask=None):
	"""
	Read in ACC data from multiple files in '' list of filenames. 
	Plot the average from all read-in files.

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
	print "\n Plotting..."
	frmt = "eps" 	## Format used to save plots ##
	## columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2] ## first and last values in cloumn 2 = q [pixels] ##
	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14		## Fontsize of axis labels ##
	sb_size = 16 #18 	## Fontsize of Sub-titles ##
	sp_size = 18#20  	## Fontsize of Super-titles ##
	l_pad = 10 ## ave_corrs-script has 50 ##

	cb_shrink, cb_padd = 1.0, 0.2  ## Colorbar padding ##

	padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181

	if 'all' in pttrn : frac = 0.8 			## Limit the vmax with this percentage ##
	else: frac = 1
	##################################################################################################################
	##################################################################################################################
	#-------------------------------- Fig.1 ACC: --------------------------------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC."
	fig1 = pypl.figure('AC', figsize=(22,15))
	#corrsum  =np.nan_to_num(corrsum)
	sig = corrsum/corrsum[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	#sig = corrsum/corrsum[:,0][:,None] / corr_count
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape #carteesian = (1738, 1742): polar =(190, 5)

	m = sig[:,padC:-padC].mean() 	# MEAN
	s = sig[:,padC:-padC].std() 	# standard Deeviation
	vmin = m-2*s
	vmax = m+2*s
	
	ax = pypl.gca()
	polar_axis = True 	## Plot with axis in polar coord and with polar labels, else pixels ##
	if polar_axis:
		im = ax.imshow( sig,
                     extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
                        vmin=vmin, vmax=vmax, aspect='auto')
	else : im =ax.imshow(sig, vmin=vmin, vmax=vmax, aspect='auto' )
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
		#print "q_label: ", q_label

		ytic=ax.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		#ylab= q_label
		ax.set_yticklabels(q_label)
		ax.set_yticks(ytic)

		# #---- Label xy-Axis: ----
		ax.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	ave_corr_title = ax.set_title(r"Average of %d corrs [%s] with limits $\mu \pm 2\sigma$"%(corr_count, pttrn),  fontsize=sb_size)
	ave_corr_title.set_y(1.08) # 1.1)
	############################## [fig.] End ##############################
	fig_name = "Diff-Auto-Corr_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#pypl.show()
	#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	fig1.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Plot saved as %s " %fig_name
	del fig_name 	# clear up memory1
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##

	##################################################################################################################
	##################################################################################################################
	#--------------------- Fig.2 ACC Normalised with correlated Mask: -----------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC Normalied wit ACC of the Mask."
	fig2 = pypl.figure('AC_N_Msk', figsize=(22,15))
	## ---- Normalize (avoide division with 0) ----- ##
	corrsum_m = np.divide(corrsum, corr_mask, out=None, where=corr_mask!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
	sig = corrsum_m/corrsum_m[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	## withtNomred 0th angle ##
	## every q : row vise normalization with Mean of q : norm each q  bin by average of that q
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape 

	#padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181
	m = sig[:,padC:-padC].mean() 	# MEAN
	s = sig[:,padC:-padC].std() 	# standard Deeviation
	vmin = m-2*s
	vmax = m+2*s
	
	ax = pypl.gca()
	polar_axis = True 	## Plot with axis in polar coord and with polar labels, else pixels ##
	if polar_axis:
		im = ax.imshow( sig,
                     extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
                        vmin=vmin, vmax=vmax*frac, aspect='auto')
	else : im =ax.imshow(sig, vmin=vmin, vmax=vmax*frac, aspect='auto' )
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
	Norm_ave_corr_title = ax.set_title(r"Average of %d corrs [%s] (Normalized with Mask) with limits $\mu \pm 2\sigma$"%(corr_count, pttrn),  fontsize=sb_size)
	Norm_ave_corr_title.set_y(1.08) # 1.1)
	############################## [fig.] End ##############################
	fig_name = "Diff-Auto-Corr_Normed_w_Mask_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#pypl.show()
	#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	fig2.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Plot saved as %s " %fig_name
	del fig_name 	# clear up memory1
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##
	##################################################################################################################
	##################################################################################################################
	#--------------------- Fig.3 ACC of the Mask: -----------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	print "\n Plotting ACC of the Mask."
	fig3 = pypl.figure('AC_of_Msk', figsize=(22,15))

	sig_m = corr_mask/corr_mask[:,0][:,None]#/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig_m  =np.nan_to_num(sig_m)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...

	#padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181
	m = sig_m[:,padC:-padC].mean() 	# MEAN
	s = sig_m[:,padC:-padC].std() 	# standard Deeviation
	vmin = m-2*s
	vmax = m+2*s
	
	ax = pypl.gca()
	polar_axis = True 	## Plot with axis in polar coord and with polar labels, else pixels ##
	if polar_axis:
		im = ax.imshow( sig_m,
                     extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
                        vmin=vmin, vmax=vmax, aspect='auto')
	else : im =ax.imshow(sig, vmin=vmin, vmax=vmax, aspect='auto' )
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
		#print "q_label: ", q_label

		ytic=ax.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		#ylab= q_label
		ax.set_yticklabels(q_label)
		ax.set_yticks(ytic)

		# #---- Label xy-Axis: ----
		ax.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	ACC_mask_title = ax.set_title(r"ACC of Mask with limits $\mu \pm 2\sigma$",  fontsize=sb_size)
	ACC_mask_title.set_y(1.08) # 1.1)
	############################## [fig.] End ##############################
	fig_name = "ACC_Mask_(qx-%i_qi-%i_nphi-%i)_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#pypl.show()
	#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	fig3.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Plot saved as %s " %fig_name
	del fig_name 	# clear up memory1
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##


	##################################################################################################################
	##################################################################################################################
	#-------------------------- Fig.4 Plotting  Mean Intensiyt Pattern --------------------------------------------#
	##################################################################################################################
	##################################################################################################################
	color_map ='jet' #'viridis'
	print "\n Plotting some Diffraction Patterns."
	fig4 = pypl.figure('Mean_Intensity', figsize=(15,12)) 
	ax_tr = pypl.gca()  ## Bottom Axis ##
	cbs,cbp =  0.98, 0.02 #0.04, 0.1: left plts cb ocerlap middle fig
	ax_tr.set_ylabel( 'y Pixels', fontsize=axis_fsize) 
	ax_tr.set_xlabel( 'x Pixels', fontsize=axis_fsize) 
	#pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'})
	## ax0.set( ylabel='y Pixels')
	## ax0.set( xlabel='x Pixels')
	# ---- Intensity Pattern : ---- #
	if mask is not None:
		Ip_ma_shot = np.ma.masked_where(mask == 0, IP_mean) ## Check the # photons ##
	else: Ip_ma_shot = IP_mean
	im_ip =ax_tr.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
	cb_ip = pypl.colorbar(im_ip, ax=ax_tr,fraction=0.046)
	cb_ip.set_label(r' Photons (mean) ') #(r'Intensity ')
	#cb_ip.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	#cb_ip.update_ticks()
	#cmap.set_bad('grey',1.) ## default: (color='k', alpha=None) Set color to be used for masked values. ##
	#cmap.set_under('white',1.) ## Set color to be used for low out-of-range values. Requires norm.clip = False ##
	
	rad_title =pypl.title('Mean Intensity of %i Patterns: ' %(shot_count), fontsize=sb_size)# %s_%s_%s_#%i\n' %(name,run,pdb,N), fontsize=sb_size)
	rad_title.set_y(1.08) # 1.1), 1.08 OK but a bit low, 1.04 =overlap with Q
	pypl.gca().tick_params(labelsize=axis_fsize, length=9)
	############################## [fig.] End ##############################
	fig_name = "Mean_Intensity.%s" %(frmt)
	#pypl.show()
	#pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	fig4.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Plot saved as %s " %fig_name
	del fig_name 	# clear up memory1
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

	cb_shrink, cb_padd = 1.0, 0.2  ## Colorbar padding ##
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
	cbs,cbp =  0.98, 0.02 #0.04, 0.1: left plts cb ocerlap middle fig
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
                        vmin=vmin, vmax=vmax, aspect='equal')
	else : im_m =ax_tl.imshow(sig, vmin=vmin, vmax=vmax, aspect='equal' )
	#cb = pypl.colorbar(im, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)
	cb = pypl.colorbar(im_m, ax=ax_tl, shrink=cb_shrink, pad= cb_padd)
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
	calc_acc(img=shots, cntr=cntr, q_map= q_mapping, mask = mask_better, data_hdf=out_hdf)

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

parser.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
      help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)

subparsers = parser.add_subparsers()#title='calculate', help='commands for calculations help')

subparsers_calc = subparsers.add_parser('calculate', help='commands for calculations ')
subparsers_calc.add_argument('-d', '--dir-name', dest='dir_name', default='this_dir', type=str,
      help="The Location of the directory with the cxi-files (the input).")
subparsers_calc.add_argument('-s', '--simulation-number', dest='sim_n', default=None, type=int, 
      help="The number of the pdb-file, which simulation is to be loaded from 0 to 90, e.g. '20' for the file <name>_4M20_<properties>l.cxi")
#subparsers_e = subparsers.add_parser('e', help='a choices')
subparsers_calc.add_argument('-e', '--exposure', dest='exp_set', default='pair', type=str, choices=['pair', 'diffs', 'all'],
      help="Select how to auto-correalte the data: 'pair' (pair-wise difference between shots),'diffs', 'all' (all shot without difference).")
# subparsers_calc.add_argument('-m', '--masked', dest='w_MASK', default=True, type=lambda s: (str(s).lower() in ['false', 'f', 'no', 'n', '0']),
#       help="Select if the mask is included in the auto-correalte calculations.")
# subparsers_calc.add_argument('-m', '--unmasked', dest='w_MASK', action='store_false',
#       help="Select if the mask is included in the auto-correalte calculations.")
subparsers_calc.add_argument('-m', dest='w_MASK', action='store_true')
subparsers_calc.add_argument('--no-mask', dest='w_MASK', action='store_false')
subparsers_calc.set_defaults(w_MASK=True)
subparsers_calc.set_defaults(func=load_and_calculate_from_cxi)

subparsers_plt = subparsers.add_parser('plots', help='commands for plotting ')
subparsers_plt.add_argument('-p', '--plot', dest='plt_set', default='single', type=str, choices=['single', 'subplot', 'all'],
      help="Select which plots to execute and save: 'single' each plot separately,'subplot' only a sublpot, 'all' both single plots and a subplot.")
subparsers_plt.set_defaults(func=load_and_plot_from_hdf5)

args = parser.parse_args()
args.func(args) ## if .set_defaults(func=) SUBPARSERS ##
## ---------------- ARGPARSE  END ----------------- ##