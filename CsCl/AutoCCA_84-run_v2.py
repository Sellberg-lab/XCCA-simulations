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
# 2019-03-11 v2 @ Caroline Dahlqvist cldah@kth.se
#			AutoCCA_84-run_v2.py
#			v2 with -s input: choose which simulated file to process
#				without -s: process hdf5-files with stored ACC-cacluations
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
import matplotlib.cm as cm
import matplotlib.pyplot as pypl
from matplotlib import ticker
# pypl.rcParams["image.cmap"] = "jet" ## Change default cmap from 'viridis to 'jet ##
from loki.RingData import RadialProfile,DiffCorr, InterpSimple ## scripts located in RingData-folder ##
#from pylab import *	# load all Pylab & Numpy
# %pylab	# code as in Matlab
import os, time
import gc
this_dir = os.path.dirname(os.path.realpath(__file__)) ## Get path of directory
#this_dir = os.path.dirname(os.path.realpath('CCA_RadProf_84-run.py')) ##for testing in ipython
if "/home/" in this_dir: #/home/cldah/cldah-scratch/ or /home or 
	os.environ['QT_QPA_PLATFORM']='offscreen'

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Radial Profile and Auto-Correlations.")

parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
      help="The Location of the directory with the cxi-files (the input).")

parser.add_argument('-o', '--outpath', dest='outpath', default='dir_name', type=str, help="Path for output, Plots and Data-files")

parser.add_argument('-s', '--simulation-number', dest='sim_n', default=None, type=int, 
	   help="The number of the pdb-file, which simulation is to be loaded from 0 to 90, e.g. '20' for the file <name>_4M20_<properties>l.cxi")
## noisy = args.dtct_noise     # Noise type: {None, "poisson", "normal"=gaussian, "normal_poisson" = gaussian and poisson}
##if noisy == "None": noisy = None

#parser.add_argument('-f', '--fname', dest='sim_name', default='test', type=str,
#      help="The Name of the Simulation. Default name is 'test'.")

parser.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
      help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)

args = parser.parse_args()
## ---------------- ARGPARSE  END ----------------- ##


# ------------------------------------------------------------
def create_out_dir(out_dir, sim_res_dir,name,run,pdb,noisy,n_spread, random_noise_1quad=False):
	"""
	Create/ Define the output directory for storing the resulting data.
	"""
	if out_dir== sim_res_dir:
		outdir = sim_res_dir+'/%s_%s_%s_(%s-sprd%s)/' %(name,run,pdb,noisy,n_spread)
	else:	outdir = args.outpath
	if not os.path.exists(outdir):
		os.makedirs(outdir)# os.makedirs(outdir, 0777)
	if random_noise_1quad:
		outdir = outdir + '/random_noise_1quad_(%s_%s_%s)/' %(pdb,noisy,n_spread)
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
#read_and_plot_acc(fnames, out_hdf, out_fname) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]
def read_and_plot_acc(list_names, data_hdf, out_name):
	"""
	Read in ACC data from multiple files in 'list_names' list of filenames. 
	Store the resutl in 'data_hdf' and then 
	Plot the average from all read-in files (saved in 'out_name'-location).
	"""
	t_load =time.time()
	corr_sum = []
	tot_corr_sum = 0
	tot_shot_sum = 0
	tot_diff_sum = 0
	q_map= None
	nphi= None
	diff_type_str= None
	for file in list_names:
		with h5py.File(file, 'r') as f:
			acc=np.asarray(f['auto-correlation'])
			#diff_count=int(np.asarray(f['number_of_diffs''])[0]) ## OBS! Not Stored yet ! 
			#shot_count=int(np.asarray(f['number_patterns'])[0]) ## OBS! Not Stored yet ! 
			#diff_count=f.attrs["number_of_diffs"]
			#shot_count= f.attrs["number_patterns"] 
			##diff_type_str =f['diff_type'] ## should not be changed between runs (True/False statements) ##
			if file==list_names[0]:
				q_map=np.asarray(f['q_mapping'])
				#nphi=int(np.asarray(f['num_phi'])[0]) ## IndexError: too many indices for array
				nphi=int(np.asarray(f['num_phi'])) 
				#diff_type_str =f['diff_type'][:] ## ValueError: Illegal slicing argument for scalar dataspace
				diff_type_str =f['diff_type'] ## ValueError: Not a dataset (not a dataset) when store l.193
				diff_type_str =str(f['diff_type'])
				# diff_type_str =f.attrs["diff_type"]
		corr_sum.extend(acc)
		tot_corr_sum+= 1 ##corr_count ## calculate the number of auto-correlations loaded ##
		#tot_shot_sum+= shot_count ## OBS! Not Stored yet ! 
		#tot_diff_sum+= diff_count ## OBS! Not Stored yet ! 
	tot_shot_sum=100*91 ## !OBS:Temp fix for forgotten to store data ##
	corr_sum = np.asarray(corr_sum)
	corr_sum = np.sum(corr_sum, 0) ## Sum all the ACC  ##
	t = time.time()-t_load
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Loading Time for %i patterns: "%(len(list_names)), t_m, "min, ", t_s, "s \n" # 5 min, 19.2752921581 s

	## ---- Store the Total Results : ---- ##
	dset =data_hdf.create_dataset( 'auto-correlation-sum', data = np.asarray(corr_sum))
	dset.attrs["diff_type"]= np.string_(diff_type_str)
	data_hdf.create_dataset( 'q_mapping', data = q_map)
	data_hdf.create_dataset( 'tot-corr', data = tot_corr_sum, dtype='i4' )
	data_hdf.create_dataset( 'tot-shots', data = tot_shot_sum, dtype='i4' )
	data_hdf.create_dataset( 'num_phi', data = nphi, dtype='i1')
	#data_hdf.create_dataset( 'diff_type', data = diff_type_str, dtype=h5py.special_dtype(vlen=str))
	# ---- Save by closing file: ----
	data_hdf.close()
	print "\n File Closed! \n"

	plot_acc(corr_sum,tot_corr_sum,q_map,nphi, diff_type_str,out_name) ## tot_shot_sum

# ------------------------------------------------------------
def plot_acc(corrsum,corr_count, q_map, nphi, pttrn, out_fname ):
	"""
	Read in ACC data from multiple files in '' list of filenames. 
	Plot the average from all read-in files.
	"""

	frmt = "eps" 	## Format used to save plots ##
	## columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2] ## first and last values in cloumn 2 = q [pixels] ##
	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14
	sb_size = 16 #18
	sp_size = 18#20
	l_pad = 10
	
	fig1 = pypl.figure('AC', figsize=(22,15))
	## subplot , constrained_layout=True 
	cb_shrink, cb_padd = 1.0, 0.2 
	## ---- if running MPI: ---- ##
	#corrsum = np.zeros_like(corr)		
	#corr_count =np.zeros(1)+(exposure_diffs.shape[0]) #exposure_diffs.shape[0] = N-1
	#tot_corr_count = np.zeros(1)
	#print "\n tot_corr_count: ", tot_corr_count
	#from mpi4py import MPI
	#comm = MPI.COMM_WORLD
	#comm.Reduce(corr,corrsum)
	#comm.Reduce(corr_count, tot_corr_count)
	#print "\n tot_corr_count after comm.Reduce(): ", tot_corr_count # N=5: [4.]

	#corrsum  =np.nan_to_num(corrsum)
	sig = corrsum/corrsum[:,0][:,None]/corr_count #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	#sig = corrsum/corrsum[:,0][:,None] / corr_count
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape #carteesian = (1738, 1742): polar =(190, 5)

	padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181
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
		print "q_label: ", q_label

		ytic=ax.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		ylab= q_label
		ax.set_yticklabels(q_label)
		ax.set_yticks(ytic)

		# #---- Label xy-Axis: ----
		ax.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)

	ave_corr_title = ax.set_title(r"Average of %d corrs [%s] with limits $\mu \pm 2\sigma$"%(corr_count, pttrn),  fontsize=sb_size)
	ave_corr_title.set_y(1.08) # 1.1)
	############################## [fig.b] End ##############################
	
	fig_name = "Diff-Auto-Corr_(qx-%i_qi-%i_nphi-%i)_w_Mask_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#pypl.show()
	pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	print "\n Subplot saved as %s " %fig_name
	del fig_name 	# clear up memory1
	#pypl.cla() ## clears axis ##
	#pypl.clf() ## clears figure ##

	exit(0)  ## Finished ! ##

# -------------------------------------- ##if SACLA dsnt work use:
def norm_data(data): 	# from Loki/lcls_scripts/get_polar_data
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
	if pol_frac is not None:
		assert (dtc_dist is not None)
		pixsize = 0.00010992
		Y,X = np.indices(m_b.shape)
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
		Polarize = np.ones_like( m_b)
	return data/Polarize
# --------------------------------------
# ------------------------------------------------------------
#def calc_acc(images,  cntr, qmax_pix, qmin_pix, q_map, mask, data_hdf):
def calc_acc(images,  cntr, q_map, mask, data_hdf):
	"""
	Caclulate the Auto-Correlation for the 'data' with LOKI.
	"""
	# ---- Some Useful Functions : ----/t/ from Sacla-tutorial                                                    
	#pix2invang = lambda qpix : np.sin(np.arctan(qpix*(ps*1E-6)/dtc_dist )*0.5)*4*np.pi/wl_A
	img= images
	t_AC = time.time()
	cart_diff, pol_diff, pair_diff = False,False,True ## only one should be true !!

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

	# ---- Make a Polar Mask : ----
	polar_mask = Interp.nearest(mask, dtype=bool).round() ## .round() returns a floating point number that is a rounded version of the specified number ##

	if cart_diff:
		# ---- Calculate the Difference in Intensities and store in a List: ----
		exposure_diffs_cart =img[:-1]-img[1:]				# Intensity Patterns in Carteesian cor
		img_mean = np.array([ img[i].mean() for i in range(img.shape[0]) ]) ## Calculate the Mean of each pattern(shot) ##
		exposure_diffs_cart= img[:-1]/img_mean[:-1]-img[1:]/img_mean[1:]  ## Normalize each pattern(shot) before subtraction ##
		del img,img_mean 	## Do not need the mean-values anymore ##
		gc.collect() ## Free up Memory: ##
		exposure_diffs_cart = np.asarray(exposure_diffs_cart) 	# = (4, 1738, 1742)
		exposure_diffs_cart = polarize(exposure_diffs_cart)  ## Polarize (if no fraction given => just 1s) ##
		## ---- Conv to polar of diff-data : ---- ##
		print "\n Starting to Calculate the Polar Images...\n"
		exposure_diffs_cart = np.array( [ polar_mask* Interp.nearest(exposure_diffs_cart[i]) for i in range(exposure_diffs_cart.shape[0]) ] ) 
		pttrn = "cart-diff"
		if 'exposure_diffs_cart' not in data_hdf.keys(): data_hdf.create_dataset( 'exposure_diffs_cart', data = np.asarray(exposure_diffs_cart))
		else: 
			del data_hdf['exposure_diffs_cart']
			dset = data_hdf.create_dataset('exposure_diffs_cart', data=np.asarray(exposure_diffs_cart))
		exposure_diffs = exposure_diffs_cart 
		del  exposure_diffs_cart 
		gc.collect() ## Free up Memory: ##
	elif pol_diff:
		## ---- Calc diff-data direct from polar Images: ---- ##
		# ---- Generate Polar Image/Images (N Diffracton Patterns) : ----
		print "\n Starting to Calculate the Polar Images...\n"
		polar_imgs = np.array( [ polar_mask* Interp.nearest(img[i]) for i in range( N) ] )
		## ---- Normalize - with function: ---- ##
		## alt try: polar_imgs_norm = norm_data(polar_imgs)
		polar_imgs_norm = polar_imgs
		exposure_diffs_pol =polar_imgs_norm[:-1]-polar_imgs_norm[1:]	# Polar Imgs
		del img, polar_imgs,polar_imgs_norm
		gc.collect() ## Free up Memory: ##
		exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)
		pttrn = "polar-diff"
		if 'exposure_diffs_pol' not in data_hdf.keys(): data_hdf.create_dataset( 'exposure_diffs_pol', data = np.asarray(exposure_diffs_pol))
		else: 
			del data_hdf['exposure_diffs_pol']
			dset = data_hdf.create_dataset('exposure_diffs_pol', data=np.asarray(exposure_diffs_pol))
		exposure_diffs = exposure_diffs_pol 
		del exposure_diffs_pol
		gc.collect() ## Free up Memory: ##
	elif pair_diff :
		## ---- Calc diff-data from Pairs of Images: ---- ##
		exp_diff_pairs = img[:-1:2]-img[1::2]
		del img
		gc.collect() ## Free up Memory: ##
		exp_diff_pairs = np.asarray(exp_diff_pairs)
		print "\n Starting to Calculate the Polar Images...\n"
		exp_diff_pairs = np.array( [ polar_mask* Interp.nearest(exp_diff_pairs[i]) for i in range(exp_diff_pairs.shape[0]) ] ) 
		pttrn = "pairs"
		if 'exp_diff_pairs' not in data_hdf.keys(): data_hdf.create_dataset( 'exp_diff_pairs', data = np.asarray(exp_diff_pairs))
		else: 
			del data_hdf['exp_diff_pairs']
			dset = data_hdf.create_dataset('exp_diff_pairs', data=np.asarray(exp_diff_pairs))
		exposure_diffs = exp_diff_pairs #exposure_diffs_cart #exposure_diffs_pol
		del exp_diff_pairs 
		gc.collect() ## Free up Memory: ##
	print "exposure diff vector's shape", exposure_diffs.shape
	del Interp

	diff_count = exposure_diffs.shape[0]  ## The number off differenses used in the correlation cacluations ##
	# ---- Autocorrelation of each Pair: ----
	print "\n Starting to Auto-Correlate the Polar Images...\n"
	#acorr = [RingData.DiffCorr( exposure_diffs_cart).autocorr(), RingData.DiffCorr( exposure_diffs_pol ).autocorr()]
	#cor_mean = [RingData.DiffCorr( exposure_diffs_cart).autocorr().mean(0), RingData.DiffCorr( exposure_diffs_pol ).autocorr().mean(0)]
	acorr = DiffCorr( exposure_diffs ).autocorr()
	#cor_mean =acorr.mean(0)
	#cor_mean = np.asarray(cor_mean)

	# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
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
	if 'num_phi' not in data_hdf.keys():	data_hdf.create_dataset( 'num_phi', data = nphi, dtype='i1') #INT, dtype='i1', 'i8'f16'
	else: 
		del data_hdf['num_phi']
		dset = data_hdf.create_dataset('num_phi',  data = nphi, dtype='i1')
	if 'q_mapping' not in data_hdf.keys():	data_hdf.create_dataset( 'q_mapping', data = q_map)
	else: 
		del data_hdf['q_mapping']
		dset = data_hdf.create_dataset('q_mapping',data = q_map)
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
	print "AutoCorrelation Time: ", t_m, "min, ", t_s, " s \n"
	exit(0)
# ------------------------------------------------------------
###################### Choose Calculations: ########################################
Ampl_image = False			## The complex images in Amplitude Patterns instead of Intensity Patterns ##
#add_noise = False						## Add Generated Noise to Correlations ##

#random_noise_1quad =False			## Generate random Noise in one quardant ##


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
print"Dim of the assembled mask: ", mask_better.shape

# ---- Centre Coordiates Retrieved Directly from File ((X,Y): fs = fast-scan {x}, ss = slow-scan {y}): ----
#cntr = np.load("%s/centers/better_cent_lt14.npy" %str(this_dir))	# from exp-file run 84-119
#print "Centre from file: ", cntr ## [881.43426    863.07597243]
cntr_msk =np.asarray([(mask_better.shape[1]-1)/2.0, (mask_better.shape[0]-1)/2.0 ]) ## (X,Y)
print "\n Centre from Mask: ", cntr_msk ##[870, 868]; [869.5, 871.5]; [871.5, 869.5]
cntr= cntr_msk 	## (X,Y) if use center point from the Msak ##
cntr_int=np.around(cntr).astype(int)  ## (rounds to nearest int) for the center coordinates in pixles as integers ##
#print "\n Centre as Integer: ", cntr_int,  "\n"


pttrn = "Int"
if Ampl_image:
	pttrn = "Ampl"
#if add_noise:
#	pttrn = "Int-add-noise-%iprc" %(nlevel*100)


## ---- Load File Names in Directory and Sort Numerically after Concentraion: ---- ##
#sim_res_dir = args.dir_name

##################################################################################################################
#------------------------------------ Mean and Plot: ----------------------------------------------------------------#
##################################################################################################################
if args.sim_n==None:	## LOAD hdf5-Files and Plot the Mean ##
	fnames = [ os.path.join(args.outpath, f) for f in os.listdir(args.outpath)
		 if f.startswith('Parts-ACC_') and f.endswith('.hdf5') ]
	if not fnames:
		print"\n No filenames for directory %s"%(str(args.dir_name))
	fnames = sorted(fnames, 
		key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0][0] )) 	## For sorting simulated-file 0-90 ##
	#fnames = sorted(fnames, key=lambda x: int(x.split('84-')[-1][4:].split('_ed')[0][0] )) 	## For sorting conc 4-6 M ##

	# ----- Parameters unique to Simulation: ---- ##
	cncntr_start = fnames[0].split('84-')[-1][4:].split('_(')[0] 	## '...84-105_6M90_ed...'=> '6M90' ##
	## Parts-ACC_Fnoise_BeamNarrInt_84-119_4M74_(none-sprd0).cxi)_Int.hdf5
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
	outdir = create_out_dir(out_dir = args.outpath, sim_res_dir=args.dir_name,name=name,run=run,pdb=pdb,noisy=noisy,n_spread=n_spread)
	out_fname = os.path.join( outdir, prefix) 
	#out_hdf = h5py.File( out_fname + 'tot_Auto-corr_%s.hdf5' %(pttrn), 'a')    # a: append, w: write
	out_hdf = h5py.File( out_fname + 'tot_Auto-corr_%s.hdf5' %(pttrn), 'w')    # a: append, w: write
	print "\n Data Analysis with LOKI.\n"
	## ---- Read in ACC from Files and Plot ----  ##
	read_and_plot_acc(fnames, out_hdf, out_fname) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]

##################################################################################################################
#------------------------------------ Loki XCCA: ----------------------------------------------------------------#
##################################################################################################################
else:	## LOAD the cxi-file (one) from 'dir_name', calculate the ACC and store result in hdf5-files in 'outpath' ##
	fnames = [ os.path.join(args.dir_name, f) for f in os.listdir(args.dir_name)
		 if f.endswith('cxi') ]
	if not fnames:
		print"\n No filenames for directory %s"%(str(args.dir_name))
	fnames = sorted(fnames, 
		key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0][0] )) 	## For sorting simulated-file 0-90 ##
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
	photon_e_eV, wl_A, ps, dtc_dist, N, shots= load_cxi_data(data_file= fname, get_parameters= True)

	# ---- Save g-Mapping (the q- values [qmin_pix to qmin_pix]): ----
	pix2invang = lambda qpix : np.sin(np.arctan(qpix*(ps*1E-6)/dtc_dist )*0.5)*4*np.pi/wl_A
	# invang2pix = lambda qia : np.tan(2*np.arcsin(qia*wl_A/4/pi))*dtc_dist/(ps*1E-6)
	qrange_pix = np.arange( args.q_range[0], args.q_range[1])
	## ---- Q-Mapping with  0: indices, 1: r [inv Angstrom], 2: q[pixels] ---- ##
	q_mapping = np.array( [ [ind, pix2invang(q), q] for ind,q in enumerate( qrange_pix)]) 
	
	# ---- Generate a Storage File: ---- ##
	#prefix = 'Parts-ACC_%s_%s_%i-shots' %(name,pdb,N) ## Partial Result ##
	prefix = 'Parts-ACC_%s_84-%s_%s_(%s-sprd%s)' %(name,run,pdb,noisy,n_spread)
	outdir = create_out_dir(out_dir = args.outpath, sim_res_dir=args.dir_name,name=name,run=run,pdb=pdb,noisy=noisy,n_spread=n_spread)
	out_fname = os.path.join( outdir, prefix) 
	# ---- Generate a Storage File for Data/Parameters: ----
	out_hdf = h5py.File( out_fname + '_%s.hdf5' %(pttrn), 'w')    # a: append, w:Write
	#out_hdf = h5py.File( out_fname + '_%s.hdf5' %(pttrn), 'a')    # a: append   


	print "\n Data Analysis with LOKI.\n"
	#shots = load_cxi_data(data_file=fname, get_parameters=False, Amplitudes=Ampl_image)*mask_better
	shots *= mask_better ## Add the MASK ##
	#calc_acc(images=shots, cntr=cntr, qmin_pix=args.q_range[0], qmax_pix=args.q_range[1], q_map= q_mapping, mask = mask_better, data_hdf=out_hdf):
	calc_acc(images=shots, cntr=cntr, q_map= q_mapping, mask = mask_better, data_hdf=out_hdf)

##################################################################################################################
##################################################################################################################