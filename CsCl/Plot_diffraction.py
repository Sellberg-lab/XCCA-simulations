#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import CXI-files from simulations with Condor (v1.0) and Plot Data
# cxi-file/s located in the same folder, result saved  in subfolder "/simulation_results_N_X"
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#   Patterson_Image '..._patterson_image_...' are from FFTshift-Fast Fourier transforms-FFTshift
#           of Intensity Patterns =  AutoCorrelated image (can be used as initial guess for phae retrieval)
# 2019-03-08   @ Caroline Dahlqvist cldah@kth.se
#			Plot_diffraction.py
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
#from pylab import *	# load all Pylab & Numpy
# %pylab	# code as in Matlab
import os, time
import gc
this_dir = os.path.dirname(os.path.realpath(__file__)) ## Get path of directory
#this_dir = os.path.dirname(os.path.realpath('Plot_diffraction.py')) ##for testing in ipython
if "/home/" in this_dir: #/home/cldah/cldah-scratch/ or /home or 
	os.environ['QT_QPA_PLATFORM']='offscreen'

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Plot Simulated Diffraction Patterns.")

parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
      help="The Location of the directory with the cxi-files (the input).")

parser.add_argument('-o', '--outpath', dest='outpath', default='dir_name', type=str, help="Path for output, Plots and Data-files")

parser.add_argument('-s', '--simulation-number', dest='sim_n', default=None, type=int, 
	   help="The number of the pdb-file, which simulation is to be loaded from 0 to 90, e.g. '20' for the file <name>_4M20_<properties>l.cxi")

#parser.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
#      help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)

args = parser.parse_args()
## ---------------- ARGPARSE  END ----------------- ##

# ------------------------------------------------------------
def load_cxi_data(data_file, ip, ap, get_parameters=False):
	"""
	Reads simulation data from 'data_file' and appends to list-objects

		In:
	data_file           the cxi-file/s from Condor to be read in.
	get_parameters      Determines if the static simulation parameters should be retured, Default = False.
	ip, ap, pi          The intensity pattern, amplitude patterns and patterson image respectively as list, the 
						     loaded vaules from 'data_file' are appended to these lists.
	=====================================================================================
		Out:
	ip, ap, pi          list-objects with the data loaded from 'data_file' appended.
	photon_e_eV         Photon Energy of source in electronVolt [eV], type = int, (if get_parameters = True)
	wl_A                Wavelength in Angstrom [A], type = float, (if get_parameters = True)
	ps                  Pixel Size [um], type = int, (if get_parameters = True)
	dtc_dist            Detector Distance [m], type = float, (if get_parameters = True)

	"""
	with h5py.File(data_file, 'r') as f:
		intensity_pattern = np.asarray(f["entry_1/data_1/data"])
		amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"])
		#intensity_pattern = np.asarray(f["entry_1/data_1/data"][0:50])
		#amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"][0:5])
				
		photon_e_eV = np.asarray(f["source/incident_energy"] )			# [eV]
		photon_e_eV = int(photon_e_eV[0]) 								# typecast 'Dataset' to int

		photon_wavelength = np.asarray(f["source/incident_wavelength"]) #[m]
		photon_wavelength= float(photon_wavelength[0])					# typecast 'Dataset' to float
		wl_A = photon_wavelength*1E+10 									#[A]

		psa = np.asarray(f["detector/pixel_size_um"])  					#[um]
		ps = int(psa[0]) 												# typecast 'Dataset' to int

		dtc_dist_arr = np.asarray(f["detector/detector_dist_m"])		#[m]
		dtc_dist = float(dtc_dist_arr[0]) 								# typecast 'Dataset' to float

	#l = np.array([]) used with l=np.append(a) or l=np.append(l,a,axis=0) || np.vstack(exposure_diffs)##
	#ip = np.append(ip,intensity_pattern, axis=0) ## OBS: np.appen is slow if large sets of data or large loops  ##
	#if ip.size == 0: 	ip, ap, pi = intensity_pattern, amplitudes_pattern, patterson_image
	#	else:	ip, ap, pi = np.append(ip, intensity_pattern, axis=0), np.append(ap, amplitudes_pattern, axis=0), np.append(pi, patterson_image, axis=0) ## ##
	ip.extend(intensity_pattern)
	ap.extend(amplitudes_pattern)

	if get_parameters:
		return photon_e_eV, wl_A, ps, dtc_dist, ip, ap
	else:
		return ip, ap
# -------------------------------------------------------


###################### Choose Calculations: ########################################
#Ampl_image = False			## The complex images in Amplitude Patterns instead of Intensity Patterns ##
#add_noise = False						## Add Generated Noise to Correlations ##

random_noise_1quad =False			## Generate random Noise in one quardant ##

##################################################################################################################
##################################################################################################################
#---------------------------- Parameters and Data: --------------------------------------------------------------#
##################################################################################################################
##################################################################################################################

## ---- Load File Names in Directory and Sort Numerically after Concentraion: ---- ##
sim_res_dir = args.dir_name
fnames = [ os.path.join(sim_res_dir, f) for f in os.listdir(sim_res_dir)
		if f.endswith('cxi') ]
        #if f.startswith('%s'%select_name) and f.endswith('cxi') ]
#fnames = [ os.path.join(sim_res_dir, f) for f in os.listdir(sim_res_dir) if f.endswith('cxi') ]
if not fnames:
	print"\n No filenames for directory %s"%(str(sim_res_dir))
fnames = sorted(fnames, 
	key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0] )) 	## For sorting simulated-file 0-90 ##
#key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0][0] )) 	## For sorting simulated-file 0-90  by 1st number##
#fnames = sorted(fnames, key=lambda x: int(x.split('84-')[-1][4:].split('_ed')[0][0] )) 	## For sorting conc 4-6 M ##
assert (len(fnames)>args.sim_n),("No such File Exists! Simulation number must be less than the number of simulations.")
#print "\n file names : ", fnames
#fname = fnames[args.sim_n] ## Choose only this pdb-files simulation ##
one_file = fnames[args.sim_n] ## Choose only this pdb-files simulation ##
del fnames
gc.collect()
fnames = [one_file ]
print "\n Selected file : ", fnames

# noisefree_Beam-NarrInt_84-105_4M90_ed_(none-sprd0).cxi

## ----- Fetch Simulation Data: ---- ##
I_list = []  ## or l = np.array([]) used with l=np.append(a) || np.vstack(exposure_diffs):From List-object to array in vertical stack ##
A_list = []
t_load = time.time()
for k in fnames :
	#i = list(fnames).index(k) 	## indices ##
	#if i==0:		## in the 1st loop: save the simulation parameters (only once sine=ce static) ##
	#	photon_e_eV, wl_A, ps, dtc_dist, I_list, A_list= load_cxi_data(data_file= k, get_parameters= True, ip= I_list, ap= A_list)
	#else: 	 I_list, A_list = load_cxi_data(data_file= k, ip= I_list, ap= A_list)
	I_list, A_list = load_cxi_data(data_file= k, ip= I_list, ap= A_list)
intensity_pattern = np.asarray(I_list)
amplitudes_pattern = np.asarray(A_list)
N = intensity_pattern.shape[0] 	## the Number of Patterns loaded ## 
del I_list, A_list
gc.collect()
t = time.time()-t_load
t_m =int(t)/60
t_s=t-t_m*60
print "\n Loading Time for %i patterns: "%(N), t_m, "min, ", t_s, "s \n" # 5 min, 19.2752921581 s


## ----- Load and Add Mask from Assembly (only 0.0 an 1.0 in data): --- ##
mask_better = np.load("%s/masks/better_mask-assembled.npy" %str(this_dir))
## if data is not NOT binary:  ione =np.where(mask_better < 1.0), mask_better[ione] = 0.0 # Make sure that all data < 1.0 is 0.0 	##  
mask_better = mask_better.astype(int) 	## Convert float to integer ##
print"Dim of the assembled mask: ", mask_better.shape
Ip_w_mb=np.multiply(intensity_pattern,mask_better)	## Intensity Pattern with Mask ##
Ap_w_mb=np.multiply(amplitudes_pattern,mask_better) ## Amplitude Pattern (Complex) with Mask ## 
Patt_w_mb =fftshift(fftn(fftshift(Ip_w_mb))) 		## patterson image || Auto- Correaltions ## 
del intensity_pattern, amplitudes_pattern
gc.collect()

# ---- Centre Coordiates Retrieved Directly from File ((X,Y): fs = fast-scan {x}, ss = slow-scan {y}): ----
#cntr = np.load("%s/centers/better_cent_lt14.npy" %str(this_dir))	# from exp-file run 84-119
#print "Centre from file: ", cntr ## [881.43426    863.07597243]
#cntr_msk =np.asarray([(mask_better.shape[1]-1)/2.0, (mask_better.shape[0]-1)/2.0 ]) ## (X,Y)
#print "\n Centre from Mask: ", cntr_msk ##[870, 868]; [869.5, 871.5]; [871.5, 869.5]
#cntr= cntr_msk 	## (X,Y) if use center point from the Msak ##
#cntr_int=np.around(cntr).astype(int)  ## (rounds to nearest int) for the center coordinates in pixles as integers ##
#print "\n Centre as Integer: ", cntr_int,  "\n"


# ----- Parameters unique to Simulation: ---- ##
cncntr_start = fnames[0].split('_ed')[0].split('84-')[-1][4:] 	## '...84-105_6M90_ed...'=> '6M90' ##
if len(fnames)>1:
	cncntr_end = fnames[-1].split('_ed')[0].split('84-')[-1][4:]
	pdb=cncntr_start +'-'+cncntr_end
else :	pdb = cncntr_start
run=fnames[0].split('84-')[-1][0:3] 		## 3rd index excluded ##
noisy = fnames[0].split('_ed_(')[-1].split('-sprd')[0]
#n_spread = fnames[0].split('-sprd')[-1].split(')_')[0] ## for files that ends with '_#N.cxi' ##
n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
name = fnames[0].split('_84-')[0].split('/')[-1]
## /.../noisefree_Beam-NarrInt_84-105_6M90_ed_(none-sprd0)_#5.cxi


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

####################################################################################
####################################################################################
#---------------------- Plot Read-in Data n_plts Patterns: ------------------------#
####################################################################################
####################################################################################

pypl.cla() ## clears axis ##
color_map ='jet' #'viridis'
print "\n Plotting some Diffraction Patterns.\n"
#pypl.rcParams["figure.figsize"] = (17,12)
pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'})
n_plts = 3
cbs,cbp =  0.98, 0.02 #0.04, 0.1: left plts cb ocerlap middle fig
selected_shots = np.linspace(0, N, n_plts).astype(int) ## a list of only 4 selected Diffraction Patterns ##
fig, axs = pypl.subplots(3, n_plts, figsize=(15,12), sharey='row', sharex='col')#, constrained_layout=True) ## constrained_layout = label not overlap ##
pypl.subplots_adjust(wspace=0.1, hspace=0.2, left=0.04, right=0.92) #, OK=left=0.01(but cb ERR)
for shot in selected_shots: 			## only plot 4 Patterns ##
	i = list(selected_shots).index(shot) 	## instead of a counter, use the index to number the plots ##
	axs[i,0].set( ylabel='y Pixels') ## Only  y labels on right- hand plots ##
	axs[2,i].set( xlabel='x Pixels') ## Only  y labels on right- hand plots ##
	# ---- Intensity Pattern : ---- #
	ax0=axs[0,i]
	Ip_ma_shot = np.ma.masked_where(mask_better == 0, Ip_w_mb[i]) ## Check the # photons ##
	im_ip =ax0.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
	ax0.set_title('Intensity Pattern %i: ' %(shot))
	cb_ip = pypl.colorbar(im_ip, ax=ax0,fraction=0.046)
	#cb_ip.set_label(r'Intensity ')
	#cb_ip.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	#cb_ip.update_ticks()

	# ---- Amplitude Pattern : ---- #
	ax1=axs[1,i]
	Ap_ma_shot = np.ma.masked_where(mask_better == 0, Ap_w_mb[i]) ## Check the # photons ##
	im_ap=ax1.imshow(abs(Ap_ma_shot), vmin=0.1, cmap=color_map ) #viridis
	ax1.set_title('Amplitude Pattern %i: ' %(shot))
	cb_ap =pypl.colorbar(im_ap, ax=ax1, shrink=cbs, pad= cbp, fraction=0.046,use_gridspec=True) #, label(r'Intensity ')
	#pypl.colorbar(im_ap, shrink=cbs, pad= cbp) #, label(r'Intensity ')
	#cb_ap.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
	#cb_ap.update_ticks()

	# ---- Patterson Image (Aut-correlation) : ---- #
	ax2=axs[2,i]
	im_pi=ax2.imshow(abs(Patt_w_mb[i]), vmin=0.1, cmap=color_map) # Without Mask: pypl.imshow(abs(patterson_image[0]))
	ax2.set_title('Patterson Image %i: ' %(shot))
	cb_ap =pypl.colorbar(im_pi, ax=ax2, shrink=cbs, pad= cbp, fraction=0.046) #, label(r'Intensity ')

	cmap = cm.jet #cm.viridis
	cmap.set_bad('grey',1.) ## (color='k', alpha=None) Set color to be used for masked values. ##
	cmap.set_under('white',1.) ## Set color to be used for low out-of-range values. Requires norm.clip = False ##
	pypl.draw()
for ax in axs.flat:		ax.label_outer()
pypl.suptitle("Intensity Pattern vs Amplitude Pattern vs Patterson Image", fontsize=16)
#### Save Plot: #### prefix = "subplot_diffraction_Patterson_w_Mask", out_fname = os.path.join( outdir, prefix)
#pypl.show()
pic_name = '/%s_%s_(%s-%s)_SUBPLOT_I-A-Diffraction_Patterson_w_Mask.%s'%(name,pdb,noisy,n_spread,frmt)
pypl.savefig(outdir_raw + pic_name)
print "Plot saved in %s \n as %s" %(outdir_raw, pic_name)
pypl.cla() ## clears axis ##
pypl.clf() ## clears figure ##
####################################################################################