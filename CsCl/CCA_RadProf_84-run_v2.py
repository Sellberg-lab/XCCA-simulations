#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import CXI-files from simulations with Condor (v1.0) and Auto-Correlate Data
# and calculate and plot the Radial Profile with LOKI
# cxi-file/s located in the same folder, result saved  in subfolder "/simulation_results_N_X"
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#           '..._data-nabs_...' without taking the modulus in the plot
#            and as a log10 plot '..._log10_...' and are in reciprocal space
#  Projection patterns '..._rs_...' in real space are from inv Fourier transform of data
#              (and FFT-shifted)
#   Patterson_Image '..._patterson_image_...' are from FFTshift-Fast Fourier transforms-FFTshift
#           of Intensity Patterns =  AutoCorrelated image (can be used as initial guess for phae retrieval)
# 2019-02-27  v2 @ Caroline Dahlqvist cldah@kth.se
#			CCA_RadProf_84-run.py
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
#from loki import RingData 	## RingData is a folder ##
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
		#intensity_pattern = np.asarray(f["entry_1/data_1/data"])
		#amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"])
		intensity_pattern = np.asarray(f["entry_1/data_1/data"])
		amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"][0:5])
		#mask = np.asarray(f["entry_1/data_1/mask"])		# Mask + Data ???
		#projection_image =fftshift(fftn(fftshift(amplitudes_pattern)))
				
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


###################### Choose Calculations: ########################################
run_with_MASK = True 			## Run with Intensity_Pattern*Better-Mask_Assembled ##
Ampl_image = False			## The complex images in Amplitude Patterns instead of Intensity Patterns ##
add_noise = False						## Add Generated Noise to Correlations ##
save_polar_param = True # False 		## For Saving Calculated Parameters to the file ##

plot_diffraction = False#		## if False : No plot of Diffractin Patterns ##
random_noise_1quad =False			## Generate random Noise in one quardant ##
calc_RadProfile = True 	## Calculate the radial Profile + Save to File ##
calc_AutoCorr = True	## Auto-correlation between diff. diffraction Patterns ##
plotting = True 		## Plot Subplot of Radial Profile and Auto-Correlation 'ave-corrs' #


##################################################################################################################
##################################################################################################################
#---------------------------- Parameters and Data: --------------------------------------------------------------#
##################################################################################################################
##################################################################################################################

pol_frac = None  ## 'fraction of in-plane polarization' (float)!! ##
#select_name = " "

## ---- Load File Names in Directory and Sort Numerically after Concentraion: ---- ##
sim_res_dir = args.dir_name
fnames = [ os.path.join(sim_res_dir, f) for f in os.listdir(sim_res_dir)
		if f.endswith('cxi') ]
        #if f.startswith('%s'%select_name) and f.endswith('cxi') ]
#fnames = [ os.path.join(sim_res_dir, f) for f in os.listdir(sim_res_dir) if f.endswith('cxi') ]
if not fnames:
	print"\n No filenames for directory %s"%(str(sim_res_dir))
fnames = sorted(fnames, 
	key=lambda x: int(x.split('84-')[-1][6:].split('_ed')[0][0] )) 	## For sorting simulated-file 0-90 ##
#fnames = sorted(fnames, key=lambda x: int(x.split('84-')[-1][4:].split('_ed')[0][0] )) 	## For sorting conc 4-6 M ##

## ----- Fetch Simulation Data: ---- ##
I_list = []  ## or l = np.array([]) used with l=np.append(a) || np.vstack(exposure_diffs):From List-object to array in vertical stack ##
A_list = []
t_load = time.time()
for k in fnames :
	i = list(fnames).index(k) 	## indices ##
	if i==0:		## in the 1st loop: save the simulation parameters (only once sine=ce static) ##
		photon_e_eV, wl_A, ps, dtc_dist, I_list, A_list= load_cxi_data(data_file= k, get_parameters= True, ip= I_list, ap= A_list)
	else: 	 I_list, A_list = load_cxi_data(data_file= k, ip= I_list, ap= A_list)
intensity_pattern = np.asarray(I_list)
amplitudes_pattern = np.asarray(A_list)
t = time.time()-t_load
t_m =int(t)/60
t_s=t-t_m*60
print "\n Loading Time for %i patterns: "%(intensity_pattern.shape[0]), t_m, "min, ", t_s, "s \n" # 5 min, 19.2752921581 s



## ----- Load and Add Mask from Assembly (only 0.0 an 1.0 in data): --- ##
mask_better = np.load("%s/masks/better_mask-assembled.npy" %str(this_dir))
## if data is not NOT binary:  ione =np.where(mask_better < 1.0), mask_better[ione] = 0.0 # Make sure that all data < 1.0 is 0.0 	##  
mask_better = mask_better.astype(int) 	## Convert float to integer ##
print"Dim of the assembled mask: ", mask_better.shape
Ip_w_mb=np.multiply(intensity_pattern,mask_better)	## Intensity Pattern with Mask ##
Ap_w_mb=np.multiply(amplitudes_pattern,mask_better) ## Amplitude Pattern (Complex) with Mask ## 
Patt_w_mb =fftshift(fftn(fftshift(Ip_w_mb))) 		## patterson image || Auto- Correaltions ## 


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
n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
name = fnames[0].split('_84-')[0].split('/')[-1]
N = intensity_pattern.shape[0] 	## the Number of Pattersn loaded ## 
## /.../noisefree_Beam-NarrInt_84-105_6M90_ed_(none-sprd0).cxi


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
if plot_diffraction:
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
		#ax=pypl.imshow(Ip_ma_single_shot, vmin=0.10#ß, cmap='viridis')
		#pypl.title('Intensity Pattern %i: ' %(shot)),pypl.ylabel('y Pixels')#, pypl.xlabel('x Pixels')
		im_ip =ax0.imshow(Ip_ma_shot, vmin=0.10, cmap=color_map ) 
		#ax.set_ylabel('y Pixels')
		ax0.set_title('Intensity Pattern %i: ' %(shot))
		#cb_ip = pypl.colorbar(im_ip, ax=axs[0,i], shrink=cbs, pad= cbp, fraction=0.05, use_gridspec=True) #,label=r'Intensity ', fraction=.1
		cb_ip = pypl.colorbar(im_ip, ax=ax0,fraction=0.046)
		#cb_ip.set_label(r'Intensity ')
		#cb_ip.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
		#cb_ip.update_ticks()

		# ---- Amplitude Pattern : ---- #
		ax1=axs[1,i]
		#pypl.imshow(abs(Ap_w_mb[i]))
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
	## Creaate All Labels and Hide labels and tick labels for top plots and right plots : ##
	#for ax in axs.flat:		ax.set(xlabel='x Pixels', ylabel='y Pixels')
	for ax in axs.flat:		ax.label_outer()
	#pypl.setp(ax2.get_xticklabels(), visible=False)
	#pypl.subplots_adjust(wspace=0.1, hspace=0.2, left=0.04, right=0.92) #, OK=left=0.01(but cb ERR)
	pypl.suptitle("Intensity Pattern vs Amplitude Pattern vs Patterson Image", fontsize=16)
#	#### Save Plot: #### prefix = "subplot_diffraction_Patterson_w_Mask", out_fname = os.path.join( outdir, prefix)
 	#pypl.show()
 	pic_name = '/%s_%s_(%s-%s)_SUBPLOT_I-A-Diffraction_Patterson_w_Mask.%s'%(name,pdb,noisy,n_spread,frmt)
 	pypl.savefig(outdir_raw + pic_name)
 	print "Plot saved in %s \n as %s" %(outdir_raw, pic_name)
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##
##################################################################################################################
##################################################################################################################
#------------------------------------ Loki XCCA: ----------------------------------------------------------------#
##################################################################################################################
##################################################################################################################
print "\n Data Analysis with LOKI.\n"
#from pylab import *	# load all Pylab & Numpy


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
pttrn = "Int"
if Ampl_image:
	mg = abs(Ap_w_mb) ## The complex images in Amplitude Patterns instead of Intensity Patterns ##       
	pttrn = "Ampl"
if add_noise:
	img = Ip_w_mb_w_Ns
	pttrn = "Int-add-noise-%iprc" %(nlevel*100)

# ---- Generate a Storage File for Data/Parameters: ----                                                      
if calc_RadProfile or (plotting and not calc_RadProfile) or save_polar_param:
	data_hdf = h5py.File( out_fname + 'Auto-corr_Radial-Profile_%s.hdf5' %(pttrn), 'a')    # a: append   

##################################################################################################################
##################################################################################################################

#
# ---- Calculate the Radial Profile for all Diffraction Patterns: ----##
#
if calc_RadProfile:
	print "\n Generating Radial Profile Instance..."
	t_RF1 = time.time()

	# ---- RadialProfile - instance: ----
	#r_p = RingData.RadialProfile(cntr, img[0].shape, mask = m_b) #dim of img[0]: (1738, 1742)
	r_p = RadialProfile(cntr, img[0].shape, mask = m_b) #dim of img[0]: (1738, 1742)

	# --- Radial Profile Calculate: ----
	print "\n Starting to Calculate the Radial Profile..."
	rad_pros = np.array( [ r_p.calculate(img[i]) for i in range( N ) ] ) # Rad profile of all

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
	if not save_polar_param and not calc_AutoCorr:
		data_hdf.close()
		print "\n File Closed! \n"

	t = time.time()-t_RF1
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Radial Profile & Calculation Time in LOKI: ", t_m, "min, ", t_s, " s \n" 
##################################################################################################################
##################################################################################################################

#
# ---- Set Radial Parameters for Interpolation in Polar-Conversion : ----
#

## ---- The q-Range to Analyse : ---- ##
#### GDrive/.../Exp_CXI-Martin/scripts/corpairs.py: min = 54; rmax = 1130 #### interesting peaks in Exp. at  200-600 alt 300-500 ## 
qmin_pix, qmax_pix =  args.q_range[0], args.q_range[1] ## [pixels] The Range for evaluating polar plots and Correlations ##

# ---- Single Pixel Resolution at Maximum q [1/A]: -----
nphi = 360#180#
#### alt. (not tested) Try other version from Loki/lcls_scripts/get_polar_data_and_mask: ####
#nphi_flt = 2*np.pi*qmax_pix #[ qmax_pix = interp_rmax in pixels, OBS adjusted edges in original ]
#phibins = qmax_pix - qmin_pix 		# Choose how many, e.g. same as # q_pixls
#phibin_fct = np.ceil( int(nphi_flt)/float(phibins) )
#nphi = int( np.ceil( nphi_flt/phibin_fct )*phibin_fct )
#print "\n nphi: ", nphi 	

# ---- Save g-Mapping (the q- values [qmin_pix,qmin_pix]): ----
qrange_pix = np.arange( qmin_pix, qmax_pix)
q_map = np.array( [ [ind, pix2invang(q)] for ind,q in enumerate( qrange_pix)])

# ---- Interpolater initiate : ----	# fs = fast-scan {x} cntr[0]; Define a polar image with dimensions: (qRmax-qRmin) x nphi
#Interp = RingData.InterpSimple( cntr[0], cntr[1], qmax_pix, qmin_pix, nphi, m_b.shape)
Interp = InterpSimple( cntr[0], cntr[1], qmax_pix, qmin_pix, nphi, m_b.shape)

# ---- Make a Polar Mask : ----
polar_mask = Interp.nearest(m_b, dtype=bool).round() ## .round() returns a floating point number that is a rounded version of the specified number ##


# ---- SAVE the calculated parameters (replace old data if exists) ---- ##
if save_polar_param:
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
	del Interp, cntr # Free up memory
	# ---- Save by closing file: ----
	if  not calc_AutoCorr and not plotting:
		data_hdf.close()
		print "\n File Closed! \n"
##################################################################################################################
##################################################################################################################


#
# ---- Calculate the Auto-Correlation of all Diffraction Patterns: ----##
#
if calc_AutoCorr:
	t_AC = time.time()
	# ---- Calculate the Difference in Intensities and store in a List: ----
	#exposure_diffs_cart =img[:-1]-img[1:]				# Intensity Patterns in Carteesian cor
	#img_mean = np.array([ img[i].mean() for i in range(img.shape[0]) ]) ## Calculate the Mean of each pattern(shot) ##
	#exposure_diffs_cart= img[:-1]/img_mean[:-1]-img[1:]/img_mean[1:]  ## Normalize each pattern(shot) before subtraction ##
	#del img_mean 	## Do not need the mean-values anymore ##
	#exposure_diffs_cart = np.asarray(exposure_diffs_cart) 	# = (4, 1738, 1742)
	#exposure_diffs_cart = polarize(exposure_diffs_cart)  ## Polarize (if no fraction given => just 1s) ##
	## ---- Conv to polar of diff-data : ---- ##
	#exposure_diffs_cart = np.array( [ polar_mask* Interp.nearest(exposure_diffs_cart[i]) for i in range(exposure_diffs_cart.shape[0]) ] ) 

	## ---- Calc diff-data direct from polar Images: ---- ##
	# ---- Generate Polar Image/Images (N Diffracton Patterns) : ----
	#print "\n Starting to Calculate the Polar Images...\n"
	#polar_imgs = np.array( [ polar_mask* Interp.nearest(img[i]) for i in range( N) ] )
	# ---- Normalize - with function: ----
	#polar_imgs_norm =norm_polar_img(polar_imgs, mask_val=0) # Function
	## alt try: polar_imgs_norm = norm_data(polar_imgs)
	#exposure_diffs_pol =polar_imgs_norm[:-1]-polar_imgs_norm[1:]	# Polar Imgs
	#exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)

	
	#print "exposure diff vector's shape", exposure_diffs_cart.shape, ' in polar ", exposure_diffs_pol.shape
	
	## ---- Calc diff-data from Pairs of Images: ---- ##
	exp_diff_pairs = img[:-1:2]-img[1::2]
	exp_diff_pairs = np.asarray(exp_diff_pairs)
	print "\n Starting to Calculate the Polar Images...\n"
	exp_diff_pairs = np.array( [ polar_mask* Interp.nearest(exp_diff_pairs[i]) for i in range(exp_diff_pairs.shape[0]) ] ) 
	
	exposure_diffs = exp_diff_pairs #exposure_diffs_cart #exposure_diffs_pol

	# ---- Autocorrelation of each Pair: ----
	print "\n Starting to Auto-Correlate the Polar Images...\n"
	#acorr = [RingData.DiffCorr( exposure_diffs_cart).autocorr(), RingData.DiffCorr( exposure_diffs_pol ).autocorr()]
	#cor_mean = [RingData.DiffCorr( exposure_diffs_cart).autocorr().mean(0), RingData.DiffCorr( exposure_diffs_pol ).autocorr().mean(0)]
	acorr = DiffCorr( exposure_diffs ).autocorr()
	cor_mean =acorr.mean(0)
	#cor_mean = np.asarray(cor_mean)

	# ---- Save the Calculated Radial Profile to the Storage File for Later Analyse/Plot (replace old data if exists): ---- ##
	if save_polar_param:
		#if 'polar_data_normalized' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data_normalized', data = polar_imgs_norm) 
		#else: 
		#	del data_hdf['polar_data_normalized']
		#	dset = data_hdf.create_dataset('polar_data_normalized', data=polar_imgs_norm)
		#if 'polar_data' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data', data = polar_imgs) 
		#else: 
		#	del data_hdf['polar_data']
		#	dset = data_hdf.create_dataset('polar_data', data=polar_imgs)

		if 'polar_data_normalized' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data_normalized', data = polar_imgs_norm) 
		else: 
			del data_hdf['polar_data_normalized']
			dset = data_hdf.create_dataset('polar_data_normalized', data=polar_imgs_norm)
		if 'polar_data' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data', data = polar_imgs) 
		else: 
			del data_hdf['polar_data']
			dset = data_hdf.create_dataset('polar_data', data=polar_imgs)
			

		if 'auto-correlation_mean' not in data_hdf.keys(): data_hdf.create_dataset( 'auto-correlation_mean', data = np.asarray(cor_mean))
		else: 
			del data_hdf['auto-correlation_mean']
			dset = data_hdf.create_dataset('auto-correlation_mean', data=np.asarray(cor_mean))
		if 'auto-correlation' not in data_hdf.keys(): data_hdf.create_dataset( 'auto-correlation', data = np.asarray(acorr))
		else: 
			del data_hdf['auto-correlation']
			dset = data_hdf.create_dataset('auto-correlation', data=np.asarray(acorr))
	if not plotting:
		# ---- Save by closing file: ----
		data_hdf.close()
		print "\n File Closed! \n"

	t = time.time()-t_AC
	t_m =int(t)/60
	t_s=t-t_m*60
	print "AutoCorrelation Time: ", t_m, "min, ", t_s, " s \n"
##################################################################################################################
##################################################################################################################

#
# ---- Plot the Mean Radial Profile and Auto-Correlation of all Diffraction Patterns: ----##
if plotting:
	if not calc_RadProfile: ## if this is not done, then fetch calculated data from file ##
		assert('radial_profiles' in data_hdf.keys()),("No Radial Profile saved!!")
		rad_pros = np.asarray(data_hdf['radial_profiles'])
		assert('qs_c_for_profile' in data_hdf.keys()),("No qs from center saved!!")
		qs_c = np.asarray(data_hdf['qs_c_for_profile'])
		#assert('qs_for_profile' in data_hdf.keys()),("No qs from edge saved!!")
		#qs = np.asarray(data_hdf['qs_for_profile'])
	if not calc_AutoCorr: ## if this is not done, then fetch calculated data from file ##
		assert('auto-correlation' in data_hdf.keys()),("No Auto-Correlation saved!!")
		acorr = np.asarray(data_hdf['auto-correlation'])
	# ---- Close thee file after loading the data or if it is still open: ----
	data_hdf.close()
	print "\n File Closed! \n"

	rad_pro_mean = rad_pros.mean(0)
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
	fig1 = pypl.figure('RP-AC', figsize=(22,15))
	## subplot , constrained_layout=True 
	pypl.subplot(121)
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
	############################## [fig.a] End ##############################

	pypl.subplot(122)
	cb_shrink, cb_padd = 1.0, 0.2 
 	#### plot as in // GDrive/.../scripts/ave_corrs & 'plot_a_corr.py' #####
	corr = None
	for i in range(acorr.shape[0]):		#acorr.shape = (4, 190, 5)
		if corr is None:	corr = acorr[i]
		else :	corr += acorr[i]		# adter loop cr.shape = (190, 5)
	#print "\n corr shape: ", acorr[ind].shape, "\n & corr : ", corr.shape
	corrsum = corr
	corr_count =np.zeros(1)+(exposure_diffs.shape[0]) 
	tot_corr_count = [ corr.shape[0] ]
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

	corrsum  =np.nan_to_num(corrsum)
	sig = corrsum/corrsum[:,0][:,None]/tot_corr_count[0] #if Polar: RuntimeWarning: invalid value encountered in divide ###### !ERROR !!!
	sig  =np.nan_to_num(sig)
	#print "corrsum: ", corrsum[:,0][:,None] # = [[ 22.39735592] [  0.        ]...
	#print "corr: ", corr[:,0][:,None]
	#print "\n sig dim: ", sig.shape #carteesian = (1738, 1742): polar =(190, 5)

	padC = 10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181
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
		print "q-bins array: ", q_bins, "/n Dim q-bins ", q_bins.shape
		q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins]
		print "q_label: ", q_label

		ytic=ax.get_yticks() # = qrange_pix[q_bins] ## choose the equivalent q pixels from q_min-q_max
		ylab= q_label
		ax.set_yticklabels(q_label)
		ax.set_yticks(ytic)

		# #---- Label xy-Axis: ----
		ax.set_xlabel(r'$\phi$', fontsize = axis_fsize)
		ax.set_ylabel(r'$q \, [\AA^{-1}]$', fontsize =axis_fsize)

	ave_corr_title = ax.set_title(r"Average of %d corrs with limits $\mu \pm 2\sigma$"%(tot_corr_count[0]),  fontsize=sb_size)
	ave_corr_title.set_y(1.08) # 1.1)
	############################## [fig.b] End ##############################
	
	#pypl.set_cmap('jet')
	pypl.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.85, bottom=0.15)  # hspace=0.2,
	## WORSE left=0.01, right=0.99), BETTER left=0.1, right=0.9, top=0.86...bottom 0.10 r label gone
	#pypl.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9) #top=0.86
	#pypl.tight_layout()  ## overrule adjustments ##
	pypl.suptitle("%s_%s_(%s-%s)_[qx-%i_qi-%i_nphi-%i]_w_Mask_%s" %(name,pdb,noisy,n_spread,qmax_pix,qmin_pix, nphi,pttrn), fontsize=sp_size)  # y=1.08, 1.0=too low(overlap radpro title)
	fig_name = "SUBPLOT_Radial-Profile-(from-center)_Diff-Auto-Corr_(qx-%i_qi-%i_nphi-%i)_w_Mask_%s.%s" %(qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	#pypl.show()
	pypl.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
	#pypl.savefig( outdir+ prefix + fig_name)
	print "\n Subplot saved as %s " %fig_name
	del fig_name, rad_pro_mean 	# clear up memory1
	pypl.cla() ## clears axis ##
	pypl.clf() ## clears figure ##

##################################################################################################################
##################################################################################################################
