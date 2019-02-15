#!/usr/bin/env python2.7
#*************************************************************************
# Import CXI-files from simulations with Condor (v1.0) and Cross-Correlate Data
# cxi-files located in the same folder, result saved  in subfolder "<name>_CCA"
# File format is optiona, indexing start with '0' 
# Currently tetsting with data from simulation of CsCl
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#           '..._data-nabs_...' without taking the modulus in the plot
#            and as a log10 plot '..._log10_...' and are in reciprocal space
#  Projection patterns '..._rs_...' in real space are from inv Fourier transform of data
#              (and FFT-shifted)
#   Patterson_Image '..._patterson_image_...' are from FFTshift-Fast Fourier transforms-FFTshift
#           of Intensity Patterns =  AutoCorrelated image (can be used as initial guess for phae retrieval)
# 2019-02-14 v3 calculate F-Transforms here (not in write file) @ Caroline Dahlqvist cldah@kth.se
#			compatable with test_CsCl_84-X_v6- generated cxi-files
#			With argparser for input from Command Line
# Prometheus path: /Users/Lucia/Documents/KTH/Ex-job_Docs/Simulations_CsCl/test_results/
# lynch path: /Users/lynch/Documents/users/caroline/Simulations_CsCl/test_results/
#*************************************************************************

import argparse
import h5py 
import numpy as np
from numpy.fft import fftn, fftshift # no need to use numpy.fft.fftn/fftshift
import matplotlib.pyplot as pypl
# %pylab	# code as in Matlab
import os, time
this_dir = os.path.dirname(os.path.realpath(__file__)) # Get path of directory

parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Correlations.")

#parser.add_argument('-r', '--run-number', dest='run_numb', required=True, type=str, help="The Name of the Experiment run to Simulate, e.g. '84-119'.")
#parser.add_argument('-r', '--run-number', dest='run_numb', required=True, type=int, help="The Number Assigned to the Experiment run to Simulate, e.g. '119' for '84-119'.")
parser.add_argument('-f', '--fname', dest='sim_name', default='test_mask', type=str, help="The Name of the Simulation.")
parser.add_argument('-pdb','--pdb-name', dest='pdb_name', default='4M0', type=str, help="The Name of the PDB-file to Simulate, e.g. '4M0' without file extension.")

parser.add_argument('-n', '--n-shots', dest='number_of_shots', required=True, type=int, help="The Number of Diffraction Patterns to Simulate.")

parser.add_argument('-dn', '--dtctr-noise', dest='dtct_noise', default=None, type=str, help="The Type of Detector Noise to Simulate: None, 'poisson', 'normal'=gaussian, 'normal_poisson' = gaussian and poisson")
parser.add_argument('-dns', '--dtctr-noise-spread', dest='nose_spread', default=None, type=float, help="The Spread of Detector Noise to Simulate, if Noise is of Gaussian Type, e.g. 0.005000.")

args = parser.parse_args()
# ----	Parameters unique to file: ----
frmt = "eps"
name = args.sim_name #"test_mask"  "test_Pnoise"
pdb= args.pdb_name #"4M0_ed"      # 92 structure-files for each concentration. 
cncntr = pdb.split('M')[0] 	## Find which concentration was used and match to Experiment name ##
assert (cncntr == "4" or cncntr == "6"),("Incorrect concentration of crystals, pdb-file must start with '4' or '6'!")
if cncntr == "4": run = "84-119"
else : run ="84-105"
#run = args.run_numb #"84-119"
#run = "84-%int" %(int(args.run_numb)) #"84-119"
noisy = args.dtct_noise #"none"
if noisy is None:	noisy = "none" # since None is valid input
n_spread = args.nose_spread
if n_spread is None:	n_spread = 0
#rt = 1		# Ratio of particles (if 1: only CsCl loaded, if != 1: mxture with Water-particle)
N = args.number_of_shots #5		# Number of iterations performed = Number of Diffraction Patterns Simulated


# ---- Make an Output Dir: ----
outdir = this_dir +'/%s_%s_%s_(%s-sprd%s)_#%i/' %(name,run,pdb,noisy,n_spread,N)
if not os.path.exists(outdir):
	os.makedirs(outdir)
#prefix = 
#out_fname = os.path.join( outdir, prefix)

# ---- Generate a Storage File for Data/Parameters: ----
#data_hdf = h5py.File( outdir + '_file_c[w-MASK].hdf5', 'w')	# w: Write/create
#data_hdf = h5py.File( outdir + '_file_c[w-MASK].hdf5', 'a')	# a: append
## If generating SUBGROUPL for LOKI and cxiLT14 in the same hdf5-file (file_hdf.create_group("cxiLT14py")) ##
#file_hdf = h5py.File( outdir + '_file_c[w-MASK].hdf5', 'w')	# w: Write/create; a:append			

# ---- Choose What to Run: ----
plot_diffraction =  False #True # if False : No plot
XCCA_Thor = False # False	# if True: run XCCA with Thor-pkg
XCCA_Loki = True #	# if True: run XCCA with Loki-pkg
XCCA_cxiLT14py = False#True#False # True	# if True: run XCCA with A.Martins Scripts

# ----	Read in Data from CXI-file: ----
#data_file = this_dir +'/%s_%s_%s_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.cxi'%(name,run,pdb,rt*10,ps,pxls,noisy,n_spread,N)
data_file =this_dir +'/%s_%s_%s_(%s-sprd%s)_#%i.cxi'%(name,run,pdb,noisy,n_spread,N)
with h5py.File(data_file, 'r') as f:
		intensity_pattern = np.asarray(f["entry_1/data_1/data"])
		amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"])
		mask = np.asarray(f["entry_1/data_1/mask"])		# Mask + Data ???
		patterson_image =fftshift(fftn(fftshift(intensity_pattern)))
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

# fftshift: Shift the zero-frequency component to the center of the spectrum
# np.fft.ifftn Compute the N-dimensional inverse discrete Fourier Transform

# ---- if N not in file-name: ----
#N = intensity_pattern.shape[0]

I_p_w_mask_arr0=np.multiply(intensity_pattern[0],mask[0]) #(1738, 1742) Mask from CXI: a = np.array([[1,2],[3,4]]), b = np.array([[5,6],

#	--- Mask from assembly (only 0.0 an 1.0 in data): ---
mask_better = np.load("%s/../masks/better_mask-assembled.npy" %str(this_dir))
## if data is not NOT binary:  ione =np.where(mask_better < 1.0), mask_better[ione] = 0.0 # Make sure that all data < 1.0 is 0.0 	##  
mask_better = mask_better.astype(int) 	## Convert float to integer ##
#print"Dim of the assembled mask: ", mask_better.shape

Ip_w_mb=np.multiply(intensity_pattern,mask_better)	# Intensity Pattern with Mask
Ap_w_mb=np.multiply(amplitudes_pattern,mask_better) # Amplitude Pattern (Complex) with Mask
Patt_w_mb=np.multiply(patterson_image,mask_better) # Patterson Image (Autocorrelation) with Mask


#	---- Gain from file (divide images with gain?): ----	# Currently not implemented !
gn = np.load("%s/../gain/gain_lr6716_r78.npy" %str(this_dir))	# from exp-file run 84-119
gn[ gn==0] = 1
#img1 = img1*mask / gain
# or polar_gain = Interp.nearest(ass_gain), corr_g = gc.autocorr(), sig_g = corr_g / corr_g[:,0][:,None]

# ---- Centre Coordiates Retrieved Directly from File: ----
## with the dimensions fs = fast-scan {x}, ss = slow-scan {y} ##
cntr = np.load("%s/../centers/better_cent_lt14.npy" %str(this_dir))	# from exp-file run 84-119
print "Centre from file: ", cntr ## [881.43426    863.07597243]
#cntr_msk =[ int((mask_better.shape[1]-1)/2), int((mask_better.shape[0]-1)/2) ] ## IndexError: index 3028466 is out of bounds for axis 1 with size 3027596
#cntr_msk =[ (mask_better.shape[1]+1)/2.0, (mask_better.shape[0]+1)/2.0 ] ## IndexError: index 3028498 is out of bounds for axis 1 with size 3027596
cntr_msk =[ (mask_better.shape[1]-1)/2.0, (mask_better.shape[0]-1)/2.0 ] ## (X,Y)
print "\n Centre from Mask: ", cntr_msk ##[870, 868]; [869.5, 871.5]; [871.5, 869.5]
cntr= cntr_msk 	## if use center point from the Msak ##
####################################################################################
####################################################################################
#------------------------- Plot Read-in Data 1st Pattern: -------------------------#
####################################################################################
####################################################################################
if plot_diffraction:
	pypl.rcParams["figure.figsize"] = (17,12)
	pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'})

	for i in range(N): # Ip_w_mb.shape[0] = N
		pypl.subplot(3,N,i+1) #(N,3, 1+i*3)
		pypl.imshow(Ip_w_mb[i]) # == pypl.imshow(abs(Ap_w_mb[0])**2)
		pypl.ylabel('y Pixels')
		pypl.xlabel('x Pixels')
		pypl.title('Pattern %i: ' %(i+1))

		pypl.subplot(3,N,N+(i+1)) #(N,3, 2+i*3)
		pypl.imshow(abs(Ap_w_mb[i]))
		pypl.ylabel('y Pixels')
		pypl.xlabel('x Pixels')

		pypl.subplot(3,N,(N*2)+(i+1)) #(N,3, 3+i*3)
		pypl.imshow(abs(Patt_w_mb[i])) # Without Mask: pypl.imshow(abs(patterson_image[0]))
		pypl.ylabel('y Pixels')
		pypl.xlabel('x Pixels')

		pypl.suptitle("Intensity Pattern vs Amplitude Pattern vs Patterson Image", fontsize=16)
		pypl.tight_layout() 
	##### Save Plot: #### prefix = "subplot_diffraction_Patterson_w_Mask", out_fname = os.path.join( outdir, prefix)
 	#pic_name = 'subplot_diffraction_Patterson_w_Mask.%s'%(frmt)
 	#pypl.savefig(outdir + pic_name)
 	#print "Plot saved in %s \n as %s" %(outdir, pic_name)
	pypl.show()
# ************************************************************************************************
#pypl.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize':'small', 'ytick.labelsize':'small'})
#pypl.rcParams["figure.figsize"] = (15,7)
# pypl.figure(figsize = (10, 10))
#pypl.subplots_adjust(wspace=0.1, hspace=0.2, left=0.07, right=0.99)
#cb = pypl.colorbar(im1, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)	# work but wrong scale
#cb.set_label(r'Intensity ')
#cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
#cb.update_ticks()
# ************************************************************************************************

####################################################################################
####################################################################################
#------------------------------------ Thor XCCA: ----------------------------------#
####################################################################################
####################################################################################
if XCCA_Thor:	# ???? how to implemwnt without 'shot' class or dector-class (wit grids) ????
	import tables # own records in Python and collections of them (i.e. a table)
	from thor import xray

	# from lynch-file:
	#ss = xray.Shotset.from_xxxx(data_file, mask = none) # require '.shot'-file !!
	#ip = ss.intensity_profile(q_spacing=0.01)
	#pypl.figure(), pypl.plot(ip[:,0], ip[:,1], lw=2), pypl.show()
	#q_values = np.arange(0.001, 0.4, 0.001) # what "rings" to interpolate onto, in inv angstroms
	#r = ss.to_rings(q_values)	#xray.Shotset.to_rings(...)  - class RINGS(object) !!
	#intra = r.correlate_intra(q1, q2)
	#inter = r.correlate_inter(q1, q2)


	#Returns:  rings : thor.xray.Rings:  A Rings instance, containing the simulated shots.
 	#  return cls(q_values, polar_intensities, k, polar_mask=None)
	import condor
	from math import *
	photon = condor.utils.photon.Photon(energy_eV =9500)
	wl = photon.get_wavelength() #[m] wavelength => [m]*1E+10 = [Angstrom]
	k_m = 2.0 * np.pi / wl 	# [1/m] wavenumber		=> [1/m]*(1/1E+10) = [1/Angstrom]
	k = k_m*1E-10	# [1/Angstrom] wavenumber
	num_phi = 360
	q_values =np.arange(0.001, 0.4, 0.001)		# from e.g.
	#pi, pm = self.interpolate_to_polar(q_values, num_phi) #falls on func impl following 5 lines
	polar_intensities_output = []
	#polar_mask = self._implicit_interpolation(q_values, num_phi, polar_intensities_output)
	# ?? _implicit_interpolation -> self.detector._basis_gird.get_grid()	??????????????????????
	if type(polar_intensities_output) == list:
		polar_intensities_output = np.vstack(polar_intensities_output)
	pi, pm = polar_intensities_output, polar_mask
	rings =thor.xray.Rings(q_values= q_values, polar_intensities=pi, k=k, polar_mask = pm)	# self, q_values, polar_intensities, k
	# from thor.utils import Parser %% thor/src/utils.py
	# in math2.py def fft_acf(data):

####################################################################################
####################################################################################
#------------------------------------ Loki XCCA: ----------------------------------#
####################################################################################
####################################################################################
if XCCA_Loki:
	from loki import RingData 	## RingData is a folder
	#from loki.RingData import RadialProfile, InterpSimple
	#from loki.RingData import InterpSimple, DiffCorr
	print "\n Data Analysis with LOKI.\n"

	guessing_c = False	# problem with RadialProfile : division with 0; # ValueError: too many values to unpack ->phis, ring = ringfetch_obj.fetch_a_ring
	from_file_c = True	# Retrieve Centre coordinateds from Exp.-file .npy

	# ---- Generate a Storage File's Prefix: ---- 
	#prefix = '%s_%s_store-param' %(name,pdb)
	prefix = '%s_%s_' %(name,pdb)
	loki_folder = outdir + '/%s_%s_%s_with_LOKI/' %(name,run,pdb)
	if not os.path.exists(loki_folder):
		os.makedirs(loki_folder)
	out_fname = os.path.join( loki_folder, prefix) 	# with directory
	#output_hdf = h5py.File( prefix + '.hdf5', 'w' )	# a - read/write/create, r - write/must-exist

	#print "MAX: ", mask_better.max(), " Min: ", mask_better.min() # = MAX:  1.0  Min:  0.0
	# ---- Mask in float() and not int(): ----- 	#	.shape() = (rows, columns)
	#rows,col = mask_better.shape[0],mask_better.shape[1]		# Rows = Y: 1742, # Columns = X:1738				
	#print"rows ", rows, "&  columns ", col
	#Matrix = [[0 for x in range(col)] for y in range(rows)] 
	#m_b = np.empty(mask_better.shape, dtype=int) 	# Empty Array for generating int(mask)
	#for i in range(rows):		# Rows = Y,  1742
	#	for j in range(col):	# Columns = X, 1738
	#		#m_b[i,j]= int(mask_better[i, j])	
	#		m_b[i][j]= int(mask_better[i][j])
	#print "MAX(int): ", m_b.max(), " Min(int): ", m_b.min()
	#m_b = mask_better.astype(int)	# instead of 2D-loop: use numpy.ndarray.astype to type-cast to int
	m_b = mask_better 	# if converting to int directly after load()
	#print "MAX(int): ", m_b.max(), " Min(int): ", m_b.min()

	#############################################################################################
	##### ----- Estimate CENTRE (where beam hits) with Ring Fit : ---- ##########################
	#############################################################################################	
	if guessing_c:	# Guess where the Forward beam hits the detector (last dimension is fast, horizontal for 2D, np.Array)
		print "\n Start Running 'Guessing Centre of Detector'.\n"

		# ---- Generate a Storage File for Guessing centre: ----
		#data_hdf = h5py.File( prefix + 'guess_c.hdf5', 'a')	# a : read/write/create, r+ : write/must-exist
		data_hdf = h5py.File( out_fname + '_guess_c.hdf5', 'a')	# a (append): read/write/create, r+ : write/must-exist
		
		img = intensity_pattern[0] 	# Only look at 1st Pattern

		########### // from loki/examples/exmaple.py #########
		cent_guess = ( 2+img.shape[1] / 2., 1+img.shape[0] / 2. )	# (X, Y)

		# ----- Approximate Radial Profile: ----
		t_RF1 = time.time()	# time instance for timing process
		rad_prof_obj = RingData.RadialProfile(cent_guess, img_shape =img.shape, mask = m_b) 	## Require Mask in integer ! 	##
		
		# ---- If profile is already calculated; save time and load it for plotting: ----
		if 'radial_profile' not in data_hdf.keys(): ## test_rp = data_hdf['radial_profile'], if test_rp == None :
			# ---- 1-D radial profile of intensity_pattern[0]: ----
			radial_profile = rad_prof_obj.calculate(img)
			#output_hdf.create_dataset( 'radial_profile', data = radial_profile)
			data_hdf.create_dataset( 'radial_profile', data = radial_profile)
		else : radial_profile = data_hdf['radial_profile']	# if Profile is already generated

		t = time.time()-t_RF1
		t_m =int(t)/60
		t_s=t-t_m*60
		print "Radial Profile #1 & Calculation Time Time: ", t_m, "min, ", t_s, " s\n"
		
		###################### Choose PLOTs: ################### 
		plot_f1 = False		# Plot Radial Profile
		plot_f2 = False 	# Plot intensity profile with estimated circle
		plot_f3 = False 	# Plot updated centre
		plot_f4 = True 		# Plot photon-counts (y-axis) vs phi (x-axis)

		blue = '#348ABD'
		red = '#E24A33'
		###################### start PLOT f1: ################### 
		if plot_f1:
			fig_G1=pypl.figure(1)
			ax=pypl.gca()
			ax.tick_params(which='both', labelsize =12, length =0)
			ax.plot(radial_profile, marker='s', color='Darkorange')
			#ax.plot( 875, 0, ms=10, marker='s', color='red', label='old center')
			#pypl.axvline(x=875) # cent_guess ~875 ?
			ax.vlines(x=cent_guess[0], ymin=0.0, ymax=1.0, color='b', label='guessed center')
			ax.set_xlabel('Radial Pixel Unit', fontsize =12)
			ax.set_ylabel('CCD Count', fontsize =12, labelpad = 15)
			#ax.set_xlim(10,130)
			#ax.set_ylim(0,np.max(radial_profile))
			ax.set_ylim(0,8E-10)
			ax.grid(lw=1, alpha=0.5, color='#777777', ls='--')
			# ax.set_axis_bgcolor('w') #AttributeError: 'AxesSubplot' object has no attribute 'set_axis_bgcolor' : plt.style.use('ggplot')
			pypl.show()
			#### Need figure -handle, show plot and then save after closing: ###
			#fig_G1.savefig(this_dir +'/%s_84-119_%s_(%s-sprd%s)_#%i/.%s' %(name,pdb,noisy,n_spread,N,frmt))
			#fig_G1.close()
			# save
		###################### end PLOT f1: ################### 
		
		# New param to optimize: (centre pixle ~875)
		peak_radii_gss = 30#20 #500 #960 		# From Plot -  Choose a nice looking Peak; edge-edge
		ring_param_guess = ( cent_guess[0], cent_guess[1], 	peak_radii_gss )
		RF = RingData.RingFit(img)
		x_cen, y_cen, peak_radii = RF.fit_circle_slow(ring_param_guess, 
													ring_scan_width =4, 
													center_scan_width =4, 
													resolution = .1)	# {peak_radii = q_ring in reciprocal-space}
		
		###################### start PLOT f2: ################### 
		if plot_f2 :
			fig_g2 = pypl.figure(2)
			#pypl.subplot(121)	# subplot(122) with img[low:high, low:high] and xy=(x_cen-low, y_cen-low)
			ax=pypl.gca()
			ax.set_xticks( [] )
			ax.set_yticks( [] )
			ax.imshow(img, cmap = 'hot', interpolation= 'nearest')
			circ = pypl.Circle( xy =(x_cen, y_cen), radius= peak_radii, ec='c', fc='none', lw=2, ls='dashed')
			ax.add_patch(circ)
			# gca().add_patch(Circle(xy=(x_cen, y_cen), radius = peak_radi, ec='k', fc='none', lw=2))
			pypl.show()
		###################### end PLOT f2: ###################

		# ---- Update tthe Radial Profile obj with better Center: ----
		t_RF2 = time.time() # time instance for timing process
		rad_prof_obj.update_center((x_cen, y_cen))
		
		# ---- If Redined profile is already calculated; save time and load it for plotting: ----
		if 'Refined_radial_profile' not in data_hdf.keys(): ## test_rp = data_hdf['radial_profile'], if test_rp == None :
			# ---- 1-D radial profile of intensity_pattern[0]: ----
			refined_radial_profile = rad_prof_obj.calculate(img)
			#output_hdf.create_dataset( 'Refined_radial_profile', data = refined_radial_profile) # also possible: f_out.create_group(model_name)
			data_hdf.create_dataset( 'Refined_radial_profile', data = refined_radial_profile) # also possible: f_out.create_group(model_name)
			data_hdf.create_dataset( 'q_ring', data = peak_radii, dtype='i8') # dtype='i1', 'i8'f16';also possible: f_out.create_group(model_name)
		else : refined_radial_profile = data_hdf['Refined_radial_profile']	# if Profile is already generated
		
		t = time.time()-t_RF2
		t_m =int(t)/60
		t_s=t-t_m*60
		print "Radial Profile #2 & Calculation Time Time: ", t_m, "min, ", t_s, " s\n"

		###################### start PLOT f3: [old and new center] ################### 
		if plot_f3:
			fig_g3 = pypl.figure(3)
			ax = pypl.gca()
			ax.tick_params(which='both', labelsize =12, length =0)
			ax.plot( radial_profile, ms=7, marker='s', color='Darkorange', label='old center')
			ax.plot( refined_radial_profile, ms=7, marker= 'o', color='blue', label='new center', alpha=.9)
			ax.set_xlabel( 'Radial Pixel Units', fontsize =12)
			ax.set_ylabel( 'CCD Count', fontsize=12, labelpad=15)
			ax.grid(lw=1, alpha=0.5, color='#777777', ls='--' )
			leg =ax.legend() 
			fr = leg.get_frame() 		# fr.set_facecolor('w')
			fr.set_alpha(.7)			# ax.set_axis_bgcolor('w')
			yl,yh,xl,xh = 4500, 9500,0,1000	# from e.q.
			ax.set_ylim(yl,yh)
			ax.set_xlim(xl,xh)
			pypl.show()
		###################### end PLOT f3: ################### 
		data_hdf.close()		# when code below fails the file isn't closed
		data_hdf = h5py.File( out_fname + '_guess_c(p4).hdf5', 'a')

		# ---- Radius-of-Interest from the Radial Profile
		interesting_peak_radius = 30#100#900 #60#100	# [radial Pixel Units] test a value 
		del radial_profile
		del refined_radial_profile
		# ---- Make the CCD -> Photon Conversion Factor: ----
		gn = np.load("%s/../gain/gain_lr6716_r78.npy" %str(this_dir))	# from exp-file run 84-119
		ph_e, gain=9500, np.mean(gn) 	# photon_energy [eV]= 12398.42 / wavelen [Angstrom]  # electron-Volts
		pcf = np.abs(gain)*3.65/ph_e 	# Photon Conversion Factor from absolute gain and Photon Energy
		#Ringfetch = RingData.RingFetch(x_cen, y_cen, img, mask= m_b, q_resolution = 0.05, #from.e.g.
		#							phi_resolution=1.25, wavelen=wl_A, pixelsize=ps*1E-6, detdist =dtc_dist, 
		#							photon_conversion_factor=pcf)
		#phis, ring = RingFetch.fetch_a_ring( interesting_peak_radius) 
		#data_hdf.create_dataset( 'Ring-100_phis-Angles', data = phis) # also possible: f_out.create_group(model_name)
		#data_hdf.create_dataset( 'Ring-100_photon-counts', data = ring) # also possible: f_out.create_group(model_name)
		
		# ---- If RingFetch is already calculated; save time and load it for plotting: ----
		if 'Ring_phis-Angles_ipr%s' %(interesting_peak_radius) not in data_hdf.keys(): ## test_rp = data_hdf['radial_profile'], if test_rp == None :
			# ---- Fetch a Ring: ----	#from.e.g. example.py q_res= 0.05 ERR => try 2x res in Angstron for 2x pixelsize {=0.02}, alt 0.5 AA {=0.1}
			#ringfetch_obj = RingData.RingFetch(x_cen, y_cen, img_shape = img.shape, img =img, mask= m_b, q_resolution = 0.02, 
			#                        phi_resolution=1.25*2, wavelen=wl_A, pixsize=ps*1E-6, detdist =dtc_dist, 
			#                        photon_conversion_factor=pcf)	# NOT TRIED the VERSION BELOW:
			ringfetch_obj = RingData.RingFetch(x_cen, y_cen, img_shape = img.shape, img =img, mask= m_b, q_resolution = 0.1, 
			                        phi_resolution=1.25*0.5, wavelen=wl_A, pixsize=ps*1E-6, detdist =dtc_dist, 
			                        photon_conversion_factor=pcf)
			# where: def __init__(self, a, b, img_shape=None, img=None, mask=None, q_resolution=0.05,phi_resolution=0.5, wavelen=None, pixsize=None,detdist=None, photon_conversion_factor=1,
			#     			interp_method='floor', index_query_fname=None):
			############## PROBLEM !!! ########################## PROBLEM !!! ##################################
			phis, ring = ringfetch_obj.fetch_a_ring( radius =interesting_peak_radius) # ValueError: too many values to unpack
			# where: def fetch_a_ring(self, q=None, radius=None, solid_angle=True):
			#			`q` is moementum transfer magnitude of ring in inverse angstroms
			# 			`radius` is radius of ring in pixel units
			#		rmin || rmax from qmin || qmax from radius and q_resolution [inverse Angstrom]; 0.05 for pxs-size 50um=> 4E-5/um
			#													phi_resolution : 1.25 for g-res 0.05; default 0.5
			data_hdf.create_dataset( 'Ring_phis-Angles_ipr%s' %(interesting_peak_radius), data = phis) # also possible: f_out.create_group(model_name)
			data_hdf.create_dataset( 'Ring_photon-counts_ipr%s' %(interesting_peak_radius), data = ring) # also possible: f_out.create_group(model_name)
		else : 
			phis = data_hdf['Ring_phis-Angles_ipr%s' %(interesting_peak_radius)]	# if Profile is already generated
			ring = data_hdf['Ring_photon-counts_ipr%s' %(interesting_peak_radius)]	# if Profile is already generated

		###################### start PLOT f4: ###################
		if plot_f4:
			fig_g4 = pypl.figure(4)
			pypl.plot( phis, ring , lw=3, color=blue )
			ax = pypl.gca()
			ax.tick_params(which='both', labelsize=12, length=0)
			ax.set_xlabel(r'$\phi\,\,(0-2\pi)$', fontsize=12, labelpad=10 )
			ax.set_ylabel(r'Photon counts', fontsize=12, labelpad=15)
			ax.grid(lw=1, alpha=0.5, color='#777777', ls='--' )
		#ax.set_axis_bgcolor('w')
			ax.set_xlim(0,2*np.pi)	# 2 pi limit
			pypl.show()
		###################### end PLOT f4: ################### 
		del phis, ring, Ringfetch

		# ----  save/Close the Storage - FIles: ----	
		#output_hdf.close()
		data_hdf.close()
		

	#############################################################################################
	##### -----  CENTRE (where beam hits) from Experiment : ---- ################################
	#############################################################################################
	if from_file_c:
		from pylab import *	# load all Pylab & Numpy
		from loki.RingData import DiffCorr, InterpSimple
		#from RingData import RingFit 	#if search centre manually
		print "\n Start Running 'Centre of Detector Loaded From File'.\n"
		# --------------------------------------
		def norm_polar_img(imgs, mask_val=-1):  #from /SACLA tutorial
			"""
			Normalise the polar image to facilitat comparision with other images/shots/simulaions    	
			"""
			norms = ma.masked_equal(imgs, mask_val).mean(axis=2)	#Mask an array where equal to a given value.
			# norms = np.ma.masked_equal(imgs, mask_val).mean(axis=2)

			#print "normalsation martix: \n", norms, "\n"
			#norms =np.nan_to_num(norms) #to Avoid {RuntimeWarning: invalid value encountered in divide}
			#print "normalsation martix after mam_to_nums: \n", norms

			#imgs /= norms[:,:,None] #RuntimeWarning: invalid value encountered in divide => fix np.nan_to_num(...)
			## instead try:	##												##	  !! NOT TRIED YET !!
			print "norm-type", type(norms), 
			print "imgs-type", type(imgs),															
			#imgs = np.divide(imgs, norms[:,:,None], out=np.zeros_like(imgs), where=norms[:,:,None]!=0)
			## TypeError: ufunc 'divide' output (typecode 'd') could not be coerced to provided output parameter (typecode 'l') according to the casting rule ''same_kind''
			imgs = np.true_divide(imgs, norms[:,:,None], out=np.zeros_like(imgs), where=norms[:,:,None]!=0, casting='unsafe')
			## RuntimeWarning: invalid value encountered in true_divide
			## alt. 	##													## 		!! NOT TRIED YET !!
			#def div_nonzero(a, b): /	# from /cxiLT14py/analysisTools/radial_profile.py
    		#	m = b != 0		## m = indices for when b is nonzero
    		#	c = np.zeros_like(a)
    		#	c[m] = a[m] / b[m].astype(a.dtype)
    		#	return c
    		
			
			#imgs[ imgs< 0 ] = mask_val	#set to -1 or 0 if mask_val =0: #RuntimeWarning: invalid value encountered in less
			ineg = np.where( imgs < 0.0 )	# if float: 0.0
			imgs[ineg] =  mask_val # 0.0
			return imgs
		# -------------------------------------- ##if SACLA dsnt work use:
		def norm_data(data): 	# from Loki/lcls_scripts/get_polar_data
			"""
			Normliaze the numpy array data, 1-D numpy array
			"""
			data2 = data- data.min()
			return data2/ data2.max()
		# --------------------------------------

		# ---- Some Useful Functions : ----/t/ from Sacla-tutorial
		pix2invang = lambda qpix : sin(arctan(qpix*(ps*1E-6)/dtc_dist )/2)*4*pi/wl_A
		#pix2invang = lambda qpix : sin(arctan(qpix*(ps*1E-6)/dtc_dist ))*4*pi/wl_A # no div2 {atan yields right triangle of h=qpix*pixel_size & det_dist along zentral nominal axis => half-angle in plane
		invang2pix = lambda qia : tan(2*arcsin(qia*wl_A/4/pi))*dtc_dist/(ps*1E-6)

		###################### Choose Calculations: ###################
		run_with_MASK = True 			# Run with Intensity_Pattern*Better-Mask_Assembled

		calc_RadProfile = False#True 		# Calculate the radial Profile + Save to File
		plot_RadProfile = False#True
		plot_polar_img = False 			# Plot 1st Diffraction Patterns Polar Image [Polar Data(normalised), Polar Data(un-norm), Mask, phi-bins, q-map]
		save_polar_param = False 		# For Saving Calculated Parameters to the file

		if N == 1: calc_AutoCorr = False	# Auto-correlation between diff. diffraction Patterns
		else:  calc_AutoCorr = True 		# only for N > 1 (ense no pattern to compare to)

		calc_CrossCorr = False #False 		# Calculate the Cross-Correlation (? HOW chooose g?? )

		# ---- Generate a Storage File for Data/Parameters: ----
		#data_hdf = h5py.File( prefix + 'file_c.hdf5', 'a')	# a : read/write/create, r+ : write/must-exist
		#data_hdf = h5py.File( out_fname + '_file_c.hdf5', 'a') 	# Img Without MASK
		#data_hdf = h5py.File( out_fname + '_file_c[w-MASK].hdf5', 'w')	# w: Write/create
		data_hdf = h5py.File( out_fname + '_file_c[w-MASK].hdf5', 'a')	# a: append
		#data_hdf = file_hdf.create_group("LOKI")

		if run_with_MASK: img = Ip_w_mb #or amplitudes: Ap_w_mb # All Patterns with Better-Mask-Assembled
		else: img = intensity_pattern 	# All Patterns saved, give shorter name for simplicity

		# ---- Centre Coordiates Retrieved Directly from File (can also be loaded globally with mask and gain): ----
		## with the dimensions fs = fast-scan {x}, ss = slow-scan {y} ##
		#cntr = np.load("%s/../centers/better_cent_lt14.npy" %str(this_dir))	# from exp-file run 84-119
		#gn = np.load("%s/../gain/.npy" %str(this_dir))	# from exp-file run 84-119
		#print "\n central coordinates : ", cntr, " and as int: ", cntr.astype(int)
		# 	= central coordinates :  [881.43426    863.07597243]  and as int:  [881 863]


		# ---- Calculate the Radial Profile for the 1st Diffraction Pattern: ----
		if calc_RadProfile:
			print "\n Generating Radial Profile Instance..."
			t_RF1 = time.time()

			####### // from get_polar_data.py: ########
			# ---- RadialProfile - instance: ----
			r_p = RingData.RadialProfile(cntr, img[0].shape, mask = m_b) #dim of img[0]: (1738, 1742)
			
			# ---- WAX -parameters: ----
			#w_min, w_max = 50, 1500  	# Pixel Units

			# --- Radial Profile Calculate: ----
			#rad_pro = r_p.calculate(img[0])[w_min:w_max] # WAXS for hit finding
			print "\n Starting to Calculate the Radial Profile..."
			#rad_pro = r_p.calculate(img[0])			# Rad profile of 1st Pattern
			rad_pros = np.array( [ r_p.calculate(img[i]) for i in range( N ) ] ) # Rad profile of all
			rad_pro = rad_pros[0]

			# ---- Calculate Q's [1/A] for using in plotting: ----
			radii = np.arange( rad_pro.shape[0])	# from GDrive: CXI-2018_MArting/scripts/plot_wax2
			radii_c = np.arange( rad_pro.shape[0]/2) 	## For plotting from center of detector and not the edge ##
			qs = 4.*np.pi*np.sin(np.arctan(radii*(ps*1E-6)/dtc_dist)*.5)/wl_A # ps[um]= (ps*1E-6)[m]; dtc_dist[m]: wl_A[A]	
			qs_c = 4.*np.pi*np.sin(np.arctan(radii_c*(ps*1E-6)/dtc_dist)*.5)/wl_A  ## For plotting from center of detector and not the edge ##
			#smoothing_factor = 50, #req:from loki.utils.postproc_helper import smooth, bin_ndarray
			#smooth_pro = smooth(rad_pro, beta = smoothing_factor) # ... check max ...
			####### // from get_polar_data.py:  end ########
			
			if 'radial_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'radial_profile', data = rad_pro)
			if 'radial_profiles' not in data_hdf.keys(): data_hdf.create_dataset( 'radial_profiles', data = rad_pros)
			if 'qs_for_profile' not in data_hdf.keys(): data_hdf.create_dataset( 'qs_for_profile', data = qs)
			del r_p 	# clear up memory
			t = time.time()-t_RF1
			t_m =int(t)/60
			t_s=t-t_m*60
			print "\n Radial Profile & Calculation Time in LOKI: ", t_m, "min, ", t_s, " s \n" # @Lynch: 38 min, @Prometheus:

		########## PLOT WAXS [fig.1 & fig.2]: (from GDrive: CXI-2018_MArting/scripts/plot_wax2)########## 
		if plot_RadProfile:
			if not calc_RadProfile:
				assert('radial_profile' in data_hdf.keys()),("No Radial Profile saved!!")
				rad_pro = data_hdf['radial_profile']
			import pylab as plt 	# actually not necessary since pylab was loaded before
			plt.figure(1, figsize=(15,10))
			plt.subplot(221)
			#plt.plot( qs, rad_pro, lw=2)
			#plt.plot( qs[int(rad_pro.shape[0]/2):-1], rad_pro[int(rad_pro.shape[0]/2):-1], lw=2)
			#for i in range(N) : plt.plot(qs_c, rad_pros[i][int(cntr[1]):-600], lw=2, label='Pattern #%i' %i) ## ... ##
			for i in range(N) : plt.plot(qs_c[:rad_pros[i][int(rad_pro.shape[0]/2):-600].shape[0]], rad_pros[i][int(rad_pro.shape[0]/2):-600], lw=2, label='Pattern #%i' %i) ## ... ##
			plt.legend()
			plt.title('%s_%s_%s_(%s-sprd%s)_from-center_#%i' %(name,run,pdb,noisy,n_spread,N))
			plt.ylabel("Mean ADU", fontsize=20)
			plt.xlabel("Q $(\AA^{-1})$", fontsize=20) ## ERR do not start at 0-> must FIX !! ##
			plt.gca().tick_params(labelsize=15, length=9)
			
			plt.subplot(222)
			#plt.plot( rad_pro, lw=2) 
			#plt.plot( rad_pro[int(rad_pro.shape[0]/2):-1], lw=2) # Only plot right of the center pixel
			#for i in range(N):	plt.plot( rad_pros[i][int(rad_pro.shape[0]/2):-1], lw=2,  label='Pattern #%i' %i)
			#for i in range(N):	plt.plot( rad_pros[i][int(cntr[1]):-600], lw=2,  label='Pattern #%i' %i)
			for i in range(N):	plt.plot( rad_pros[i][int(rad_pro.shape[0]/2):-600], lw=2,  label='Pattern #%i' %i)
			# or ? plt.plot( np.transpose(rad_pros[i][int(rad_pro.shape[0]/2):-1]), lw=2,  label='Pattern #%i' %i)
			plt.legend()
			plt.title('%s_%s_%s_(%s-sprd%s)_from-center_#%i' %(name,run,pdb,noisy,n_spread,N))
			plt.ylabel("Mean ADU", fontsize=20)
			plt.xlabel("r (pixels)", fontsize=20)
			plt.gca().tick_params(labelsize=15, length=9)

			#### ----	PLOT THE MEAN RADIAL PROFILE: ----	####	
			rad_pro_mean = rad_pros.mean(0)
			plt.subplot(223)
			plt.plot(qs_c[:rad_pro_mean[int(rad_pro_mean.shape[0]/2):-600].shape[0]], rad_pro_mean[int(rad_pro.shape[0]/2):-600], lw=2, label='Pattern #%i' %i) ## ... ##
			plt.title('%s_%s_%s_(%s-sprd%s)_from-center_#%i\n' %(name,run,pdb,noisy,n_spread,N))
			plt.ylabel("Mean ADU", fontsize=20)
			plt.xlabel("Q $(\AA^{-1})$", fontsize=20) ## ERR do not start at 0-> must FIX !! ##
			plt.gca().tick_params(labelsize=15, length=9)
			
			plt.subplot(224)
			plt.plot( rad_pro_mean[int(rad_pro.shape[0]/2):-600], lw=2,  label='Pattern #%i' %i)
			plt.title('%s_%s_%s_(%s-sprd%s)_from-center_#%i\n' %(name,run,pdb,noisy,n_spread,N))
			plt.ylabel("Mean ADU", fontsize=20)
			plt.xlabel("r (pixels)", fontsize=20)
			plt.gca().tick_params(labelsize=15, length=9)

			pypl.subplots_adjust(wspace=0.4, hspace=0.4, left=0.07, right=0.99)
			pypl.tight_layout() 
			#plt.show()
			fig_name = "Figure_1_Radial-Profile-SUBPLOT_(from-center)_w_Mask.%s" %(frmt)
			pypl.savefig( out_fname + fig_name)
			print "\n Subplot saved as %s " %fig_name
			del fig_name

			#print "shape/dim of radial_profile: ", rad_pro.shape 	## (1800,)
			radial_dim = rad_pro.shape
			del rad_pro 	# clear up memory

			###################################

			plt.figure(2, figsize=(15,8))
			plt.subplot(121)
			for i in range(N) : plt.plot(qs, rad_pros[i], lw=2, label='Pattern #%i' %i) ## ... ##
			plt.legend()
			plt.title('%s_%s_%s_(%s-sprd%s)_from-edge_#%i\n' %(name,run,pdb,noisy,n_spread,N))
			plt.ylabel("Mean ADU", fontsize=20)
			plt.xlabel("Q $(\AA^{-1})$", fontsize=20) ## ERR do not start at 0-> must FIX !! ##
			plt.gca().tick_params(labelsize=15, length=9)
			
			plt.subplot(122)
			for i in range(N):	plt.plot( rad_pros[i], lw=2,  label='Pattern #%i' %i)
			plt.legend()
			plt.title('%s_%s_%s_(%s-sprd%s)_from-edge_#%i\n' %(name,run,pdb,noisy,n_spread,N))
			plt.ylabel("Mean ADU", fontsize=20)
			plt.xlabel("r (pixels)", fontsize=20)
			plt.gca().tick_params(labelsize=15, length=9)
			#plt.show()
			fig_name = "Figure_2_Radial-Profile-SUBPLOT_(from-edge)_w_Mask.%s" %(frmt)
			#pypl.savefig( out_fname + fig_name)
			#print "\n Subplot saved as %s " %fig_name
			del fig_name

			#data_hdf.close()
			#print "\n File Closed! \n"

		#######################################################################################
		#data_hdf = h5py.File( out_fname + '_file_c[w-MASK].hdf5', 'a')

		# ---- Set Radial Parameters for Interpolation in Polar-Conversion : ----
		#### Crystal Dim from CsCL-PDB-files ~33x33x36 Angstrom => 0.03x0.03x0.028 [1/A]
		#q_min, q_max = 2.64, 2.71 		# [1/A]  //test values from Sacla-tutorial for Gold nanoparticle
		#nphi =  int( 2 * pi * q_max ) 	# //test values from Sacla-tutorial for Gold nanoparticle
		# Radial Plot l440-455: Q[1/A]: 6.8046(6.80456)-6.8056; r[pxls]: 863-1160; with pxls on diagonal
		#if !calc_RadProfile & ('radial_profile' in data_hdf.keys()): rad_pro = data_hdf['radial_profil']
		#else: 
		if not calc_RadProfile and not run_with_MASK: 
			assert('radial_profile' in data_hdf.keys()),("No Radial Profile saved!!")
			rad_pro = data_hdf['radial_profile']
			radial_dim = rad_pro.shape
			del rad_pro 	# clear up memory
		
		#rad_cent =int(radial_dim[0]/2) 	# Estimate the center point of the Radial (diagonal) Profile
		
		# ----- Run Without MASK or With MASK {values estimated from Radial Profile Plot}: ----
		if not run_with_MASK:
			rad_cent =int(radial_dim[0]/2) 	# Estimate the center point of the Radial (diagonal) Profile
			qmin_pix = 900-rad_cent #890-rad_cent#870-rad_cent #864-rad_cent #5 #invang2pix ( q_min ), q from origin{center-pixel}
			qmax_pix = 1135-rad_cent#1145-rad_cent #1159-rad_cent #700#invang2pix ( q_max ) # 900 : IndexError: index 3028543 is out of bounds for axis 1 with size 3027596 for polar_mask = Interp.nearest(m_b, dtype=bool).round()
		else: 
			#### GDrive/.../Exp_CXI-Martin/scripts/corpairs.py: min = 54; rmax = 1130 ####
			rmin, rmax = 300, 500 ## interesting peaks in Exp. at  200-600 alt 300-500## 
			qmin_pix = rmin#60#50# 	## q from origin{center-pixel} estimated from radial Profile #
			qmax_pix = rmax# 250#150 #250

		# ---- Single Pixel Resolution at Maximum q [1/A]: -----
		#nphi =   int( 2 * pi * pix2invang(qmax_pix) ) 	# single pixel resolution at maximum q
		nphi = 180#360#180##90#
		print "\n nphi: ", nphi 	# = ,  20 (qmx=1135),  5(qmx=250)
		#### alt. (not tested) Try other version from Loki/lcls_scripts/get_polar_data_and_mask: ####
		#nphi_flt = 2*np.pi*qmax_pix #[ qmax_pix = interp_rmax in pixels, OBS adjusted edges in original ]
		#phibins = qmax_pix - qmin_pix 		# Choose how many, e.g. same as # q_pixls
		#phibin_fct = np.ceil( int(nphi_flt)/float(phibins) )
		#nphi = int( np.ceil( nphi_flt/phibin_fct )*phibin_fct )

		# ----- nphi (ANGULAR) Tic Parameters for choosen Pixel Resolution (for PLOTTING): ----
		#nphi_bins = [nphi/4, nphi/2, 3*nphi/4]
		nphi_bins = [nphi/8.0, nphi/4.0, nphi/2.0, 3*nphi/4.0, 7*nphi/8.0]	# 	# [nphi/4, nphi/2, 3*nphi/4]
		nphi_bins_5 = [nphi/5.0, 2*nphi/5.0, 3*nphi/5.0, 4*nphi/5.0, 5*nphi/5.0 ]	# div 5
		#nphi_label = [ r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
		nphi_label = [r"$\pi/4$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$7\pi/4$" ]	# center of columen for full Period 2pi, # [ r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
		nphi_label_5 = [r"$2\pi/5$", r"$2*2\pi/5$", r"$3*2\pi/5$", r"$4*2\pi/5$", r"$2\pi$" ]	# div 5
		#print "\n pix inversion: ", pix2invang(qmax_pix) # 0.5271935131739763
		#print "\n ps = %i [um], dtc_dist = %f [m], wavelength = %f [A]" %(ps, dtc_dist, wl_A) # = ps = 110 [um], dtc_dist = 0.150000 [m], wavelength = 1.305097 [A]
	
		# ---- Save g-Mapping (the q- values [qmin_pix,qmin_pix]): ----
		#qrange_pix = np.arange( qmin_pix, qmax_pix)
		g_diff = qmax_pix -qmin_pix
		#q_step = 5	# Define how many vaues between min and max ? Required?? (not in Sacla tutorial) 
		#qrange_pix = np.arange( qmin_pix, qmax_pix, g_diff/q_step)
		qrange_pix = np.arange( qmin_pix, qmax_pix)
		q_map = np.array( [ [ind, pix2invang(q)] for ind,q in enumerate( qrange_pix)])

		# ----- q (RADIAL) Tic Parameters for choosen Pixel Resolution (for PLOTTING): ----
		#qmap_max = np.max(q_map)
		nq = q_map.shape[0]#q_step 	# Number of q radial polar bins
		q_bins = arange( 0, nq, nq/10 )	# 6 ticks arrange(start, stop, step-size)
		q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins]


		# ---- Interpolater initiate : ----	# fs = fast-scan {x} cntr[0]; Define a polar image with dimensions: (qRmax-qRmin) x nphi
		Interp = RingData.InterpSimple( cntr[0], cntr[1], qmax_pix, qmin_pix, nphi, m_b.shape)
		####	qRmin,qRmax  - cartesianimage will be interpolated from qR-qRmin, qR + qRmax in pixel units ####
		####	Default in CXI-2018_MArting/scripts/ave_corrs : 700, 0		####

		# ---- Make a Polar Mask : ----
		polar_mask = Interp.nearest(m_b, dtype=bool).round() ## .round() returns a floating point number that is a rounded version of the specified number ##

		# ---- Generate Polar Image/Images (N Diffracton Patterns) : ----
		print "\n Starting to Calculate the Polar Images...\n"
		#for i in range(N): polar_imgs = np.array( [polar_mask* Interp.nearest(img[i])])
		polar_imgs = np.array( [ polar_mask* Interp.nearest(img[i]) for i in range( N) ] )
		#print " Dim of the Polar Images: ", polar_imgs.shape # = (5, 90, 3)

		# ---- Normalize - with function: ----
		polar_imgs_norm =norm_polar_img(polar_imgs, mask_val=0) # Function
		
		# --- LookUp-Map : ----
		# numb_map = {}
		# for indx, N in Range (N):		#enumerate( exposure_tags)
		#	numb_map{N} = index


		# ---- SAVE the generated Polar Data (normalise and un-normalised), Mask, phi-bins, q-mapping
		if save_polar_param:
			if 'polar_data_normalized' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data_normalized', data = polar_imgs_norm) 
			if 'polar_data' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data', data = polar_imgs) 
			if 'polar_mask' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_mask', data = polar_mask.astype(int)) 
			if 'num_phi' not in data_hdf.keys():	data_hdf.create_dataset( 'num_phi', data = nphi, dtype='i1') #INT, dtype='i1', 'i8'f16'
			if 'q_mapping' not in data_hdf.keys():	data_hdf.create_dataset( 'q_mapping', data = q_map)
			del Interp, cntr, radial_dim # Free up memory

		# ---- Save by closing file: ----
		data_hdf.close()
		print "\n File Closed! \n"


		######################  PLOT 1st Pattern in Polar Coordinates (Norm)[fig.3]{/SACLA tutorial}: ################### 
		if plot_polar_img :
			print "\n Starting to Plot the Polar Images...\n"
			nmbr = 0 	# Pattern number 'nmbr'
			# ---- Look at 1 Centered Image in Polar Coordinates: ----
			polar_im = polar_imgs_norm[nmbr] 	#[0]
			#polar_im =polar_imgs[0]
			plt.figure(3, figsize=(15,6))
			#imshow(polar_im, vmax = 1000, aspect='auto')
			imshow(polar_im, aspect='auto', vmax = polar_im.max())#, cmap='jet')	#default cmap = 'viridis', mtlab =jet
			cb = plt.colorbar()#polar_im)#, ax=ax, shrink=cb_shrink, pad= cb_padd)#, label=r'$\log_{10}\ $')
			cb.set_label(label=' --- [A.U]', weight='bold', size=12)

			# #---- Adjust the X-axis: ----		### NEED ADJ
			xtic = nphi_bins 	#[nphi/4, nphi/2, 3*nphi/4]
			xlab = nphi_label 		# [ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
			xticks(xtic, xlab)

			# #---- Adjust the Y-axis: ----
			#qmap_max = np.max(q_map), print "... max index in q_map: ", qmap_max
			#len_qmap = len(q_map),print "... and q_map shape (dim): ", q_map.shape #,"... and length of q_map: ", len_qmap, 
			#print " q_map: \n", q_map
			#ind, val = enumerate(q_map),print "highest in q_map: ", np.max(ind)
			#nq = q_map.shape[0]#q_step 	# Number of q radial polar bins
			ytic = q_bins #arange( 0, nq, nq/10 )	# 6 ticks arrange(start, stop, step-size)
			#print "ytic: ", ytic
			ylab = q_label # [ '%.3f'%(q_map[x,1]) for x in ytic]
			yticks(ytic, ylab)

			# #---- Label xy-Axis: ----
			xlabel(r'$\phi$', fontsize = 18)
			ylabel(r'$q \, [\AA^{-1}]$', fontsize =18)
			title('Polar Image of Pattern nr. %i (normalised)' %(nmbr+1),  fontsize=14)
			show()

			fig_name = "Figure_3_polar_image_(qx-%i_qi-%i)_w_Mask.%s" %(qmax_pix,qmin_pix,frmt)
			#pypl.savefig( out_fname + fig_name)
			#print "\n Plot saved as %s " %fig_name
			del fig_name
		################################################################################ 

		############## Compare Diffractions with Each Other (AUTO-Correlation) for N Patterns: ##############
		#		from Loki/Salca CXS tutorial
		if calc_AutoCorr:
			t_AC = time.time()
			print "\n Starting to Auto-Correlate the Polar Images...\n"
			# ---- Calculate the Difference in Intensities and store in a List: ----
			exposure_diffs = []
			#for i in range(N-1):	exposure_diffs.append(img[i]-img[i+1])
			# 	print "exposure diff vector's shape", len(exposure_diffs)  
			# exposure_diffs = np.asarray(exposure_diffs) 	# = (4, 1738, 1742)
			
			#exposure_diffs.append(img[:-1]-img[1:]) 		# = (1, 4, 1738, 1742)
			#exposure_diffs = np.asarray(exposure_diffs[0]) # = (4, 1738, 1742)
			exposure_diffs_cart =img[:-1]-img[1:]				# Intensity Patterns in Carteesian cor
			exposure_diffs_cart = np.asarray(exposure_diffs_cart) 	# = (4, 1738, 1742)
			
			exposure_diffs_pol =polar_imgs_norm[:-1]-polar_imgs_norm[1:]	# Polar Imgs
			exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)
			
			#print "exposure diff vector's shape", exposure_diffs_cart.shape, ' in polar ", exposure_diffs_pol.shape

			# ---- Autocorrelation of each Pair: ----
			#DC_crt = RingData.DiffCorr( exposure_diffs_cart)
			#DC_plr = RingData.DiffCorr( exposure_diffs_pol )
			#cor_crt = DC_crt.autocorr()		#Return the Difference Autocorrelation of Carteesian space
			#cor_plr = DC_plr.autocorr()		#Return the Difference Autocorrelation of the Polar imgs
			#cor_mean = [cor_crt.mean(0), cor_plr.mean(0)]
			acorr = [RingData.DiffCorr( exposure_diffs_cart).autocorr(), 
							RingData.DiffCorr( exposure_diffs_pol ).autocorr()]
			cor_mean = [RingData.DiffCorr( exposure_diffs_cart).autocorr().mean(0), 
							RingData.DiffCorr( exposure_diffs_pol ).autocorr().mean(0)]

			# ---- Choose in which coordinate Systam to plot Autocorrelation in: ----				
			polar_cord = True#False
			if not polar_cord: 	# In Carteesian Coordinate System
				ind =0
				cord_sys = "Carteesian"
			else:			# In Polar Coordinate System
				ind = 1
				cord_sys = "Polar"
			cor_mean = np.asarray(cor_mean[ind]) # 0 = cart; 1 = polar
			#print "\n Dim of cor: ", cor.shape 				# =(4, 1738, 1742); (4, 190, 5)
			#print "\n Dim of cor_mean: ", cor_mean.shape 	#(1738, 1742); (190, 5)

			t = time.time()-t_AC
			t_m =int(t)/60
			t_s=t-t_m*60
			print "AutoCorrelation Time: ", t_m, "min, ", t_s, " s \n"
			######################  PLOT Auto-Correlations [fig.4] : ################################################################## 
			pypl.figure(4, figsize=(18,10))
			cb_shrink, cb_padd = 1.0, 0.2
			pypl.subplot(121) #ALT. figure(figsize = (10, 10))
			#imshow ( cor_mean[:, int( nphi*0.05 ) : -int( nphi*0.05 )], aspect='auto')
			im=imshow ( cor_mean, aspect='auto')
			cb = pypl.colorbar(im, orientation="horizontal", shrink=cb_shrink, pad= cb_padd, format = '%8.1e')	
			cb.set_label(r'Auto-correlation of Intensity Difference [a.u.]',  size=12) # normed intensity
			#cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
			#cb.update_ticks()
			if polar_cord:
				# #---- Adjust the X-axis: ----  		## NEED Adj x tic for 5 tics
				xtic = nphi_bins 	#[nphi/4, nphi/2, 3*nphi/4]
				xlab = nphi_label 			# [ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
				pypl.xticks(xtic, xlab)

				# #---- Adjust the Y-axis: ----
				#qmap_max = np.max(q_map)
				#nq = q_map.shape[0]#q_step 	# Number of q radial polar bins
				ytic = q_bins # arange( 0, nq, nq/10 )	# 6 ticks arrange(start, stop, step-size)
				ylab = q_label # [ '%.3f'%(q_map[x,1]) for x in ytic]
				pypl.yticks(ytic, ylab)

				# #---- Label xy-Axis: ----
				pypl.xlabel(r'$\phi$', fontsize = 18)
				pypl.ylabel(r'$q \, [\AA^{-1}]$', fontsize =18)

			pypl.title("Auto-Correlation [%s]" %cord_sys,  fontsize=16)
			############### [fig.4b]  Plot Sigma (Normed Auto-Correlation with +-2 std as limits)  ##############
			subplt_ave_corrs = True#False 		#### Plot the 2nd plot {adapted from in // GDrive/.../scripts/ave_corrs & 'plot_a_corr.py'} ####
			if subplt_ave_corrs:
				pypl.subplot(122) 	#### plot as in // GDrive/.../scripts/ave_corrs & 'plot_a_corr.py' #####
				corr = None
				for i in range(acorr[ind].shape[0]):		#acorr[ind].shape = (4, 190, 5)
					if corr is None:	corr = acorr[ind][i]
					else :	corr += acorr[ind][i]		# adter loop cr.shape = (190, 5)
				#print "\n corr shape: ", acorr[ind].shape, "\n & corr : ", corr.shape
				corrsum = np.zeros_like(corr)		
				corr_count =np.zeros(1)+(exposure_diffs_pol.shape[0]) #exposure_diffs_pol.shape[0] = N-1
				#print "corr_count: ", corr_count
				#test = np.zeros(1)
				#test += 4
				#print "np.zeroes(1) ", np.zeros(1), " \n and add 1 to array: ", test, "\n"
				tot_corr_count = np.zeros(1)
				#print "\n tot_corr_count: ", tot_corr_count
				from mpi4py import MPI
				comm = MPI.COMM_WORLD
				comm.Reduce(corr,corrsum)
				comm.Reduce(corr_count, tot_corr_count)
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
				if polar_cord:
					im = ax.imshow( sig,
            	            extent=[0, 2*np.pi, qmax_pix, qmin_pix], 
            	            vmin=vmin, vmax=vmax, aspect='auto')
				else : im =ax.imshow(sig, vmin=vmin, vmax=vmax, aspect='auto' )
				cb = pypl.colorbar(im, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)	
				cb.set_label(r'Auto-correlation of Intensity Difference [a.u.]',  size=12) # normed intensity
				#cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
				#cb.update_ticks()

				#ax.images[0].set_data(sig) # ! only use if there is previous imshow
				#ax.images[0].set_clim(vmin, vmax)
				ax.set_title(r"Average of %d corrs [%s] with limits $\mu \pm 2\sigma$"%(tot_corr_count[0], cord_sys),  fontsize=16)
				############### [fig.4b] End ###############
			#pypl.rcParams.update({'axes.labelsize': 14, 'xtick.labelsize':'small', 'ytick.labelsize':'small'})
			pypl.subplots_adjust(wspace=0.2, hspace=0.2, left=0.07, right=0.99)
			#pypl.tight_layout()  ## overrule adjustments ##
			#show() 
			fig_name = "Figure_4_Diff-Auto-Corr_SUBPLOT_(qx-%i_qi-%i_nphi-%i)_w_Mask_%s.%s" %(qmax_pix,qmin_pix, nphi,cord_sys,frmt)
			pypl.savefig( out_fname + fig_name)
			print "\n Sub-plot saved as %s " %fig_name
		################################################################################ 

		############## Cross Correlate: ############## ## ? Must have diffrence between shots -> possible with self
		####   def crosscorr(self(3xQxNphi), qindex): Correlates ring denoted by qindex with every other ring (including itself) ####
		if calc_CrossCorr:				# shot[qindex]
			plt_1_q = True#False #True		# Plot image of self-Cross-Correlation for 1 q-value
			if plt_1_q: qindex_to_plt = 50 	# Choose an index to look at
			plt_trend_q = True#False#True 	# Plot Self-CC for several q ?
			
			print "\n Starting to Cross-Correlate the Polar Images...\n"
			t_CC = time.time()
			polar_im= polar_imgs_norm#[0]	# or choose which pattern


			# ---- Calculate the Difference in Intensities and store in a List: ----
			exposure_diffs = []
			exposure_diffs_pol =polar_imgs_norm[:-1]-polar_imgs_norm[1:]	# Polar Imgs
			exposure_diffs_pol = np.asarray(exposure_diffs_pol) 	# = (4, 190, 5)xposure_diffs = np.vstack(exposure_diffs) # From List-object to array in vertical stack (row wise)
			
			# ---- Correlation-instance of each Pair: ----
		 	#DC = RingData.DiffCorr( polar_im , pre_dif=True)#False)	# img in polar; pre_dif: default True, assume shots is difference in intensities
		 	#	Set pre_dif=True and shots= current polar image instead of shot-difference(intensities) for plot of self correlation
			DC = RingData.DiffCorr( exposure_diffs_pol , pre_dif=True)

			####### 1 q-index with difference (Self if N=1) : ############################################### !! works for N=1 MUST FIX FOR N>1 !!
			if plt_1_q:

				#DC = RingData.DiffCorr( polar_im , pre_dif=True)#False)	WORKS for N=1 MUST FIX FOR N>1 !!
				q_map_ind = q_map[:,0].astype(int) 	# turn index vaules from float to int
				assert(qindex_to_plt in q_map_ind)	# Make sure the choosen index exists in the q_map (in the range of qs') 
				#ccor = DC.crosscorr(q_map_ind[qindex_to_plt]) 	#def crosscorr(self, qindex)
				ccor = DC.crosscorr(qindex_to_plt) 	#def crosscorr(self, qindex)
				#ccor=RingData.DiffCorr( exposure_diffs_pol[0] ).crosscorr(q_map_ind[50]) # for 1 pattern
				#cor=RingData.DiffCorr( exposure_diffs_pol ).crosscorr(q_map_ind[50]).mean()
				ccor = np.array(ccor) 	# (1, 100,42) for N=1;  (4, 190, 5) for N=5
				ccor_mean = ccor.mean(0) #(100,42) for N=1, (190, 5) for N=5
				
				t = time.time()-t_CC
				t_m =int(t)/60
				t_s=t-t_m*60
				print "CrossCorrelation Time: ", t_m, "min, ", t_s, " s\n" 
				print "\n Dim of ccor: ", ccor.shape
				print "\n Dim of ccor_mean: ", ccor_mean.shape

				######################  PLOT Cross-Correlations [fig.5] : ################################################################## 
				pypl.figure(5, figsize=(15,8))
				#imshow( ccor_mean[:, int( nphi*0.05 ) : -int( nphi*0.05 )], aspect='auto') #IndexError: invalid index to scalar variable.
				imshow( ccor_mean, aspect='auto') #IndexError: invalid index to scalar variable.
				#title("Cross-Correlation with self for q_index = %i" %(q_map_ind[50])) when N =1
				title("Mean Cross-Correlation for q_index = %i" %(qindex_to_plt))
				################ Adjust Axis, Titles & Tics (same as "plot_polar_img") ###############
				cb = plt.colorbar()#polar_im)#, ax=ax, shrink=cb_shrink, pad= cb_padd)#, label=r'$\log_{10}\ $')
				cb.set_label(label=' --- [A.U]', weight='bold', size=12)

				# #---- Adjust the X-axis: ----  		## NEED Adj x tic for 5 tics
				xtic = nphi_bins		 #[nphi/4, nphi/2, 3*nphi/4]
				xlab = nphi_label			# [ r'$\pi/2$', r'$\pi$', r'$3\pi/2$'] #nphi_bins or phi_bins_5
				xticks(xtic, xlab)

				# #---- Adjust the Y-axis: ----
				qmap_max = np.max(q_map)
				nq = q_map.shape[0]#q_step 	# Number of q radial polar bins
				ytic = q_bins #arange( 0, nq, nq/10 )	# 6 ticks arrange(start, stop, step-size)
				ylab = q_label # [ '%.3f'%(q_map[x,1]) for x in ytic]
				yticks(ytic, ylab)

				# #---- Label xy-Axis: ----
				xlabel(r'$\phi$', fontsize = 18)
				ylabel(r'$q \, [\AA^{-1}]$', fontsize =18)

				show() 
				#pypl.savefig( out_fname +'Cross-correlation_0.%s' %(frmt))
				#print "Sub-plot saved as %s " %"Cross-correlation_0"   
				fig_name = "Figure_5_Cross-correlation_g-%i_(qx-%i_qi-%i)_w_Mask.%s" %(qindex_to_plt,qmax_pix,qmin_pix,frmt)
				#pypl.savefig( out_fname + fig_name)
				#print "Plot saved as %s " %fig_name
				del fig_name
			################################################################################
			
			####### All q-index with Self : #### works for N=1 MUST FIX FOR N>1 !!
			if plt_trend_q:
				#DC = RingData.DiffCorr( polar_im[0] , pre_dif=True)#False) works for N=1, FIX FOR N>1
				q_map_ind = q_map[:,0].astype(int) 	# turn index vaules from float X. to int X; Array (100,)
				#print "dim q_map_ind: ", q_map_ind.shape 	# =   (190,)
				#print "q_map_ind[0]: ", q_map_ind[0]
				#print "values in g_map[:,1]",q_map[:,1] 	# 190 values
				ccor_q_m = []#None 	# List object, add by 'obj.append('values')'
		 		#ccor_q = np.empty([exposure_diffs_pol.shape[0],q_map.shape[0]], dtype=float) # Empty vector for each q-value in q_map
		 		ccor_q = np.empty([q_map_ind.shape[0], exposure_diffs_pol.shape[0], exposure_diffs_pol.shape[1],exposure_diffs_pol.shape[2]], dtype=float) # Dim = (190, 4, 190, 5)
		 		#ccor_q_m = np.empty([q_map_ind.shape[0], exposure_diffs_pol.shape[1],exposure_diffs_pol.shape[2]], dtype=float) # Dim = (190, 190, 5)
		 		import sys 	# for 'sys.stdout.flush()' force print in loop
		 		for ind in range(q_map_ind.shape[0]):
				#	#if ccor is None:	ccor = DC.crosscorr(q_index)
				#	#else:	ccor += DC.crosscorr(q_index)	#Return the Cross-correlation of img[0]
				#	ccor.append( DC.crosscorr(q_index) )	#Return the Cross-correlation of img[0]
				#	print "index: ", ind, " ccor: ", ccor[ind], " q_index: ", q_index
				#	#ccor[ind] = DC.crosscorr(q_index)
					ccor_i = DC.crosscorr(ind)
					ccor = np.array(ccor_i) 	#  (4, 190, 5)
					#print "\n Dim of ccor: ", ccor.shape#, end = ' ')# "\n", # NO PRINT IN LOOP IN PYT2.7 ?!
					#sys.stdout.write("\n Dim of ccor: %i, %i" %(ccor.shape[0],ccor.shape[1])), sys.stdout.flush() #, ccor.shape
					#sys.stdout.write("\n Dim of ccor:"), sys.stdout.write(str(ccor.shape)), sys.stdout.flush()
					ccor_mean = ccor.mean(0) #(190, 5)
					#ccor_q_m += ccor_mean
					#ccor_q_m[ind,:,:] =ccor_mean
					ccor_q_m.append( ccor_mean ) 	# for N=1
					#ccor_q_m = np.asarray(ccor_q_m)	# for N=1, (190, 190, 5)
					#ccor_q.append( ccor_i ) 
					ccor_q[ind] = ccor_i  
					#ccor_q = np.asarray(ccor_q) #  (4, 190)
				#print "all 'q_map_ind': ", q_map_ind
				ccor_q_m = np.array(ccor_q_m)
				t = time.time()-t_CC
				t_m =int(t)/60
				t_s=t-t_m*60
				print "CrossCorrelation Time: ", t_m, "min, ", t_s, " s\n" 
				
				#print "\n Dim of ccor: ", ccor.shape
				print "Dim of ccor_q: ", ccor_q.shape #len(ccor_q)  #((190, 4, 190, 5)
				print "Dim of ccor_q_mean: ", ccor_q_m.shape #len(ccor_q)  #(190, 190, 5)
				print "Dim of Transpose of ccor_q_mean: ", np.transpose(ccor_q_m).shape
				#print "\n Dim of ccor_mean: ", ccor_mean.shape
				#print "ccor_q_mean values: \n",ccor_q_m#[:,0,0]  # only ZERO!
				#print "ccor_q values: \n",ccor_q#[:,0,0]
				######################  PLOT Auto-Correlations [fig.6] : ################################################################## 
				pypl.figure(6, figsize=(12,5))
				ax = pypl.gca()
				ax.tick_params(which='both', labelsize =12, length =0)
				#figure(figsize = (10, 10)) #pypl.figure(figsize = (10, 10))
				#plot(ccor_q)
				#plot(q_map[:,1], ccor_q[0]) 	# 1st Patterns values,  alt. Subplot ALL?
				#for i in range(exposure_diffs_pol.shape[0]): plot(q_map[:,1], ccor_q[:,i,0,0], lw=2,  label=r"$\Delta$I #%i" %i)
				#for i in range(ccor_q_mean.shape[0]): 
					#ax.plot(q_map[:,1], ccor_q_mean[i,:,3], lw=2, ms=7, marker= 'o', label=r"$\Delta$I q_index=%i" %i) # plots 1 straight line
					#ax.plot(ccor_q_mean[i,:,3], lw=2, ms=7, marker= 'o', label=r"$\Delta$I q_index=%i" %i) # plots 1 straight line
					# 	ccor_q[:,i,None,None]
				plot(np.transpose(ccor_q_m)[2])	# phi[2] = pi or 2pi/5 of indx= 0,1,2,3,4
				## --- try as DataFrame:
				#from pandas import *
				#df = DataFrame(ccor_q_mean[:,:,0], index=[range(q_map.shape[0]),range(q_map.shape[0])])#, columns=list(range(q_map_ind[0])))
				#df = df.cumsum()
				#df.plot(colormap = 'jet')

				
				# #---- Adjust the X-axis to inverse Angstrom: ----
				qmap_max = np.max(q_map)
				nq = q_map.shape[0]#q_step 	# Number of q radial polar bins
				xtic = q_bins #arange( 0, nq, nq/10 )	# 6 ticks arrange(start, stop, step-size)
				xlab = q_label # [ '%.3f'%(q_map[x,1]) for x in ytic]
				xticks(xtic, xlab)
				xlabel(r'$q \, [\AA^{-1}]$', fontsize =16) #("q index", fontsize =16) #xlabel(r'radial q \, [pixels]', fontsize =16) #("q index", fontsize =16)

				title("Mean Cross-Correlation for all q_index at $\phi=\pi$",  fontsize =18) # phi_bins
				ylabel(r"Cross-Correlation [---]",  fontsize =16)
				#legend()
				show() 
				fig_name = "Figure_6_Cross-correlation_(qx-%i_qi-%i)_w_Mask.%s" %(qmax_pix,qmin_pix,frmt)
				#pypl.savefig( out_fname + fig_name)
				#print "Plot saved as %s " %fig_name
				del fig_name
				#pypl.savefig( out_fname +'Cross-correlation_0.%s' %(frmt))
				#print "Sub-plot saved as %s " %"Cross-correlation_0"   
			################################################################################
		#######################################################################################


####################################################################################
####################################################################################
#------------------------------------ cxiLT14py XCCA: -----------------------------#
####################################################################################
####################################################################################
if XCCA_cxiLT14py:		## from AngularCorrelation.py {- simple script to calculate angular correlations and check convergence}
	#from pylab import *	
	#import AnalysisTools as anto 	## import whole folder (run init-file) Folder must be located in run-directory
	from AnalysisTools import correlation  ## e.g.from loki import RingData; from loki.RingData import RadialProfile, InterpSimple
	print "\n Data Analysis with cxiLT14py.\n"

	images = Ip_w_mb # or amplitudes: Ap_w_mb ## All Patterns with Better-Mask-Assembled,
	
	###################### Choose Caculations: ###################
	data_normalize = False 	## Normalise data ##
	diffCor = False 		## Choose if to calculate for Diff of current with random frame ##
	randomXcorr = False  	##Calculate the correlation between current frame and a random frame. ##
	save_polar_param = False ## Save the Calculated data ##

	###################### Choose PLOTs: ###################
	plot_polar_mask = True#False#True
	plot_acorr_mask = True#False#True
	plot_acorr_raw = True 		## Plot directly after calculation ##
	plot_acorr_proc = True 		## Plot after corrections ##

	# ---- Some Useful Functions from: cxiLT14py/AnalysisTools/radial_profile.py -----
	# --------------------------------------
	def div_nonzero(a, b):				## Currently not in use ##
		m = b != 0
		c = np.zeros_like(a)
		c[m] = a[m] / b[m].astype(a.dtype)
		return c
	# --------------------------------------
	def make_radial_profile(image, x, y, mask = None, rs = None): 		## Currently not in use ##
		"""
		"""
		if rs is None:
			rs = np.round(np.sqrt(x**2 + y**2), 0).astype(np.uint16).ravel()

		if mask is None:
			mask = np.ones_like(image, dtype=np.bool) 
		m = mask.ravel().astype(np.bool)

		r_count = np.bincount(rs[m], minlength=rs.max()+1)
		r_int   = np.bincount(rs[m], image.ravel()[m].astype(np.float),  minlength=rs.max()+1)
		#print('r_count.shape, r_int.shape', r_count.shape, r_int.shape, m.shape, rs.shape)
		return div_nonzero(r_int, r_count), rs
	# --------------------------------------    

	# ---- Generate a Storage File's Prefix: ----
	prefix = '%s_%s_store-param' %(name,pdb)
	LT14_folder = outdir + '/%s_%s_%s_with_cxiLT14/' %(name,run,pdb)
	if not os.path.exists(LT14_folder):
		os.makedirs(LT14_folder)
	out_fname = os.path.join( LT14_folder, prefix) 	# with directory
	if save_polar_param:
		data_hdf = h5py.File( out_fname + 'parameters-ACC.hdf5', 'w')	# a (append): read/write/create, r+ : write/must-exist
		#data_hdf = file_hdf.create_group("cxiLT14py")	


	# ----- Angular Correlation Structure: -----
	#ac = anto.correlation.angular_correlation() 	##if #import AnalysisTools as anto 
	ac = correlation.angular_correlation()			## class angular_correlation in Correlation.py

	# ---- Some Parameters for Polar Plot: ----
	## polarRange", nargs=4, help="pixel range to evaluate polar plots of assembled images ", type=float ##
	## pix2invang = lambda qpix : sin(arctan(qpix*(ps*1E-6)/dtc_dist )/2)*4*pi/wl_A
	if cntr[1]> cntr[0] :	theta_half = np.arctan((cntr[0])*(ps*1E-6)/dtc_dist) 	# use smaller center value => less data outside radius
	else :	theta_half = np.arctan((cntr[1])*(ps*1E-6)/dtc_dist) 
	#polarRange = [60, 250, 0, theta_half]; r
	## 
	#### GDrive/.../Exp_CXI-Martin/scripts/corpairs.py: min = 54; rmax = 1130 ####
	#polarRange = [60, 250, 0, (2)*np.pi] # 0, (2)*np.pi]; np.pi, (2)*np.pi]; (1/2.0)*np.pi, (2)*np.pi];  np.pi, (4)*np.pi; np.pi, (3)*np.pi]
	#polarRange = [54, 1130, 0, (2)*np.pi] ## from  GDrive/.../Exp_CXI-Martin/scripts/corpairs.py: ##
	polarRange = [60, 1130, 0, (2)*np.pi] ## 60 from radial_Profile in LOKI ###
	## thmax=polarRange[3] with thmin=0:yields 0s polar-mask for theta_half, pi; yields 1s polar-mask for pi2/3, pi3/4
	## atp.parser.add_argument( "--nq", help="number of radial (q) polar bins ", type=int 	##
	## atp.parser.add_argument( "--nth", help5="number of angular polar bins ", type=int 	##
	nq, nth = polarRange[1]-polarRange[0], 180#360#180##90#30#5 # Try same as in LOKI above in line 568-569 & 572; with nq = qmax_pix-qmin_pix, nth = nphi
	qmin, qmax, thmin, thmax = polarRange[0], polarRange[1], polarRange[2], polarRange[3]
	cenx, ceny = cntr[1], cntr[0]		## Center-file loaded above in line 101
	
	
	# ----- Tic Parameters for choosen Pixel Resolution (for PLOTS): ----
	nphi_label = [ r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
	#nphi_label = [r"$\pi/4$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$7\pi/4$" ]	# center of columen for full Period 2pi, #div 4 & 8	
	#nphi_label = [r"$\pi/3$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$5\pi/3$" ]	# center of columen for full Period 2pi, #div 4 & 6 
	#nphi_label_5 = [r"$2\pi/5$", r"$2*2\pi/5$", r"$3*2\pi/5$", r"$4*2\pi/5$"]	# div 5
	nphi_bins = [nth/4.0, nth/2.0, 3*nth/4.0]
	#nphi_bins = [nth/8.0, nth/4.0, nth/2.0, 3*nth/4.0, 7*nth/8.0]	# div 4 & 8	
	#nphi_bins = [nth/6.0, nth/4.0, nth/2.0, 3*nth/4.0, 5*nth/6.0]	# div 4 & 6	
	#nphi_bins_5 = [nth/5.0, 2*nth/5.0, 3*nth/5.0, 4*nth/5.0]	# div 5
	qrange_pix = np.arange( qmin, qmax)
	#q_map = np.array( [ [ind, pix2invang(q)] for ind,q in enumerate( qrange_pix)])
	nq = qrange_pix.shape[0] #q_map.shape[0] 	# Number of q radial polar bins
	q_bins = np.arange( 0, nq, nq/10 )	# 6 ticks arrange(start, stop, step-size)
	q_label = [ '%i'%(qrange_pix[x]) for x in q_bins]#[ '%.3f'%(q_map[x,1]) for x in q_bins]


	# ---- Calculate the Mask's Polar Plot: ----
	print " Starting to Calculate the Polar Mask...\n"
	mask = mask_better.astype(float) 	## Loaded previously ##
	#print "max in mask: ", mask.max(),
	pplot_mask = ac.polar_plot( mask, nq, nth, qmin, qmax, thmin, thmax, cenx+mask.shape[0]/2, ceny+mask.shape[1]/2, submean=True ) 
	#print "... max in polar_plot: ", pplot_mask.max(), "... min : ", pplot_mask.min()
	ineg = np.where( pplot_mask < 0.0 )
	pplot_mask[ineg] = 0.0
	pplot_maskones = pplot_mask*0.0 + 1.0 ## ? create a new mask of ones from pplot_mask ?? ##
	pplot_maskones[ineg] = 0.0				## ? set the new mask's zeroes at same indices as pplot_mask ?? ##
	#print "... max in pplot_maskones: ", pplot_maskones.max(), "... min : ", pplot_maskones.min()
	
	# ---- Angular Correlation of Each q-Shell With Itself: ----
	corrqq_mask = ac.polarplot_angular_correlation( pplot_mask ) ## Complex Value ##
	
	########## PLOT Polar MAsks [fig.1]: ##############################
	if plot_polar_mask:
		pypl.figure(1, figsize=(15,8))
		pypl.subplot(121)
		pypl.imshow(pplot_mask, aspect='auto', vmax = pplot_mask.max())
		pypl.title("'pplot_mask'",  fontsize=14), pypl.colorbar()
		pypl.xticks(nphi_bins, nphi_label), pypl.yticks(q_bins, q_label)
		pypl.xlabel(r'$\phi$', fontsize = 18), pypl.ylabel(r'q  [pixels]', fontsize =18)#(r'$q \, [\AA^{-1}]$', fontsize =18)
		pypl.subplot(122)
		pypl.imshow(pplot_maskones, aspect='auto',vmax = pplot_maskones.max())	
		pypl.title("'pplot_maskones'",  fontsize=14), pypl.colorbar() 
		pypl.xticks(nphi_bins, nphi_label), pypl.yticks(q_bins, q_label)
		pypl.xlabel(r'$\phi$', fontsize = 18), pypl.ylabel(r'q  [pixels]', fontsize =18)#(r'$q \, [\AA^{-1}]$', fontsize =18)
		
		fig_name = "Figure_1_SUBPLOT_Polar_MASk_(qx-%i_qi-%i)_%iphibins_w_Mask_.%s" %(qmax,qmin,nth,frmt)
		#pypl.show()
		pypl.savefig( out_fname + fig_name)
		print "Plot saved as %s " %fig_name
		del fig_name
	######################################################################

	########## PLOT  MAsks ACC [fig.2]: ################################
	if plot_acorr_mask:	# ifft (conjunate) => Ned ABS-value
		pypl.figure(2, figsize=(15,8))
		pypl.imshow(abs(corrqq_mask), aspect='auto', vmax = pplot_mask.max())
		pypl.title("MASK: Polar Angular Correlation",  fontsize=14), pypl.colorbar()
		pypl.xticks(nphi_bins, nphi_label), pypl.yticks(q_bins, q_label)
		pypl.xlabel(r'$\phi$', fontsize = 18), pypl.ylabel(r'q  [pixels]', fontsize =18)#(r'$q \, [\AA^{-1}]$', fontsize =18)
		
		fig_name = "Figure_2_MASK-ACC_(qx-%i_qi-%i)_%iphibins_w_Mask.%s" %(qmax,qmin,nth,frmt)
		#pypl.show()
		pypl.savefig( out_fname + fig_name)
		print "\ Plot saved as %s " %fig_name
		del fig_name
	######################################################################

	# ---- Sum Frames of Data from a Diffraction Pattern: ----
	s = images.shape
	datasum =  np.zeros( (s[0], s[1], s[2], 2 ))
	pplotsum = np.zeros( (nq,nth,2) )
	pplot_mean_sum = np.zeros( (nq,nth,2) )
	#pplotsvdsub = np.zeros( (nq,nth,atp.args.svdnsub,2) )
	corrqqsum = np.zeros( (nq,nth,2) )
	#corrqqsum_svdsub = np.zeros( (nq,nth,atp.args.svdnsub,2) )
	nprocessed = np.zeros( 2 )
	if randomXcorr == True:		corrqqsumX = np.zeros( (nq,nth,2) )
	totalIList = []		## total intenisty

	print " Starting to Calculate the Angular-Crosscorrelations...\n"
	t_ACC = time.time()
	for i in range(images.shape[0]): #range(images.shape[0]-1) # N-1 times
		print "Processing event ", i
		m = i % 2

		# ---- Collect Frames: ---
		rand_frame = np.random.randint(0,images.shape[0])
		while rand_frame ==i : rand_frame = np.random.randint(0,images.shape[0])
		data1= images[i]
		data2 = images[np.random.randint(0,images.shape[0])]#images[i+1]; randint() low (inclusive), high (exclusive))
		
		# ---- Total Intensity: ---
		totalI1= data1.sum()
		totalI2= data2.sum()
		print "Total Intenisty :", totalI1, totalI2
		nprocessed[m] += 1		## Number of event procesed
		totalIList.append( [totalI1, totalI2] )

		# ---- Normalize: ---
		if data_normalize==True:
			data1 *= 1.0/np.average(data1*mask)
			data2 *= 1.0/np.average(data2*mask)

		# ---- Sum the Data: ---
		datasum[:,:,:,m] += data1*mask

		# ---- diffCor: ---	
		if diffCor == True: img = data1-data2 	## Diff of current with random frame
		else:	img = images[i]					## current with random frame

		# ---- Calculate the Polar Image: ---	
		pplot_mean = ac.polar_plot( img, nq, nth, qmin, qmax, thmin, thmax, cenx+img.shape[0]/2, ceny+img.shape[1]/2, submean=False )
		pplot = ac.polar_plot( img, nq, nth, qmin, qmax, thmin, thmax, cenx+img.shape[0]/2, ceny+img.shape[1]/2, submean=True ) ## with 'polar_plot_subtract_rmean'
		#pplotcopy = np.copy( pplot) ## for SVDn subtraction
		pplot_mean_sum[:,:,m] += pplot_mean
		pplotsum[:,:,m] += pplot
		
		# ---- Angular Correlation of Image: ---	
		corrqq = ac.polarplot_angular_correlation( pplot * pplot_maskones )
		#corrqqsum[:,:,m] += corrqq ## TypeError: Cannot cast ufunc add output from dtype('complex128') to dtype('float64') with casting rule 'same_kind'
		#corrqqsum[:,:,m] = np.add(corrqqsum[:,:,m], corrqq, out=corrqqsum[:,:,m], casting="unsafe") ##ComplexWarning: Casting complex values to real discards the imaginary part
		corrqqsum[:,:,m] += np.abs(corrqq)**2
		
		# ---- Angular Correlation of Current Frame and a Random Frame: ---
		if randomXcorr== True:       ##Calculate the correlation between current frame and a random frame.
			img1 = images[i]
			img2 = images[rand_frame]
			pplot1 = ac.polar_plot( img1, nq, nth, qmin, qmax, thmin, thmax, cenx+img.shape[0]/2, ceny+img.shape[1]/2, submean=True )
			pplot2 = ac.polar_plot( img2, nq, nth, qmin, qmax, thmin, thmax, cenx+img.shape[0]/2, ceny+img.shape[1]/2, submean=True )
			corrqq_rand = ac.polarplot_angular_correlation( pplot1, pplot2 )
			#corrqqsumX[:,:,m] += corrqq ## Complex Value ##
			#corrqqsumX[:,:,m] = np.add(corrqqsumX[:,:,m], corrqq, out=corrqqsumX[:,:,m], casting="unsafe") ##ComplexWarning: Casting complex values to real discards the imaginary part
			corrqqsumX[:,:,m] += np.abs(corrqq)**2

	t = time.time()-t_ACC
	t_m =int(t)/60
	t_s=t-t_m*60
	print "CrossCorrelation Time: ", t_m, "min, ", t_s, " s\n" 


	# ---- SAVE the generated Polar Data (normalise and un-normalised), Mask, phi-bins, q-mapping
	if save_polar_param:
		if 'Angular Crosscorrelation' not in data_hdf.keys():	data_hdf.create_dataset( 'Angular Crosscorrelation', data = corrqqsum) 
		if 'polar_data' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_data', data = pplotsum[:,:,m]) 

		if 'polar_mask' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_mask', data = pplot_mask.astype(int)) 
		if 'polar_mask_ones' not in data_hdf.keys():	data_hdf.create_dataset( 'polar_mask_ones', data = pplot_maskones.astype(int)) 
		
		if 'num_q' not in data_hdf.keys():	data_hdf.create_dataset( 'num_q', data = nq, dtype='i1') #INT, dtype='i1', 'i8'f16'
		if 'num_nphi' not in data_hdf.keys():	data_hdf.create_dataset( 'num_nphi', data = nth, dtype='i1') #INT, dtype='i1', 'i8'f16'
		
		if 'q_range' not in data_hdf.keys():	data_hdf.create_dataset( 'q_range', data = [qmin, qmax]) #INT, dtype='i1', 'i8'f16'
		if 'th_range' not in data_hdf.keys():	data_hdf.create_dataset( 'th_range', data = [thmin, thmax]) #INT, dtype='i1', 'i8'f16'
		#del  # Free up memory

		# ---- Save by closing file: ----
		data_hdf.close()

	########## PLOT ACC [fig.3]: ################################
	from matplotlib import ticker
	if plot_acorr_raw:	# ifft (conjunate) => Ned ABS-value
		pypl.figure(3, figsize=(15,8))
		f = corrqqsum.shape[2]+1		# Number of Subplots
		plt_name = ["Even Frame", "Odd Frame", "Total"]
		if corrqqsum[:,:,0].max() > corrqqsum[:,:,1].max():	cb_max =corrqqsum[:,:,0].max()
		else:	cb_max =corrqqsum[:,:,1].max()
		for i in range(f):
			pypl.subplot(2,f, i+1)#pypl.subplot(1,f, i+1)
			ax=pypl.gca()
			if i==corrqqsum.shape[2]: im=ax.imshow(np.sum(corrqqsum,2), aspect='auto', vmax = cb_max) # or norm : Normalize ??
			else:	im=ax.imshow(corrqqsum[:,:,i], aspect='auto')#, vmax = cb_max) #		## if storing np.abs(Complex128)**2  ##
					#im=ax.imshow(abs(corrqqsum[:,:,i]), aspect='auto', vmax = cb_max) #	## if storing Complex Matrices ##
			ax.set_title("Polar Angular Correlation: %s" %plt_name[i],  fontsize=14), pypl.colorbar(im, orientation="horizontal") #label=r'...')
			pypl.subplot(2,f, f+i+1)
			ax=pypl.gca()
			if i==corrqqsum.shape[2]: im=ax.imshow(np.sum(pplotsum,2), aspect='auto', vmax = cb_max) # or norm : Normalize ??
			else:	im=ax.imshow(pplotsum[:,:,i], aspect='auto')#, vmax = cb_max) #		## if storing np.abs(Complex128)**2  ##
					#im=ax.imshow(abs(corrqqsum[:,:,i]), aspect='auto', vmax = cb_max) #	## if storing Complex Matrices ##
			ax.set_title("Polar Plot Sum: %s" %plt_name[i],  fontsize=14)
			cb=pypl.colorbar(im, orientation="horizontal") #label=r'...')
			cb.locator = ticker.MaxNLocator(nbins=5) # from matplotlib import ticker
			cb.update_ticks()
		#pypl.xticks(nphi_bins, nphi_label), pypl.yticks(q_bins, q_label)
		#pypl.xlabel(r'$\phi$', fontsize = 18), pypl.ylabel(r'q  [pixels]', fontsize =18)#(r'$q \, [\AA^{-1}]$', fontsize =18)
		#pypl.show()
		fig_name = "Figure_3_SUBPLOT_ACC_(qx-%i_qi-%i)_%iphibins_w_Mask_.%s" %(qmax,qmin,nth,frmt)
		pypl.savefig( out_fname + fig_name)
		print "\n Plot saved as %s " %fig_name
		del fig_name

	######################################################################
	##
	# ---- Divide by Number of Processed Frames: ----
	corrqqsum[:,:,0] *= 1.0/float(nprocessed[0])
	corrqqsum[:,:,1] *= 1.0/float(nprocessed[1])
	##
	print "Number of procesed frames even/odd", nprocessed
	totalIarr = np.array(totalIList)
	print "Min max integrated intensity :", np.min(totalIarr), np.max(totalIarr)

	# ---- Correct Correlations  by Mask Correlation: ----
	#corrqqsum[:,:,0] = ac.mask_correction( corrqqsum[:,:,0], corrqq_mask )
	#corrqqsum[:,:,1] = ac.mask_correction( corrqqsum[:,:,1], corrqq_mask )
	corrqqsum[:,:,0] = ac.mask_correction( corrqqsum[:,:,0], np.abs(corrqq_mask)**2 )
	corrqqsum[:,:,1] = ac.mask_correction( corrqqsum[:,:,1], np.abs(corrqq_mask)**2 )

	if randomXcorr == True:
		#corrqqsum[:,:,0] = ac.mask_correction( corrqqsumX[:,:,0], corrqq_mask) 
		#corrqqsum[:,:,1] = ac.mask_correction( corrqqsumX[:,:,1], corrqq_mask )
		corrqqsumX[:,:,0] += ac.mask_correction( corrqqsumX[:,:,0], np.abs(corrqq_mask)**2)
		corrqqsumX[:,:,1] += ac.mask_correction( corrqqsumX[:,:,1], np.abs(corrqq_mask)**2)
	##
	## pearson correlation to measure similarity of odd/even frame angular correlations
	## ...
	##

	########## PLOT ACC [fig.4]: ################################
	if plot_acorr_proc:	# ifft (conjunate) => Ned ABS-value
		pypl.figure(4, figsize=(15,8))
		f = corrqqsum.shape[2]+1		# Number of Subplots
		plt_name = ["Even Frame", "Odd Frame", "Total"]
		if corrqqsum[:,:,0].max() > corrqqsum[:,:,1].max():	cb_max =corrqqsum[:,:,0].max()
		else:	cb_max =corrqqsum[:,:,1].max()
		for i in range(f):
			pypl.subplot(1,f, i+1)
			ax=pypl.gca()
			if i==corrqqsum.shape[2]: im=ax.imshow(np.sum(corrqqsum,2), aspect='auto', vmax = cb_max) # or norm : Normalize ??
			else:	im=ax.imshow(corrqqsum[:,:,i], aspect='auto')#, vmax = cb_max) #		## if storing np.abs(Complex128)**2  ##
					#im=ax.imshow(abs(corrqqsum[:,:,i]), aspect='auto', vmax = cb_max) #	## if storing Complex Matrices ##
			ax.set_title("Polar Angular Correlation: %s" %plt_name[i],  fontsize=14), pypl.colorbar(im, orientation="horizontal") #label=r'...')
		#pypl.xticks(nphi_bins, nphi_label), pypl.yticks(q_bins, q_label)
		#pypl.xlabel(r'$\phi$', fontsize = 18), pypl.ylabel(r'q  [pixels]', fontsize =18)#(r'$q \, [\AA^{-1}]$', fontsize =18)
		#pypl.show()
		fig_name = "Figure_4_SUBPLOT_ACC-corrected_(qx-%i_qi-%i)_%iphibins_w_Mask.%s" %(qmax,qmin,nth,frmt)
		pypl.savefig( out_fname + fig_name)
		print "\n Plot saved as %s " %fig_name
		del fig_name

	######################################################################
### GDrive/.../Exp_CXI-Martin/scripts/plot_a_corr.py:
#def get_cor(fname, detdist, Q=None, R=None, bins=None, wave=2.0695, plot=True):
    
#    data = np.load(fname)
#    corr_count = data["corr_count"]
#    corrsum = data["corrsum"]
#    sig = corrsum/corrsum[:,0][:,None] / corr_count 
#    rmin,rmax = data["rmin"], data["rmax"]

 #   radii = np.arange(rmin, rmax)
#    th = np.arctan( radii * 0.00010992/detdist)*.5

#    qs = 4 * np.pi * np.sin(th)/wave 

 #   print "Data bounds:"    
#    print qs.min(), qs.max()
#    if Q is not None:
#        idx = np.argmin( np.abs( qs - Q) )
#    else:
#        assert( R is not None)
#        idx = np.argmin( np.abs( radii-R) )
#        Q = qs[idx]
#    
#    phis = np.arange( sig.shape[1] ) * 2 * np.pi / sig.shape[1]
#    
#    cos_psi = np.cos( phis) * np.cos(th[idx])**2 + np.sin(th[idx])**2 

#    if plot:
#        plt.figure(figsize=(12,6))
#        if bins is not None:
#            plt.plot( cos_psi, sig[idx-np.int(bins/2):idx+np.int(bins/2)+1].mean(axis=0) )
#            plt.title("%s, correlation at Q=%.3f $\AA^{-1}$ (R=%d-%d pixels)" % (fname, Q, R - np.int(bins/2), R + np.int(bins/2)), fontsize=18)
#        else:
#            plt.plot( cos_psi, sig[idx] )
#            plt.title("%s, correlation at Q=%.3f $\AA^{-1}$ (R=%d pixels)" % (fname, Q, R), fontsize=18)
#        plt.xlabel(r"$\cos \,\psi$", fontsize=18)
#        plt.gca().tick_params(labelsize=15, length=9)
#        plt.show()
## ##
## ## ### GDrive/.../Exp_CXI-Martin/scripts/corpairs.py:
#rmin = 54; rmax = 1130 => Must re run radial Profile in LOKI + Auto Correlations
#chunksize = 50
#min_residual = 10