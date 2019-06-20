#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#****************************************************************************************************************
# Import HDF5-files from calculations 
# Simulations based on for CsCl CXI-Martin run 18-105/119
# Currently tetsting with data from simulation of CsCl
# 2019-05-1?? v9 @ Caroline Dahlqvist cldah@kth.se
#			CrossCA_84-run_qvsq-fqc.py
#				Load previously calculated and stored CC-maps (2 halves of data) from 'CrossCA_84-run_qvsq.py' |(HDF5-files)
#					on different ranges of pdb-#(time instances) and evaluate Convergence through FQC [Kurta2017]
#				with plots of simularity eq.13 [Suppl, Kurta2017] in the  3D (Cross-) Correlations.
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


def read_and_plot_cc(list_names, out_name, mask):
	"""
	Read in CC data from multiple files in 'list_names' list of filenames. 
	Plot the average from all read-in files (saved in 'out_name'-location).
	"""

	## ---- Load Calculations from Files and Calculate the Mean : ---- ##
	t_load =time.time()
	
	#corr_sum = []
	cross_corr = []
	mask_corr = []
	tot_corr_sum = 0	## the number of ACC perfomed, one per loaded file ##
	tot_shot_sum = 0	## the total number of shots (diffraction pattersn) from simulation ##
	tot_diff_sum = 0	## the number of diffs or the number of patterns in input to ACC ##
	q_map= None
	nphi= None
	diff_type_str= None
	for file in list_names:
		print "\nFile being read: ", file
		with h5py.File(file, 'r') as f:
			if file==list_names[0]:
				q_map=np.asarray(f['q_mapping'])
				#set_Q_R(radii=q_map[:,2], qs= q_map[:,1], Q=args.Q , R=args.R)  ## Set the selected q-value to collect ##
				#q2_idx,q2_invA = set_Q_R(radii=q_map[:,2], qs= q_map[:,1], Q=args.Q , R=args.R) ## if with return statement ##
			dset_cc = f['cross-correlation_sum']#[q2_idx,:,:] ## Data-set with Cross-Correlations (3D); Select only q2 ##
			#ccsummed =np.asarray(dset_cc[q2_idx,:,:])		## for importing only a specific q-values CC  ##
			ccsummed =np.asarray(dset_cc)	## Read in ALL q_2, OBS, must limit the number of files to read in from se line 1291-1295 ##
			diff_count = dset_cc.attrs["tot_number_of_diffs"]
			shot_count = dset_cc.attrs["tot_number_patterns"]
			#tot_corr_sum= dset_cc.attrs["tot_number_of_corrs"]
			mask_crosscor = np.asarray(f['mask_cross-correlation'])
			if file==list_names[0]:
				nphi=int(np.asarray(f['num_phi'])) 
				diff_type_str =dset_cc.attrs["diff_type"]
				dtc_dist_m = dset_cc.attrs["detector_distance_m"]
				wl_A = dset_cc.attrs["wavelength_Angstrom"]
				ps_m = dset_cc.attrs["pixel_size_m"]
				be_eV = dset_cc.attrs["beam_energy_eV"] 
		cross_corr.append(ccsummed)  ## (2,Qidx,Q,phi)
		mask_corr.append(mask_crosscor)
		tot_corr_sum+= 1 ##corr_count ## calculate the number of auto-correlations loaded ##
		tot_shot_sum+= shot_count 
		tot_diff_sum+= diff_count 
	#tot_shot_sum=100*91 ## !OBS:Temp fix for forgotten to store data ##
	#cross_corr_sum  = np.sum(np.asarray( cross_corr_sum ), 0) ## sum of all the cc-summs  sum(91x(Q,Q,phi),0)=(Q,Q,phi)##
	#cross_corr_sum  = np.sum(np.vstack( cross_corr_sum ), 0) ## sum of all the cc-summs   sum(91x(Q,Q,phi),0)=(Q,Q,phi))##
	cross_corr = np.asarray(cross_corr)
	mask_corr = np.asarray(mask_corr)

	#print "\n Dim of intershell-CC (@ selected q2): ", cross_corr_sum.shape ##(500,360)
	print "\n Dim of intershell-CC : ", cross_corr.shape		# =(2, 500, 500, 360)
	t = time.time()-t_load
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Loading Time for %i patterns: "%(len(list_names)), t_m, "min, ", t_s, "s " #   0 min,  4.42866182327 s


	#if str(args.plt_set).lower() in ('single', 'all'):
	plot_cc(tot_corr_sum,q_map,nphi,diff_type_str,tot_shot_sum,dtc_dist_m,wl_A,ps_m,be_eV, cross_corr, mask_corr,  out_name)#, mask=mask) ## tot_shot_sum
	# if str(args.plt_set).lower() in ('subplot','all'):
	# 	subplot_cc(cross_corr_sum,mask_crosscor,tot_corr_sum,q_map,nphi,diff_type_str,tot_shot_sum, out_name, mask=mask) ## tot_shot_sum


	exit(0)  ## Finished ! ##

# ------------------------------------------------------------

def plot_cc(corr_count, q_map, nphi, pttrn, shot_count,dtc_dist_m,wl_A,ps_m,be_eV, cross_corrsum, mask_crosscor,  out_fname):#, mask=None):
	"""
	Read in CC data from multiple files in '' list of filenames. 
	Plot the average from all read-in files.

	In:
	corr_count 			 The total number of corrlations calculated in the 'corrsum',
							e.g. the total number of differeneses that was used to calculate the correlations.
	q_map 				The momentum transfer vaules, Qs', in pixels for which the correlations were calculated.
	nphi 				The angles for which the correlations were calcucated (e.g. 180 or 360).
	pttrn 				A string informing which differences were used in the correlation calcutaion, such as 
							if the correlatsion were calculated from pairwise diffreence or N-1 diffrens for N shots.
	corr_mask			The Auto-correlation of the mask.

	mask 				The Detector Mask (for plotting the intensity Pattern wth mask)
	shot_count 			The total number of shots summed in 'IP_mean' and loaded.
	cross_corrsum 		 ... (Qindex, Qs, phi)
	out_fname 			The pathname and prefix of the filename of the plot.

	Out:				Plot of the auto-correlation.
	"""
	
	t_calc_plt = time.time()

	## --- Font sizes and label-padding : ---- ##
	axis_fsize = 14		## Fontsize of axis labels ##
	tick_fsize = 12		## Size of Tick-labels ##
	tick_length = 9		## the Length of the ticks ##
	sb_size = 16 #18 	## Fontsize of Sub-titles ##
	sp_size = 18#20  	## Fontsize of Super-titles ##
	l_pad = 10 			## Padding for Axis Labels ##
	cb_shrink, cb_padd = 1.0, 0.2  ## Colorbar padding ##
	padC = 50#10#50 	# pad the edges of the correlation (remove self-correlation junk) /l.181,  ave_corrs-script has 50 ##

	if 'all' in pttrn : frac = 1#0.8 		## Limit the vmax with this percentage, for 2M need 1.0 ##
	else: frac = 1

	## ---- Local Parameters for all plots: ---- ##
	frmt = "eps" 	## Format used to save plots ##
	## columns 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
	qmin_pix, qmax_pix = q_map[0,2] , q_map[-1,2] ## first and last values in cloumn 2 = q [pixels] ##
	radii = q_map[:,2] 	 ## column 2: q [pixels]. ##
	qs = q_map[:,1]

	
	# ## ---- Calculate the average intensity in Q (row-bby-row): ---- ##
	# qs_mean= np.asarray([	corrsum_m[i,:].mean() for i in range(corrsum_m.shape[0])	])  ## (Q,phi) where row: Q; columns: phi ##
	# # qs_mean = corrsum_m.mean(axis=0) ## Row-wise mean ##
	# print "\n Dim of Qs mean: ", qs_mean.shape

	## interesting pixels/invA:  400 - 487/1.643 invAA : ##
	#r_pixel=425 	## approx 1.45 invAA ##
	#idx = np.argmin( np.abs(   radii-r_pixel  ) ) ## the index for the pixel ##

	## ---- Divide the Cross-Correlation(data) mapxQ2xQ1xphi at Q=idx with the Cross-Correlation(Mask) mapxQ2xQ1xphi: ---- ##
	cross_sum_m = np.divide(cross_corrsum, mask_crosscor, out=None, where=mask_crosscor!=0) ##corrsum /= corr_mask but not where corr_mask = 0  ##
	#cross_sum_m = cross_sum_m[q2_idx] ## not needed if already selected q2 ##
	#print "\n Dim of 'cross_sum_m': ", cross_sum_m.shape


	## ---- Normalise the CC with the MASK || MAsk & Variance {0th angle; cross_sum_m[:,0][:,None]}: ---- ##
	sig_cc = cross_corrsum/corr_count 
	#sig_cc_var = cross_sum/cross_sum[:,0][:,None]/corr_count

	## ---- FFT if the CC- data: ---- ##
	sig_cc_Aft =  fft(sig_cc, axis =-1)
	#sig_cc_var_Aft =  fft(sig_cc_var, axis =-1)
	del sig_cc#, sig_cc_var

	## ---- For the whole range, summed over all Qs : ----- ##
	## eq.14 [Suppl, Kurta2017]:CC= sum_{q1<q}( F-CC_1(q1,q) *  conjugate[F-CC_2(q1,q)] ) + sum_{q2<q}( F-CC_1(q2,q) *  conjugate[F-CC_2(q2,q)] )  ##
	CC_12 =  np.sum( sig_cc_Aft[0]*np.conjugate( sig_cc_Aft[1] ) ,-2)  + np.sum( sig_cc_Aft[1]*np.conjugate( sig_cc_Aft[0] ) ,-3) 
	CC_11 = np.sum( sig_cc_Aft[0]*np.conjugate( sig_cc_Aft[0] ) ,-2)  + np.sum( sig_cc_Aft[0]*np.conjugate( sig_cc_Aft[0] ) ,-3) 
	CC_22 = np.sum( sig_cc_Aft[1]*np.conjugate( sig_cc_Aft[1] ) ,-2)  + np.sum( sig_cc_Aft[1]*np.conjugate( sig_cc_Aft[1] ) ,-3) 
	#cc = np.asarray([	np.sum( sig_cc_Aft[i]*np.conjugate( sig_cc_Aft[i] ) ,-2)  + np.sum( sig_cc_Aft[i]*np.conjugate( sig_cc_Aft[i] ) ,-3) for i in range(2)	])
	FQC = np.abs(CC_12)/ np.sqrt( CC_11*CC_22  ) ##  n = 2, 4, 6, 8, 10, 12 ## eq.13 [Suppl, Kurta2017] ##
	print "Dim of FQC:", FQC.shape  # = Dim of FQC: (500, 360)
	del CC_12, CC_11, CC_22

	## ---- Summed in parts with the Q-bins : ----- ##
	#FQC_q =[]
	## eq.14 [Suppl, Kurta2017]:CC= sum_{q1<q}( F-CC_1(q1,q) *  conjugate[F-CC_2(q1,q)] ) + sum_{q2<q}( F-CC_1(q2,q) *  conjugate[F-CC_2(q2,q)] )  ##
	for q in range(sig_cc_Aft.shape[1]):		# sig_cc.shape[1] = Q2
		## sig_cc_Aft[map, Q2, Q1, n] ##
		## 	v3:	only choosen q-value in other q-dim:	 ##	 # selecting 0,:,q,: 4D->2D
		CC_12 = np.sum( sig_cc_Aft[0,q,0:q+1,:]*np.conjugate( sig_cc_Aft[1,q,0:q+1,:] ) ,-2)  + np.sum( sig_cc_Aft[1,0:q,q,:]*np.conjugate( sig_cc_Aft[0,0:q,q,:] ) ,-2)
		CC_11 = np.sum( sig_cc_Aft[0,q,0:q+1,:]*np.conjugate( sig_cc_Aft[0,q,0:q+1,:] ) ,-2)  + np.sum( sig_cc_Aft[0,0:q,q,:]*np.conjugate( sig_cc_Aft[0,0:q,q,:] ) ,-2) 
		CC_22 = np.sum( sig_cc_Aft[1,q,0:q+1,:]*np.conjugate( sig_cc_Aft[1,q,0:q+1,:] ) ,-2)  + np.sum( sig_cc_Aft[1,0:q,q,:]*np.conjugate( sig_cc_Aft[1,0:q,q,:] ) ,-2) 
		## ValueError: 'axis' entry is out of bounds
		print "Dim of CC-12:", CC_12.shape #=Dim of CC-12: (500, 360)
		fqc = np.abs(CC_12)/ np.sqrt( CC_11*CC_22  )  ## eq.13 [Suppl, Kurta2017] ##
		FQC_q.append(fqc)
		#FQC_q.append(fqc)
		del fqc, CC_12, CC_11, CC_22
	FQC_q = np.asarray(FQC_q)
	#FQC = np.abs(CC_12)/ np.sqrt( CC_11*CC_22  ) ##  n = 2, 4, 6, 8, 10, 12 ## eq.13 [Suppl, Kurta2017] ##
	print "Dim of FQC_q:", FQC_q.shape #Dim of FQC: v2= (500, 360)

	#print "Dim of FQC_q:", FQC_q.shape
	
	## ---- Save the calculations: ---- ##
	out_hdf = h5py.File( out_fname + '_calc.hdf5', 'w') 
	out_hdf.create_dataset( 'fqc', data = np.asarray(FQC))
	out_hdf.create_dataset( 'fqc_q', data = np.asarray(FQC_q))
	#out_hdf.create_dataset( 'fqc_q', data = np.asarray(FQC_q))
	out_hdf.create_dataset( 'q_mapping', data = q_map)
	#out_hdf.create_dataset( 'number_of_maps_compared', data = corr_count)

	out_hdf.close()
	print "\n File Closed! \n"

	#set_Q_R(radii=q_map[:,2], qs= q_map[:,1], Q=args.Q , R=args.R)  ## Set the selected q-value to collect ##
		
	def set_axis_invAng(fig_ax, q_map, q_ax='x'):
		'''
		Set the axis to inverse Angstrom from the saved range in q_map
		===========================================================
		fig_ax 		axis handle of figure to adjust
		q_map 		the array (bins, 3) with the distances; 0:index, 1: inverse Angstrom, 2: pixels
		q_ax 		which axis to adjust: 'x', 'y' or 'xy' (=both x- and y- axis)
		'''
		## ---- Number of q radial polar bins, where q_map is an array q_min-q_max ---- ##
		nq = q_map.shape[0]-1	
		#if q_ax=='x':	Nticks= fig_ax.get_yticks().shape[0]
		#elif q_ax=='y':	Nticks= fig_ax.get_yticks().shape[0]
		#elif q_ax=='xy':	Nticks= fig_ax.get_xticks().shape[0]
		if q_ax[0]=='x':	Nticks= fig_ax.get_xticks().shape[0]
		if q_ax[0]=='y':	Nticks= fig_ax.get_yticks().shape[0]
		#elif q_ax=='xy':	Nticks= fig_ax.get_xticks().shape[0]

		q_bins = np.linspace(0, nq, num= Nticks, dtype=int)  ## index of axis tick vector ##
		q_ticks = np.array([ q_map[x,2] for x in q_bins] ) ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
		#print "q_label: ", q_label

		if q_ax=='x':
			fig_ax.set_xticklabels(q_label), fig_ax.set_xticks(q_bins)
			fig_ax.tick_params(axis='x', labelsize=tick_fsize, length=tick_length)
		elif q_ax=='y':
			fig_ax.set_yticklabels(q_label), fig_ax.set_yticks(q_bins)
			fig_ax.tick_params(axis='y', labelsize=tick_fsize, length=tick_length)
		elif q_ax=='xy': 
			fig_ax.set_yticklabels(q_label), fig_ax.set_yticks(q_bins)
			fig_ax.set_xticklabels(q_label), fig_ax.set_xticks(q_bins)
			fig_ax.tick_params(axis='both', labelsize=tick_fsize, length=tick_length)
		return fig_ax

	def plot_even_odd(fig, data, norm_name, fig_name, lim):
		"""
		Plot the even and odd coefficients separately in 
		side-by-side plots
			========================
		fig			the figure to plot in, figure handle
		data 		the calculated FFT-coefficients to plot, 2D array
		norm_name 	the normalisations except for the mask, string.
		filename	part of figure name unique to plot, string
		lim 		The number of coefficients to include
		"""
		
		## ---- Phi even-odd = [1,20]: ---- ##
		#lim=21#13 	## the number of coefficients to plot. 51 is too many !
		ax_even=fig.add_subplot(121)
		ax_odd=fig.add_subplot(122, sharex=ax_even, sharey=ax_even)
		pypl.subplots_adjust(wspace=0.2, hspace=0.2, left=0.08, right=0.95, top=0.85, bottom=0.15)
		cmap = pypl.get_cmap('jet')
		for i in range(1,lim):  
			color = cmap(float(i)/lim)
			if i % 2 == 0:		## only even i:s ##
				#ax_even.plot(data[:,i].real, c=color, lw=2, label= 'n=%i' %i)#'Angle $ \phi=%i $' %i)
				ax_even.plot(data[:,i], c=color, lw=2, label= 'n=%i' %i)#'Angle $ \phi=%i $' %i)
				ax_even.set_title("even n:s")
				ax_even.legend()
				q_bins = np.linspace(0, (q_map.shape[0]-1), num= ax_even.get_xticks().shape[0], dtype=int)  ## index of axis tick vector ##
				q_label = [ '%.3f'%(q_map[x,1]) for x in q_bins] ## 0: indices, 1: r [inv Angstrom], 2: q[pixels] ##
				ax_even.set_xticklabels(q_label), ax_even.set_xticks(q_bins)
				ax_even.set_xlabel(r'q', fontsize = axis_fsize)
				ax_even.set_ylabel('FQC$^n$(q)', fontsize = axis_fsize)

			else:				## only odd ##
				#ax_odd.plot(data[:,i].real, c=color, lw=2, label= 'n=%i' %i)#'Angle $ \phi=%i $' %i)
				ax_odd.plot(data[:,i], c=color, lw=2, label= 'n=%i' %i)#'Angle $ \phi=%i $' %i)
				ax_odd.set_title("odd n:s")
				ax_odd.legend()
				ax_odd.set_xticklabels(ax_even.get_xticklabels()), ax_odd.set_xticks(ax_even.get_xticks())
				ax_odd.set_xlabel(r'q', fontsize = axis_fsize)
				ax_odd.set_ylabel('FQC$^n$(q)', fontsize = axis_fsize)
		pypl.subplots_adjust(wspace=0.3, hspace=0.2, left=0.08, right=0.95, top=0.85, bottom=0.15)
			############################### [fig.] End ##############################
		#pypl.title('Re{FFT} $\phi [1,%i] $ of Average of %d corrs [%s] \n(Normalized with Mask)'%(lim,corr_count, pttrn),  fontsize=sb_size)
		#pypl.suptitle("Re{FFT} $\phi [1,%i]$ of Average of %d corrs for $q_2= %.2f$ [%s] \n (Normalized with Mask %s)"%(lim,corr_count,q2_invA, pttrn, norm_name),  fontsize=sb_size)
		pypl.suptitle("FFT $\phi [1,%i]$ of Average of %d corrs [%s] \n (Normalized with Mask %s)"%(lim,corr_count, pttrn, norm_name),  fontsize=sb_size)
		#pypl.show()
		#fig_name = fig_name + 'phi-1-%i_(qx-%i_qi-%i_nphi-%i)_%s.%s'%(lim,qmax_pix,qmin_pix, nphi ,pttrn,frmt)
		fig.savefig( out_fname + fig_name) # prefix = '/%s_%s_' %(name,pdb), out_fname = os.path.join( outdir, prefix)
		print "\n Subplot saved as %s " %fig_name
		del fig_name, data, norm_name 	# clear up memory
		gc.collect() 
		pypl.cla() ## clears axis ##
		pypl.clf() ## clears figure ##


	print "\n Plotting..."
	##################################################################################################################
	#------ Fig.19 FGC  eq.13 [Suppl, Kurta2017] of CC Normalised with correlated Mask &  || variance: --------------#
	##################################################################################################################
	##################################################################################################################
	## ---- FQC (FFT-angle) even-odd = [1,20]: ---- ##
	fig19 = pypl.figure('FQC', figsize=(22,10)) ## height 8inch too low for math mode with sqrt ##
	lim=21 ## plot the 20 coefficients after the 0th ##
	#fig19 = pypl.figure('FQC', figsize=(11,8)); lim=13
	fig_name = "_q1_vs_q2_FQC-%i_(qx-%i_qi-%i_nphi-%i)_%s.%s"%(lim,qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	norm_name=' '
	title="Re{FFT} $\phi [1,%i]$ of Average of %d corrs [%s] \n (Normalized with Mask %s)"%(lim,corr_count, pttrn, norm_name)
	#plot_even_odd(fig=fig19, data=FQC, q_map=q_map, fig_name=fig_name, title=title, out_fname=out_fname,lim=lim) ## if outside current function ##
	plot_even_odd(fig=fig19 , data=FQC, norm_name=norm_name,  fig_name=fig_name, lim=lim)
	## if <<1 then the maps are substantially different


	t = time.time()-t_calc_plt
	t_m =int(t)/60
	t_s=t-t_m*60
	print "\n Total Time for Load and Plot: ", t_m, "min, ", t_s, "s " #  47 min,  59.9696178436 s

	##################################################################################################################
	#------ Fig.20 FGC_q  eq.13 [Suppl, Kurta2017] of CC Normalised with correlated Mask &  || variance: --------------#
	##################################################################################################################
	##################################################################################################################
	## ---- FQC (FFT-angle) even-odd = [1,20]: ---- ##
	#fig20 = pypl.figure('FQC', figsize=(22,10)) ## height 8inch too low for math mode with sqrt ##
	fig20 = pypl.figure('FQC_q', figsize=(11,5)) ## height 8inch too low for math mode with sqrt ##

	lim=21 ## plot the 20 coefficients after the 0th ##
	fig_name = "_q1_vs_q2_FQC_qs-%i_(qx-%i_qi-%i_nphi-%i)_%s.%s"%(lim,qmax_pix,qmin_pix, nphi ,pttrn,frmt)
	norm_name=' '
	title="Re{FFT} $\phi [1,%i]$ of Average of %d corrs [%s] \n (Normalized with Mask %s)"%(lim,corr_count, pttrn, norm_name)
	plot_even_odd(fig=fig20 , data=FQC_q, norm_name=norm_name,  fig_name=fig_name, lim=lim)

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
	start_i = fnames[0].split('_(')[0].split('_')[-1].split('-')[0].split('M')[-1] 	## '...6M0-6M41_(none-sprd0)...'=> '0' ##
	end_i = fnames[0].split('_(')[0].split('_')[-1].split('-')[-1].split('M')[-1] ## '...6M0 -6M41_(none-sprd0)...'=> '41' ##
	## Segment-CC__Pnoise_BeamNarrInt_84-119_4M8_(poisson-sprd0)_Int.hdf5
	if len(fnames)>1:
		start_f = fnames[-1].split('_(')[0].split('_')[-1].split('-')[0].split('M')[-1]
		end_f = fnames[-1].split('_(')[0].split('_')[-1].split('-')[-1].split('M')[-1]
		#cncntr_start_f = fnames[-1].split('_(')[0].split('_')[-1].split('-')[0]
		#cncntr_end_f = fnames[-1].split('_(')[0].split('_')[-1].split('-')[-1]

		## end points of file number range ##
		#if start_i<start_f: start_file=start_i
		#else:	start_file=start_f
		#if end_i<end_f: end_file=end_f
		#else:	end_file=end_i
		#pdb= cncntr +'M' + start_file +'-'+ cncntr +'M'+ end_file

		pdb= cncntr +'M' + start_i +'-'+  end_i + '_' + cncntr +'M'+ start_f +'-'+  end_f 
	else :	pdb = cncntr
	#run=fnames[0].split('84-')[-1][0:3] 		## 3rd index excluded ##
	#noisy = fnames[0].split('_ed_(')[-1].split('-sprd')[0]
	noisy = fnames[0].split('_(')[-1].split('-sprd')[0] ## if not '4M90_ed' but '4M90' in name
	#n_spread = fnames[0].split('-sprd')[-1].split(')_')[0] ## for files that ends with '_#N.cxi' ##
	n_spread = fnames[0].split('-sprd')[-1].split(')')[0] ## for files that ends with '.cxi' ##
	name = fnames[0].split('/')[-1].split('_(')[0].split('_')[0:2]
	name = name[0]+name[1]
	pttrn = fnames[0].split(')_')[-1].split('.hdf5')[0]
	## /.../noisefree_Beam-NarrInt_84-105_6M90_ed_(none-sprd0)_#5.cxi
	# /.../Fnoise_Beam-NarrNarrInt_6M0-6M41_(none-sprd0)_tot-from-42-files_Cross-corr_Int.hdf5
	
	# ---- Generate a Storage File: ---- ##
	prefix = '%s_%s_(%s-sprd%s)_FQC' %(name,pdb,noisy,n_spread) ## Final Result ##
	out_fname = os.path.join( outdir, prefix) 
	#out_hdf = h5py.File( out_fname + 'tot_Auto-corr_%s.hdf5' %(pttrn), 'a')    # a: append, w: write
	#out_hdf = h5py.File( out_fname + 'tot-from-%i-files_Cross-corr_%s.hdf5' %(numb,pttrn), 'w')    # a: append, w: write
	print "\n Data Analysis with LOKI."
	
	## ---- Read in CC from Files and Plot ----  ##
	#read_and_plot_cc(fnames, out_hdf, out_fname, mask_better) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]
	read_and_plot_cc(fnames, out_fname, mask_better) ##  qmin_pix=args.q_range[0], qmax_pix=args.q_range[1]

##################################################################################################################
##################################################################################################################

## --------------- ARGPARSE START ---------------- ##
parser = argparse.ArgumentParser(description="Analyse Simulated Diffraction Patterns Through Auto-Correlations.")

# parser.add_argument('-d', '--dir-name', dest='dir_name', required='True', type=str,
#       help="The Location of the directory with the cxi-files (the input).")

parser.add_argument('-o', '--outpath', dest='outpath', default='this_dir', type=str, help="Path for output, Plots and Data-files")

subparsers = parser.add_subparsers()#title='calculate', help='commands for calculations help')

# ## ---- For Calculating the ACC of sile #X and storing in separat hdf5-file: -----##
# subparsers_calc = subparsers.add_parser('calculate', help='commands for calculations ')
# subparsers_calc.add_argument('-d', '--dir-name', dest='dir_name', default='this_dir', type=str,
#       help="The Location of the directory with the cxi-files (the input).")
# subparsers_calc.add_argument('-s', '--simulation-number', dest='sim_n', default=None, type=int, 
#       help="The number of the pdb-file, which simulation is to be loaded from 0 to 90, e.g. '20' for the file <name>_4M20_<properties>l.cxi")
# #subparsers_e = subparsers.add_parser('e', help='a choices')
# subparsers_calc.add_argument('-q', '--q-range', dest='q_range', nargs=2, default="0 850", type=int, 
#       help="The Pixel range to caluate polar images and their correlations. Default is '0 850'.") ##type=float)
# subparsers_calc.add_argument('-e', '--exposure', dest='exp_set', default='pair', type=str, choices=['pair', 'diffs', 'all', 'all-dc', 'all-pairs'],
#       help="Select how to auto-correalte the data: 'pair' (pair-wise difference between shots),'diffs', 'all' (all shot without difference), 'all-dc' (all shot without difference, difference done by correlation script).")
# # subparsers_calc.add_argument('-m', '--masked', dest='w_MASK', default=True, type=lambda s: (str(s).lower() in ['false', 'f', 'no', 'n', '0']),
# #       help="Select if the mask is included in the auto-correalte calculations.")
# # subparsers_calc.add_argument('-m', '--unmasked', dest='w_MASK', action='store_false',
# #       help="Select if the mask is included in the auto-correalte calculations.")
# subparsers_calc.add_argument('-m', dest='w_MASK', action='store_true')
# subparsers_calc.add_argument('--no-mask', dest='w_MASK', action='store_false')
# subparsers_calc.set_defaults(w_MASK=True)
# subparsers_calc.set_defaults(func=load_and_calculate_from_cxi)


## ---- For Loading Nx# files, Summarise and Plotting the total result: -----##
subparsers_plt = subparsers.add_parser('plots', help='commands for plotting ')
subparsers_plt.add_argument('-p', '--plot', dest='plt_set', default='single', type=str, choices=['single', 'subplot', 'all'],
      help="Select which plots to execute and save: 'single' each plot separately,'subplot' only a sublpot, 'all' both single plots and a subplot.")

parser_group = subparsers_plt.add_mutually_exclusive_group(required=True)
parser_group.add_argument('-R', '--R', dest='R', default=None, type=int ,
      help="Which radial pixel to look at.")
parser_group.add_argument('-Q', '--Q', dest='Q', default=None, type=float ,
      help="Which reciprocal space coordinate to look at.")

subparsers_plt.set_defaults(func=load_and_plot_from_hdf5)



args = parser.parse_args()
args.func(args) ## if .set_defaults(func=) SUBPARSERS ##
## ---------------- ARGPARSE  END ----------------- ##