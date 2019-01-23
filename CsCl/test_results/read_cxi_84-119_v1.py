#*************************************************************************
# Import CXI-files from simulations with Condor (v1.0) and plot data
# cxi-files located in the same folder, result saved  in subfolder "plots--"
# File format is optiona, indexing start with '0' 
# Currently tetsting with data from simulation eith 1AON from Condor examples
# 2019-01-11 v1 @ Caroline Dahlqvist cldah@kth.se
# lynch path: ipython /test_results/read_cxi_84-119_v1.py
# 
#*************************************************************************

import h5py 
import numpy as np
import matplotlib.pyplot as pypl
# %pylab	# code as in Matlab
import os
this_dir = os.path.dirname(os.path.realpath(__file__)) # Get path of directory

# ----	Parameters unique to file: ----
frmt = "eps"
pdb= "6M90"      # 92 structure-files for each concentration. Tried: "4M0"
rt = 1		# Ratio of particles (if 1: only CsCl loaded, if != 1: mxture with Water-particle)
ps = 110	# [um] pixel width
pxls = 1516	# number of Pixels
noisy = "none"
n_spread = 0
N = 2		# Number of iterations performed = Number of Diffraction Patterns Simulated

intensity_pattern = []	# Empty list
#with h5py.File(this_dir +'/test_condor_84-119__%s_(r%i-ps%ium-ny%i-%s-sprd%s)_#%i.cxi'%(pdb,rt*10,ps,pxls,noisy,n_spread,N), 'r') as f:
with h5py.File(this_dir +'/test_condor_84-119_%s_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.cxi'%(pdb,rt*10,ps,pxls,noisy,n_spread,N), 'r') as f:
#with h5py.File(this_dir +'/test_condor_84-119__1AON_(r9-ps110um-ny1024-none-sprd0)_#3.cxi', 'r') as f:
		intensity_pattern = np.asarray(f["entry_1/data_1/data"])
		amplitudes_pattern = np.asarray(f["entry_1/data_1/data_fourier"])
		patterson_image = np.asarray(f["patterson_image"]) # fftshift(fftn(fftshift(intensity_pattern)))
		projection_image = np.asarray(f["projection_image"]) #fftshift(fftn(fftshift(amplitudes_pattern)))
		#real_space = np.asarray(f["real_space"]) #nump.fft.fftshift(numpy.fft.ifftn(res["entry_1"]["data_1"]["data_fourier"])) #not saved in 'test_1AON_84-119_v2'
# fftshift: Shift the zero-frequency component to the center of the spectrum
# np.fft.ifftn Compute the N-dimensional inverse discrete Fourier Transform
print "Number of Patterns Reorded: ", len(intensity_pattern) # lenght of list = # Patterns

# From "Reading CXI files -- Condor":
#print("Maximum intensity value in first pattern: %f photons" % intensity_pattern[0].max())
#print("Maximum intensity value in second pattern: %f photons" % intensity_pattern[1].max())
for i in range(len(intensity_pattern)):
	print("Maximum intensity value in %i pattern: %f photons" % (i,intensity_pattern[i].max()) )

# ------------------- Calculate the Absolute value for the Plots: -------------------
I_p_abs = np.abs(intensity_pattern) 	# all Patterns in Array; 1st <name>[0], 2st <name>[1]
I_p0 = np.abs(intensity_pattern[0])
I_p1 = np.abs(intensity_pattern[1])
#I_p2 = np.abs(intensity_pattern[2])	# 3rd Pattern

Pr_i_abs =np.abs(projection_image)

#print I_p
#print pypl.I_p.shape	# Python tells size of Array


# ---- Select which Plots tp view in Run: ----
plt_I0, plt_I0_zoom, plt_I0_log = True, False, False
plt_Ip_sub, plt_Ip_Pr_sub, plt_Ip_sub_log = False, True, True	# Subplots


# --- Show Intensity Pattern of 1st Pattern (Full Image): ----
if plt_I0:
#pypl.imshow(I_p.reshape(201,201), interpolation='nearest', vmax=500000)
	fig_int0_1 = pypl.figure()
	#pypl.imshow(I_p0)
	px2= pxls/2	# Half the pixel size for center 0
	pypl.imshow(I_p_abs[0], extent=[-px2,px2,-px2,px2])
	#pypl.ylim([450,580])
	#pypl.xlim([450,580])
	pypl.ylabel('Pixels')
	pypl.xlabel('Pixels')
	cb = pypl.colorbar()
	cb.set_label('Intensity')
	pypl.show()	# works ok

# --- Show Intensity Pattern of 1st Pattern (Zoom Image): ----
if plt_I0_zoom:
	fig_int0_2 = pypl.figure()
	#pypl.imshow(I_p_abs[0])
	px2= pxls/2	# Half the pixel size for center 0
	pypl.imshow(I_p_abs[0], extent=[-px2,px2,-px2,px2])
	zx, zy = 50, 50
	pypl.ylim([-zx,zx])
	pypl.xlim([-zy,zy])
	pypl.ylabel('y [Pixels]')
	pypl.xlabel('x [Pixels]')
	cb = pypl.colorbar()
	cb.set_label('Intensity')
	pypl.show()	# works ok

# --- Show Intensity Pattern of 1st Pattern (Log10 Image): ----
if plt_I0_log:
	fig_int0_log = pypl.figure()
	#pypl.imshow(I_p_abs[0])
	pypl.imshow(np.log10(I_p_abs[0]))
	pypl.ylabel('Pixels')
	pypl.xlabel('Pixels')
	cb = pypl.colorbar()
	cb.set_label(r'Intensity log$_{10}$')
	pypl.show()	# works ok

# #########################################################################
# -------- SUBPLOTS: --------
# --- Subplot with Show N=3 Intensity Pattern : ----
if plt_Ip_sub:
	px2= pxls/2	# Half the pixel size for center 0
	zx, zy = 50, 50
	#pypl.subplots_adjust(hspace=0.5, left=0.07, right=0.95)
	pypl.subplots_adjust(wspace=0.5, left=0.07, right=0.95)
	for i in range(N):		#Projection Images (I_p_abs)
		pypl.subplot(2,3,i+1) # unneccessary if fig_sp, axs = pypl.subplots(ncols = 3, ...)
		ax = pypl.gca()
		im1 = ax.imshow(I_p_abs[i], extent=[-px2,px2,-px2,px2])
		ax.set_xlim([-zx, zx])	# Zoom at centre
		ax.set_ylim([-zy, zy])	# Zoom acentret 
		ax.set_ylabel('y Pixles')
		ax.set_xlabel('x Pixles')
		cb = pypl.colorbar(im1, orientation="horizontal", shrink=0.9)	# work but wrong scale
		cb.set_label(r'Intensity ')
	pypl.show()

# --- Subplot with Show N=3 Intensity Pattern & Projected Image: ----
if plt_Ip_Pr_sub:
	px2= pxls/2	# Half the pixel size for center 0
	zx, zy = 50, 50		# Zoom to pxl dist do in x and y direcction, from centre,
	#print "Length of projection_image: ", len(projection_image[0]) 
	print "Shape of projection_image: ",  projection_image[0].shape
	# ----- Rescaling Projection axis (real space): ---- OBS! need scaling factor
	rs = 1.0/ps 	#need float; Condor_publication fig.5: 1AON (GroEL-GroES): ~20 nm
	print "convert: ", rs 	# WRONG !!
	# rs = len(projection_image[0])/2
	#pypl.subplots_adjust(hspace=0.5, left=0.07, right=0.95)
	pypl.subplots_adjust(wspace=0.5, hspace=0.2, left=0.07, right=0.95)
	cb_shrink = 0.9		# Shrinkage of ColorBar; alter length of colorbar prop to image
	cb_padd = 0.1		# Padding of ColorBar; prevent overlap with axes
	for i in range(N):		#1st Row: Projection Images (I_p_abs)
		pypl.subplot(2,N,i+1) # unneccessary if fig_sp, axs = pypl.subplots(ncols = 3, ...)
		ax = pypl.gca()
		im1 = ax.imshow(I_p_abs[i], extent=[-px2,px2,-px2,px2]) #extent = scaling
		ax.set_xlim([-zx, zx])	# Zoom at centre
		ax.set_ylim([-zy, zy])	# Zoom acentret 
		ax.set_ylabel('y Pixles')
		ax.set_xlabel('x Pixles')
		cb = pypl.colorbar(im1, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)	# work but wrong scale
		cb.set_label(r'Intensity ')
	for i in range(N):	#2nd Row: Projection Images (Pr_i_abs)
		pypl.subplot(2,N, N+i+1) # unneccessary if fig_sp, axs = pypl.subplots(ncols = 3, ...)
		ax = pypl.gca()
		im1 = ax.imshow(Pr_i_abs[i], extent=[-px2*rs,px2*rs,-px2*rs,px2*rs])#, WRONG
		ax.set_ylabel('y [um]')
		ax.set_xlabel(r'x [$\mu$m]')
		cb = pypl.colorbar(im1, orientation="horizontal", shrink=cb_shrink, pad= cb_padd)	# work but wrong scale
		cb.set_label(r'Electron Densitu [a.u.]')
	pypl.show()
		#fig_sp.savefig(this_dir + "/test_results/test_%s_diff-rs_subplot_(-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ps,pxls,noisy,n_spread,N,frmt))


# --- Subplot with Show N=3 log10(Intensity Pattern ): ----  (Loop, independen of N value)
if plt_Ip_sub_log :
	px2= pxls/2	# Half the pixel size for center 0
	#from matplotlib.colors import LogNorm
	#from matplotlib.ticker import LogLocator
	fig_sp, axs = pypl.subplots(ncols = N, sharey=True) # (1,3), sharex = True,sharey=True
	#fig_sp.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
	fig_sp.subplots_adjust(wspace=0.5, hspace=0.2,  left=0.07, right=0.95)#, bottom=0.1, top=0.95) 
	cb_shrink = 0.8		# Shrinkage of ColorBar; alter length of colorbar prop to image
	cb_padd = 0.1		# Padding of ColorBar; prevent overlap with axes
	for i in range(N):
		im=axs[i].imshow(np.log10(I_p_abs[i]), extent=[-px2,px2,-px2,px2])
		#		For plotting in log10 without np.log10 calculation first, by using LogNorm & LogLocator (colorbar with 10^x):
		#im = axs[i].imshow(I_p_abs[i], extent=[-px2,px2,-px2,px2], norm=LogNorm()) # extent or  aspect ='equal'
		#cb = fig_sp.colorbar(im, ax=axs[i], ticks=LogLocator(), shrink=cb_shrink, pad= cb_padd, label=r'$\log_{10}\ $')  # ticks=None, should yield small ticks (NOTWOKRING)
		#cb.update_ticks()
		axs[i].set_xlabel('x Pixles')
		cb = fig_sp.colorbar(im, ax=axs[i],shrink=cb_shrink, pad= cb_padd, label=r'$\log_{10}\ $')
	axs[0].set_ylabel('y Pixles')
	# adjust spacing between subplots so  title and  tick labels don't overlap
	fig_sp.tight_layout()
	pypl.show()
	#fig_sp.savefig(this_dir + "/test_results/test_%s_log10_subplot_(-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ps,pxls,noisy,n_spread,N,frmt))


# --- Plot data in one row of Pixels of 1st Pattern : ----
#	from matplotlib.pyplot import figure,draw	#use functions draw() & figure()
#	fig1 = figure()	# if load %pylab or from matplotlib import figure, draw
#fig_int0_plt = pypl.figure()	# Intensity Profile 
#ax = fig_int0_plt.gca()
#ax.plot(I_p0[:,0], lw=2) # 1st diff pattern or I_p_abs[0,:,0] e.g. (ip[:,0], ip[:,1], lw=2)
#ax.set_ylabel('Intensity')
#pypl.xlabel('Pixels') #	e.g. pypl.xlabel(r'$q \ \ [\AA^{-1}]$')
#pypl.show()	# or if previous plt: pypl.draw()

# (pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt
#pypl.imsave(this_dir + "/test_%s_log10_(r%i-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), numpy.log10(res["entry_1"]["data_1"]["data"]), format=frmt)
#pypl.imsave(this_dir + "/test_%s_data-nabs_(r%i-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), intensity_pattern, format=frmt)
 #  Save Reconstructed Image:
#pypl.imsave(this_dir + "/test_%s_rs_(r%i-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt),  abs(real_space), format=frmt) # "Reconstructed Image"
####pypl.imsave(this_dir + "/test_1AON_projection_image.png",  abs(res["projection_image"]))    # "Projection Image"  == abs (real_space)
#  Save "Pattersson" Image:
#pypl.imsave(this_dir + "/test_%s_patterson_image_(r%i-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), abs(res["patterson_image"]), format=frmt)

# e.g. save in svg: fg = figure()  fg.savefig('<pah/name.svg>', bbox_inches = 'tight')
# e.g. pypl.savefig('<pah/name.svg>')

