#*************************************************************************
# run Simulations in Python2.7 with Condor (v1.0) for CsCl CXI-Martin run 18-119
# N number of diffraction patterns are simulated and can be either plotted directly
# with 'plotting != false' and/or saved to a '.cxi'-file
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#           '..._data-nabs_...' without taking the modulus in the plot
#            and as a log10 plot '..._log10_...' and are in reciprocal space
#  Projection patterns '..._rs_...' in real space are from inv Fourier transform of data
#              (and FFT-shifted)
#   Patterson_Image '..._patterson_image_...' are from FFTshift-Fast Fourier transforms-FFTshift
#           of Intensity Patterns
#   CXI-file and/or plots are stored in sub-folder 'test_results'
# PDB id: 4M0 CRYST1   33.019   33.019   35.771  90.00  90.00  90.00 P 1           1
#       CsCl    C:4.09 mol/dm3 (experiment),  Molecular Weight:168.355 g/mol
#       water   C: 55.5 mol/dm3   
#   Ratio of C: 4.09/55.5 ??
# Currently tetsting with pdb-file 1AON from Condor-examples
# 2019-01-11 v3 No Mask, only Write || only plot, ONLy Plot @ Caroline Dahlqvist cldah@kth.se
#
# Applying the standard Condor source file in Python-site-package, as in test_simple_ex.py
#*************************************************************************
import numpy
from numpy.fft import fftn, fftshift # no need to use numpy.fftn/fftshift
# %pylab
import condor
import os,time #os = for saving in specific dir; time= measure time
this_dir = os.path.dirname(os.path.realpath(__file__)) # from Condor/example.py, get run-directory

# from Condor/example.py
# Import plotting form matlab but include Exceptions = Errors :
try:
    import matplotlib.pyplot as pypl
    plotting = True
except Exception as e:
    print(str(e))
    plotting = False

# Import  Logging for Condor for debugging code:
import logging
logger = logging.getLogger("condor")
logger.setLevel("DEBUG")
#logger.setLevel("WARNING")
#logger.setLevel("INFO")   # informs user of "Missed Particle"

#plotting = False    # comment out to save data in plots directly
writing = False

# ---- Construct X-ray Source Instance at 9.5 keV: -----
#   (highest values  f_d = 0.1-0.2: p_e = 3.E-3 )
photon = condor.utils.photon.Photon(energy_eV = 9500.)  # [eV]
src = condor.Source(wavelength=photon.get_wavelength(), focus_diameter=200E-9, pulse_energy=1.E-3, profile_model= "gaussian")

# ---- Construct Detector Instance (distance= 200mm, 150 mm, 1516x1516 pixels): -----
#   det = condor.Detector(distance=0.15, pixel_size=110E-06, nx=1516, ny=1516, noise = "poisson")
N = 1       # number of diffraction patterns to simulate
ps = 110    # [um] pixel size in um= 1E-6m
pxls = 512    # number of pixels, 128 (N=1:Runtime=48s), 256 (N=1:Runtime=3min), 512 (N=3:Runtime= 4.6 h),  1024 (N=1:Runtime= 1h 20min, N=3: 12h [Write+plot]), 1516 (N=3:Runtime=32h [plot], 11 h [Write])
x_gap = 0   # [pixels] gap size isn x-dim
noisy = None # Noise type: {None, "poisson", ”normal"=gaussian, "normal_poisson" = gaussian and poisson}
# # (if 'normal'||'normal_poisson' additional argument noise_spread is required [photons])
n_spread = None

if noisy=="normal" or noisy=="normal_poisson": n_spread = 0.005000  # online GUI start at 0,5; tried: 0,000005
det = condor.Detector(distance=0.15, pixel_size=ps*1E-6, nx=pxls-x_gap, ny=pxls, x_gap_size_in_pixel = x_gap, noise = noisy, noise_spread = n_spread)
# --- Detector with Mask: ----
# det = condor.Detector(distance=0.15, pixel_size=ps*1E-6, nx=pxls, ny=pxls_y, noise = noisy, noise_spread = n_spread, mask = mask_array)
#	? is nx and ny necessary if numpy-array-Mask is applied?
# det = condor.Detector(distance=0.15, pixel_size=ps*1E-6, noise = noisy, noise_spread = n_spread, mask = mask_array)
t_det_i = time.time()


# ---- Construct Particle_Atoms Instance of (CsCl) in Water: ----
#   (instance values borrowed from fig6 in Condor-article[Hatanke, 2016 IUCr Condor])
ratio = 1     # ratio of Molecule/Salt : Water
#pdb= "4M0"      # 92 structure-files for each concentration. Tried: "4M0", 6M90, "4M0_test", "4M0_test2"
#pdb_file = "./CsCl-PDB/%s" %(pdb)    # the PDB-id name, "./CsCl-PDB/name"
pdb= "6M90_ed" #"4M90_ed" #"4M0_ed"   # Original Files lack column # 78 with 1st letter of Atom
pdb_file = "./CsCl-PDB_ed/%s" %(pdb)    # the PDB-id name, "./CsCl-PDB/name"
par ={
	# CsCl in water from simulated pdb:
    "particle_atoms_cscl" :
        condor.ParticleAtoms (number = ratio, arrival = "random", pdb_filename = "%s.pdb" %pdb_file, rotation_formalism="random", position_variation = None)
    }
    # Single Proteins of GroEL–GroES complex from pdb: 1AON (pdb_id = "1AON" or pdb_filename = "1AON.pdb"):
    #"particle_atoms_0" :
    #    condor.ParticleAtoms (number = ratio, arrival = "random", pdb_filename = "%s.pdb" %pdb_file, rotation_formalism="random", position_variation = "uniform", position_spread=[50E-9, 50E-9, 1E-9]),
    #}# Water Droplets:
#   "particle_spheroid_0" :
#   condor.ParticleSpheroid(number=1-ratio, arrival="random", diameter = 8E-9, diameter_variation="normal", diameter_spread=2E-9, flattening = 0.9, flattening_variation = "uniform", flattening_spread = 0.2, material_type ="water", position_variation = "uniform", position_spread = [50E-9, 50E-9, 1E-9]),
#   }


# Combine source, detector and particle by constructing an Experiment Instance:
Exp = condor.Experiment(src, par, det)
t_exp = time.time() # time marker for measuring propagation-time


# Output File:
if noisy is None: noisy = "none" # since None is valid input
if n_spread is None: n_spread = "0" # since None is valid input
if writing: W = condor.utils.cxiwriter.CXIWriter("./test_results/test_condor_84-119_%s_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.cxi"  %(pdb,ratio*10,ps,pxls,noisy,n_spread,N)) 

# Simulate N Images:
for i in range(N):	#require indent for loop
    t_loop = time.time() # time marker for measuring propagation-time
    # Propagate the Experiment = Generate a Diffraction Pattern:
    res = Exp.propagate()
    # ---- Calculate Diffrraction Pattern and Obtain Results in a Dicionary: ----
    #intensity_pattern = res["entry_1"]["data_1"]["data"]
    #amplitudes_pattern = res["entry_1"]["data_1"]["data_fourier"]
    #real_space = numpy.fft.fftshift(numpy.fft.ifftn(res["entry_1"]["data_1"]["data_fourier"]))
    #res["real_space"] = real_space      # same as "projected_image" but inverted
    #res["patterson_image"] = fftshift(fftn(fftshift(intensity_pattern)))
    #res["projection_image"] = fftshift(fftn(fftshift(amplitudes_pattern)))

    # Write results to Output File:
    if writing: W.write(res)	# Write the result (Dictionary) to a CXI-file; comment out if only wants direct plots saved
    #	print("Writing File Time:", time.time()-t_loop)  #  {-- if Pyton3 --}: 
    print "Writing File Time:", time.time()-t_loop #   {-- Pyton2.7 --}
    # ---- Plot each Pattern: ----
    if plotting:
        frmt = "eps" #{png, pdf, ps, eps, svg} # File Formats for SAVING
        intensity_pattern = res["entry_1"]["data_1"]["data"]
        amplitudes_pattern = res["entry_1"]["data_1"]["data_fourier"]
        # ---- Save Intensity Patterns in Plots:  "Intensity Patterns" = |"Amplitude Patterns"|^2
        pypl.imsave(this_dir + "/test_results/test_%s_log10_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), numpy.log10(intensity_pattern), format=frmt)
        pypl.imsave(this_dir + "/test_results/test_%s_data-nabs_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), intensity_pattern, format=frmt)
        # ---- Save Reconstructed Image: ----
        #pypl.imsave(this_dir + "/test_results/test_%s_rs_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt),  abs(real_space), format=frmt) # "Reconstructed Image"
    #pypl.imsave(this_dir + "/test_%s_projection_image_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt),  abs(res["projection_image"]))    # "Projection Image"
        # ---- Save "Pattersson" Image: ----
        #pypl.imsave(this_dir + "/test_results//test_results/test_%s_patterson_image_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), abs(res["patterson_image"]), format=frmt)
if writing: W.close()  #comment out if only wants direct plots saved


print "\t =>finished!"     #{-- if Pyton3 --}: print("\t =>finished!")
t_prop=time.time()-t_exp    # [s] Propagation Time measured from Exp construction
if t_prop%3600:                  # Print in h, min, s
    t_h = int(t_prop)/3600
    t_m =int(t_prop-(t_h)*3600)/60
    t_s = t_prop-t_h*3600-t_m*60
    print "Propagation Time: ", t_h, "h, ", t_m, " min", t_s, " s"
elif t_prop%60:                  # or Print in min, s
    t_m =int(t_prop)/60
    t_s=t_prop-t_m*60
    print "Propagation Time: ", t_m, "min, ", t_s, " s"
