#*************************************************************************
# run Simulations in Python2.7 with Condor (v1.0) for CsCl CXI-Martin run 18-105/119
# N number of diffraction patterns are simulated and can be either plotted directly
# with 'plotting != false' and/or saved to a '.cxi'-file
#   plots saved to choosen file-format 'frmt'
#   Intensity Pattern = abs(Amplitude_Pattern)^2) are plotted directly as
#           '..._data-nabs_...' without taking the modulus in the plot
#            and as a log10 plot '..._log10_...' and are in reciprocal space
#    Intensity Patterns =  AutoCorrelated image (can be used as initial guess for phae retrieval)
#   CXI-file and/or plots are stored in sub-folder 'test_results'
# PDB id: 4M0/6M0 CRYST1   33.019   33.019   35.771  90.00  90.00  90.00 P 1           1
#       CsCl    C:4.09 mol/dm3 (experiment),  Molecular Weight:168.355 g/mol
#       water   C: 55.5 mol/dm3   
#   Ratio of C: 4.09/55.5 ??
# 2019-02-14 v6 with Mask (from exp) Write only, no Plots, no F-tr. Store Source properties. 
#               With Argparser
#        @ Caroline Dahlqvist cldah@kth.se
#       Read, F-transforms & Plots with /test_results/read_cxi_84-119_v3 or CCA_cxi_84-119_v2
# Applying the standard Condor source file in Python-site-package, as in test_simple_ex.py
#*************************************************************************
import argparse
import numpy
from numpy.fft import fftn, fftshift # no need to use numpy.fftn/fftshift
# %pylab
import condor
import os,time #os = for saving in specific dir; time= measure time
this_dir = os.path.dirname(os.path.realpath(__file__)) # from Condor/example.py, get run-directory

# from Condor/example.py
# ----- Import plotting form matlab but include Exceptions = Errors : -----
try:
    import matplotlib.pyplot as pypl
    plotting = True
except Exception as e:
    print(str(e))
    plotting = False

# ----- Import  Logging for Condor for debugging code: -----
import logging
logger = logging.getLogger("condor")
#logger.setLevel("DEBUG")
#logger.setLevel("WARNING")
logger.setLevel("INFO")   ## informs user of "Missed Particle" ##


# ----- Import Arguments from Command Line: -----
parser = argparse.ArgumentParser(description= "Simulate a FEL Experiment with Condor for Silulated CsCl Molecules in Water Solution. ")

#parser.add_argument('-r', '--run-number', dest='run_numb', required=True, type=str, help="The Name of the Experiment run to Simulate, e.g. '84-119'.")
#parser.add_argument('-r', '--run-number', dest='run_numb', required=True, type=int, help="The Number Assigned to the Experiment run to Simulate, e.g. '119' for '84-119'.")
parser.add_argument('-f', '--fname', dest='sim_name', default='test_mask', type=str, help="The Name of the Simulation.")
parser_group = parser.add_mutually_exclusive_group(required=True)
parser_group.add_argument('-pdb','--pdb-name', dest='pdb_name', default='4M0', type=str, help="The Name of the PDB-file to Simulate, e.g. '4M0' without file extension.")
parser_group.add_argument('-pdbp','--pdb-path', dest='pdb_path', default=None, type=str, help="The the PDB-file to Simulate, including path, e.g. './CsCl-PDB_ed/4M0' without file extension.")

parser.add_argument('-n', '--n-shots', dest='number_of_shots', required=True, type=int, help="The Number of Diffraction Patterns to Simulate.")

parser.add_argument('-dn', '--dtctr-noise', dest='dtct_noise', default=None, type=str, help="The Type of Detector Noise to Simulate: None, 'poisson', 'normal'=gaussian, 'normal_poisson' = gaussian and poisson")
parser.add_argument('-dns', '--dtctr-noise-spread', dest='nose_spread', default=None, type=float, help="The Spread of Detector Noise to Simulate, if Noise is of Gaussian Type, e.g. 0.005000.")


args = parser.parse_args()
## args.'dest' == args.'long-call-name'

# ----- Set if Plotting and/or Writing to CXI-file : -----
plotting = False    # comment out to save data in plots directly
writing = True

# ---- Name Secific Run: ----
name = args.sim_name#"test_mask"  ## Name of Simulation ##
#run = args.run_numb #"84-119"     ## Run number to compare with in Experiment at SLAC ##
# run = "84-%i" %(str(args.run_numb))


# ----- Construct X-ray Source Instance at 9.5 keV: -----
#   (highest values  f_d = 0.1-0.2: p_e = 3.E-3 )
photon_energy_eV = 9500.
photon = condor.utils.photon.Photon(energy_eV = photon_energy_eV)  # [eV]
#src = condor.Source(wavelength=photon.get_wavelength(), focus_diameter=200E-9, pulse_energy=1.E-3, profile_model= "gaussian")
src = condor.Source(wavelength=photon.get_wavelength(), focus_diameter=3E-9, pulse_energy=1.E-1, profile_model= "gaussian")

# ----- Construct Detector Instance (distance= 200mm, 150 mm, 1516x1516 pixels): -----
#   det = condor.Detector(distance=0.15, pixel_size=110E-06, nx=1516, ny=1516, noise = "poisson")
N = args.number_of_shots #5#3#1         # number of diffraction patterns to simulate
#   N=5  1 h,  40  min 54.5528719425  s
ps = 110        # [um] pixel size in um= 1E-6m
dtc_dist = 0.15 # [m] Detector Distance from Sample
#pxls = 1738    # number of pixels, 128 (N=1:Runtime=48s), 256 (N=1:Runtime=3min), 512 (N=3:Runtime= 4.6 h),  1024 (N=1:Runtime= 1h 20min, N=3: 12h [Write+plot]), 1516 (N=3:Runtime=32h [plot], 11 h [Write])
#pxls_y = 1742 # from mask size (1738x1742)
#x_gap = 0      # [pixels] gap size isn x-dim
#h_dia = 0      # [pixels] diameter of central hole
noisy = args.dtct_noise     # Noise type: {None, "poisson", "normal"=gaussian, "normal_poisson" = gaussian and poisson}
# # (if 'normal'||'normal_poisson' additional argument noise_spread is required [photons])
n_spread = args.nose_spread #None
#if noisy=="normal" or noisy=="normal_poisson": n_spread = 0.005000  # online GUI start at 0,5; tried: 0,000005
mask_file = "./masks/better_mask-assembled.npy"	# Mask file from Experiment [CXI-Martin run 18-119] size (1738x1742)
#mask_file = "better_mask-assembled.npy" # Mask file from Experiment [CXI-Martin run 18-119] (1738x1742)
mask_array = numpy.load(mask_file)
#det = condor.Detector(distance=0.15, pixel_size=ps*1E-6, nx=pxls-x_gap, ny=pxls, x_gap_size_in_pixel = x_gap, hole_diameter_in_pixel=h_dia, noise = noisy, noise_spread = n_spread)

# --- Detector with Mask: ----
det = condor.Detector(distance=dtc_dist, pixel_size=ps*1E-6, noise = noisy, noise_spread = n_spread, mask = mask_array)
t_det_i = time.time()


# ----- Construct Particle_Atoms Instance of (CsCl solution): -----
#   (instance values borrowed from fig6 in Condor-article[Hatanke, 2016 IUCr Condor])
ratio = 1     # ratio of Molecule/Salt : Water; in this case the PDB-file already includes the water
if args.pdb_path is None: 
    pdb= args.pdb_name # "4M0_ed"      # 92 structure-files for each concentration.(XXX_ed 78th column had to be added)
    pdb_file ="./CsCl-PDB_ed/%s" %(pdb)    # the PDB-id name, "./CsCl-PDB/name"
else: 
    pdb_file = args.pdb_path    # the PDB-id name including the path, "./CsCl-PDB/4M0"
    parts = args.pdb_path.split('/') 
    pdb =  parts[-1]
cncntr = pdb.split('M')[0] ## Find which concentration was used and match to Experiment name ##
assert (cncntr == "4" or cncntr == "6"),("Incorrect concentration of crystals, pdb-file must start with '4' or '6'!")
if cncntr == "4": run = "84-119"
else : run ="84-105"

par ={
	# CsCl in water from simulated pdb:
    "particle_atoms_cscl" :
        condor.ParticleAtoms (number = ratio, arrival = "random", pdb_filename = "%s.pdb" %pdb_file, rotation_formalism="random", position_variation = None)
    }


# ----- Combine source, detector and particle by constructing an Experiment Instance: -----
Exp = condor.Experiment(src, par, det)
t_exp = time.time() # time marker for measuring propagation-time


# ---- Make an Output Directory: ----
#outdir = this_dir +'/%s_%s_%s_(%s-sprd%s)_#%i/' %(name,run,pdb,noisy,n_spread,N)
out = this_dir +'/simulation_results/' 
if not os.path.exists(out):
    os.makedirs(out)
#prefix = 
#out_fname = os.path.join( outdir, prefix)

# ----- Output File: -----
if noisy is None: noisy = "none" # since None is valid input
if n_spread is None: n_spread = "0" # since None is valid input
#if writing: W = condor.utils.cxiwriter.CXIWriter("./test_results/%s_84-119_%s_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.cxi"  %(name,pdb,ratio*10,ps,pxls,noisy,n_spread,N))
#if writing: W = condor.utils.cxiwriter.CXIWriter("./simulation_results/%s_%s_%s_(%s-sprd%s)_#%i.cxi"  %(name,run,pdb,noisy,n_spread,N)) # for reading ps, pxls from file; with ratio, ps & pxls = constant
if writing: W = condor.utils.cxiwriter.CXIWriter(out+ "/%s_%s_%s_(%s-sprd%s)_#%i.cxi"  %(name,run,pdb,noisy,n_spread,N)) # for reading ps, pxls from file; with ratio, ps & pxls = constant

# ---- Make an Output Directory for PLOTS: ----
if plotting:
    outdir = this_dir +'/%s_%s_%s_(%s-sprd%s)_#%i/' %(name,run,pdb,noisy,n_spread,N)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

# ----- Simulate N Images: -----
for i in range(N):	#require indent for loop
    t_loop = time.time() # time marker for measuring propagation-time
    # ----- Propagate the Experiment = Generate a Diffraction Pattern: -----
    res = Exp.propagate()
    # Calculate Diffrraction Pattern and Obtain Results in a Dicionary:
    #intensity_pattern = res["entry_1"]["data_1"]["data"]
    #amplitudes_pattern = res["entry_1"]["data_1"]["data_fourier"]
    #real_space = numpy.fft.fftshift(numpy.fft.ifftn(res["entry_1"]["data_1"]["data_fourier"]))
    #res["real_space"] = real_space      # same as "projected_image" but inverted
    #res["patterson_image"] = fftshift(fftn(fftshift(intensity_pattern)))
    #res["projection_image"] = fftshift(fftn(fftshift(amplitudes_pattern)))
    res["source"]["incident_energy"] = photon_energy_eV                 #[eV]
    res["source"]["incident_wavelength"] = photon.get_wavelength()      #[m]
    res["detector"]["pixel_size_um"] = ps                               #[um]
    res["detector"]["detector_dist_m"] = dtc_dist                       #[m]

    # ----- Write results to Output File: -----
    if writing: W.write(res)	# Write the result (Dictionary) to a CXI-file
    #	print("Writing File Time:", time.time()-t_loop)  #  {-- if Pyton3 --}: 
    print "Writing File Time:", time.time()-t_loop #   {-- Pyton2.7 --}
    # ---- Plot each Pattern: ----
    if plotting:
        frmt = "eps" #{png, pdf, ps, eps, svg} # File Formats for SAVING
        intensity_pattern = res["entry_1"]["data_1"]["data"]
        amplitudes_pattern = res["entry_1"]["data_1"]["data_fourier"]
        # ---- Save Intensity Patterns in Plots:  "Intensity Patterns" = |"Amplitude Patterns"|^2
        pypl.imsave( outdir  + "%s_%s_log10_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(name,pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), numpy.log10(intensity_pattern), format=frmt)
        pypl.imsave( outdir  + "%s_%s_data-nabs_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(name,pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), intensity_pattern, format=frmt)
        # ---- Save Reconstructed Image: ----
        #pypl.imsave( outdir + "%s_%s_rs_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(name,pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt),  abs(real_space), format=frmt) # "Reconstructed Image"
    #pypl.imsave( outdir + "%s_%s_projection_image_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(name,pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt),  abs(res["projection_image"]))    # "Projection Image"
        # ---- Save "Pattersson" Image: ----
        #pypl.imsave(outdir  + "%s_%s_patterson_image_(r%iof10-ps%ium-ny%i-%s-sprd%s)_#%i.%s" %(name,pdb,ratio*10,ps,pxls,noisy,n_spread,i,frmt), abs(res["patterson_image"]), format=frmt)
if writing: W.close()

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
