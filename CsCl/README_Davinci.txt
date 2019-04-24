######################################################################################
######################################################################################
				README
######################################################################################
######################################################################################

1. Path to Condor-simulated files in CXI-format of CsCl:  
/home/cldah/cldah-scratch/condor_cxi_files/ ...


100 shots per pdb-file for all 91 PDB files of 4 Molar conentration with 1 nm beam-focus:
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmfocus/
	with sub-folders for noise-free detector and poisson noise in detector:
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmfocus/Fnoise_Beam-NarrNarrInt/ 
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmfocus/Pnoise_Beam-NarrNarrInt/ 


100 shots per pdb-file for all 91 PDB files of 6 Molar conentration with 1 nm beam-focus:
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmf6M/
	with sub-folders for noise-free detector and poisson noise in detector:
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmf6M/Fnoise_Beam-NarrNarrInt/ 
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmf6M/Pnoise_Beam-NarrNarrInt/ 


100 shots per pdb-file for all 91 PDB files of 2 Molar conentration with 1 nm beam-focus:
/home/cldah/cldah-scratch/condor_cxi_files/N-100-91x1nmf2M/



early simulations but with the wrong atom in pdb-files
5 shots each from concentrations 4M0 and 6M90 with and without poisson noise in the detector:
/home/cldah/cldah-scratch/condor_cxi_files/test_N-5/
######################################################################################
######################################################################################
######################################################################################

2. Paths to plots and calculations of Auto-correlation (of Condor-simulated files in CXI-format of CsCl) and intensity patterns (diffraction patterns):
/home/cldah/cldah-scratch/CCA_RadProf_plots/ ...
 

ERR_column_C_in_PDB/:
100 shots from 4M0-pdb simulation A:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-1/
          
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-2/

100 shots per pdb for 2 pdb files with poisson noise in the detector:           
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-2pdbPssn/

100 shots per pub for 2 pdb files with a beam-focus of 1 nm, noise-free detector:    
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-2x1nmfocus/  

100 shots per pub for 2 pdb files with a beam-focus of 1 nm, detector with poisson noise: 
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-2x1nmfocusPssn/


100 shots per pdb-file for all 91 PDB files of 4 Molar conentration with 3 nm beam-focus, noise-free detector:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x3nmfocus/Fnoise_Beam-NarrInt/N-100-2x48pdb/
	and without Mask:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x3nmfocus/Fnoise_Beam-NarrInt/no_mask/
	and with auto-correlation of ALL data and not pair-wise differences :
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x3nmfocus/Fnoise_Beam-NarrInt/not_diffs/




Under the following directories there are 1-3 subfolders: 
all/  		Auto-correlation from all shots
pair/  		Auto-correlation from shot pair differences
shot_sample/	Plot of 3 shots; Row 1: Intensity Pattern, Row 2: Amplitude Pattern (absolute values), Row 3: Patterson Image (or autocorrelation image in real space)


100 shots per pdb-file for all 91 PDB files of 4 Molar conentration with 1 nm beam-focus, noisefree detector:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x1nmfocus/Fnoise_Beam-NarrInt/
	and without Mask:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x1nmfocus/Fnoise_Beam-NarrInt/no_mask/


100 shots per pdb-file for all 91 PDB files of 6 Molar conentration with 1 nm beam-focus, noise free in the detector:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x1nmf6M/Pnoise_Beam-NarrNarrInt/


100 shots per pdb-file for all 91 PDB files of 6 Molar conentration with 1 nm beam-focus, Poisson noise in the detector:
/home/cldah/cldah-scratch/CCA_RadProf_plots/N-100-91x1nmf6M/Pnoise_Beam-NarrNarrInt/


######################################################################################
######################################################################################
