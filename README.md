# XCCA-simulations
Simulations of X-ray Cross-Correlation Analysis from CsCl
Diffraction patterns generated with Condor from FXI-hub
..............................................................

script in CsCl/

AutoCCA_84-run_v5.py
Calculate the Auto-Correlations with ‘Loki’ and plot, including the mean intensity pattern, Fourier-coefficients, detector masks AC, subplot of; mean intensity patter, detector mask’s AC and AC of polar images. Data loaded from CXI-file
-plot ’single’ :  17
-plot ‘subplot’ :  1
-plot ‘all’ : 18

CCA_RadProf_84-run_v2.py
Calculate the Radial Profile, from CXI-file, with ‘Loki’ and the Auto-Correlation. Plot the result including som e of the diffraction patterns.
subplots : 2 

CrossCA_84-run_v7.py
Calculate and plot the Cross-Correlations from CXI-file. Plots of Fourier Coefficients for selected q_2 values (from command line) and CC of detector mask and cos(psi) with psi as the reciprocal space angle.
plots : 5

fix_all_pdb_for_condor.py
Add the missing columns 77-78 in the PDB-files. Condor requires these columns.

Plot_diffraction.py
Plots 3 diffraction patterns from CXI-file.
subplot : 1

RadProf_84-run_v2.py
Load one CXI-file and calculate the Radial profile. plot together with with the mean intensity pattern. There are multiple options for the subplots, e.g. 2 radial profile plots; one of the second half of the profile and one of the entire profile.
subplot : 1

run-CsCl-w-condor-in-terminal.py
Python script for Command Line Arguments to pass to ‘simulate_CsCl_84-X.py’

simulate_CsCl_84-X.py
Simulate diffraction experiment with Condor 

test_gnoise_asics.py
Load the detector mask and then locate the ASICs or Tiles. Generate normal distributed noise per ASIC (or Tile) and plot.
subplot : 1

run_2loop_loki_slurm.sh
Slurm script for the Davinci-cluster:
runs AutoCCA_84-run_v5.py or 
CrossCA_84-run_v7.py  for each CXI file in designated folder. Runs all calculation in parallel, one job for each CXI-file. After no more jobs are queued, runs the plot version of the script which reads all the separate calculations (storesd in HDF5-files), sum and plot the result

run_CCA_slurm.sh
Slurm script for the Davinci-cluster: runs RadProf_84-run_v2.py

run_sim_slurm_new.sh
Slurm script for the Davinci-cluster:
runs the Condor simulation script, simulate_CsCl_84-X.py, for each PDB-file in designated folder (in parallell)

run_plot_diff_slurm.sh 
Slurm script for the Davinci-cluster: runs Plot_diffraction.py


script in CsCl/test_result


CCA_cxi_84-X_v3.py
Load a CXI-file. Plot some the diffraction patterns in a subplot (intensity, amplitude and pattersson), calculate CCA with ‘Loki’ or ‘CXILT14’ and plot the results. Option to make one quadrant noisy (3 options /implementations of noise). Plot: 1-14

run_CCA_84_script.py
Python script for Command Line Arguments to pass to ‘CCA_cxil_84-X_v3.py’

read_CXI_84-119_v3.py
Load a CXI-file and plot. Several plotting options. plots : 6
