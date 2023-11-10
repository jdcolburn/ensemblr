# ensemblr
python notebook with workflow for sorting AF2 reduced-MSA ensembles to find intermediate states, paths between end-states, etc. 

# How to use
Just set the base directory and the "structures" directory to where your structure ensemble is located (e.g. the output from colabfold_local: https://github.com/YoshitakaMo/localcolabfold.git). Also provide two reference structures (i.e. end states) with which to compare your ensemble. Choose appropriate settings in the first cell and execute.

The workflow will run PCA, compute some properties (RMSDs and SASA), and run MC simulated annealing to find the best possible path between your end states (or any states), along some collective variable. You can use this to find occluded/intermediate state models by selecting from structures along this path. 