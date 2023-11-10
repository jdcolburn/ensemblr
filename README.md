# ensemblr
SBCB python notebook with workflow for sorting AF2 reduced-MSA ensembles to find intermediate states, paths between end-states, etc. 

## Dependencies
Packages used:

- numpy
- pandas
- MDAnalysis
- Biopython
- matplotlib
- seaborn
- nglview

## How to use
Just set the base directory (which is where outputs are written), and set the "structures" directory to where your ensemble pdb files are located (e.g. the output directory from colabfold: https://github.com/YoshitakaMo/localcolabfold). Make sure these directories aren't the same. Also provide two reference structures (i.e. end states) with which to compare your ensemble. Choose appropriate settings in the first cell and execute.

The workflow will run PCA, compute some properties (RMSDs and SASA), and run MC simulated annealing to find the best possible path between your end states (or any states), along some collective variable. You can use this to find occluded/intermediate state models by selecting from structures along this path. 