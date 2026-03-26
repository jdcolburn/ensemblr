# bin/bash

# uses local colabfold to generate ensemble of predictions 
# output filestructure is compatible with the python workflow

input="input.fasta"
desired_models_per_run=2000

seed="$RANDOM"
num_seeds=$((desired_models_per_run / 5))

flags="--random-seed $seed --num-seeds $num_seeds --num-models 5 --num-recycle 1 --use-dropout --amber --relax-max-iterations 100 --use-gpu-relax --rank plddt"

echo $flags

# range of MSA depths
colabfold_batch $input output_4-8       $flags --max-msa 4:8
colabfold_batch $input output_8-16      $flags --max-msa 8:16
colabfold_batch $input output_16-32     $flags --max-msa 16:32
colabfold_batch $input output_32-64     $flags --max-msa 32:64

