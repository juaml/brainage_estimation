#!/bin/bash
#source ~/.venvs/py3smore/bin/activate
#OMP_NUM_THREADS=5 python3 $@
#deactivate

if [ $# -lt 2 ]; then
    echo "This script is meant to run a command within a python environment"
    echo "It needs at least 2 parameters."
    echo "The first one must be the environment name."
    echo "The rest will be the command"
    exit 1
fi

env_name=$1
echo "Activating ${env_name}"
source ~/.venvs/${env_name}/bin/activate
shift 1
echo "Running ${@} in virtual environment"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

$@


