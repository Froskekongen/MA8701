#!/bin/sh
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="ea_omniglot_classifier"
#SBATCH --output=omniglot.classified
#SBATCH --partition=EPIC2

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "$PWD"

source ${WORKDIR}/load_tf_modules.sh
source ${WORKDIR}/ttenv/bin/activate

python ${WORKDIR}/simple_image_classifier/omniglot_classifier --test_run=False

deactivate
