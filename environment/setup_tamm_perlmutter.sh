#!/usr/bin/env bash

# Load the GNU programming environment
module load PrgEnv-gnu

module load gcc

# Load the Cray MPICH library for MPI support
module load cray-mpich

# Load CUDA toolkit
module load cpe-cuda
module load cudatoolkit
module unload craype-accel-nvidia80

# Source the baseline Perlmutter settings from the main environment directory
source ./environment/setup_perlmutter.sh

# Link against dynamic libraries
export CRAYPE_LINK_TYPE=dynamic

# Disable MPICH's native GPU support (use OFI instead)
export MPICH_GPU_SUPPORT_ENABLED=0

# Fix one OpenMP thread per MPI rank by default
export OMP_NUM_THREADS=1

# Skip NIC symmetry tests in MPICH‐OFI
export MPICH_OFI_SKIP_NIC_SYMMETRY_TEST=1

# Enable verbose OFI diagnostics (optional; you can unset if too chatty)
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_VERBOSE=1

# Tell TAMM how many GA progress ranks per node
export GA_NUM_PROGRESS_RANKS_PER_NODE=1
export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1

export PPn=4

