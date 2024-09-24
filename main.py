# -*- coding: utf-8 -*-
"""
=============================================================
Created on:    January 2023
Authors:      Jaime Taylor and Manuela Bastidas
Description:  "Neural Networks for Solving PDEs in H_0(curl) Space"
=============================================================
"""


# ========================================
# Import Python Packages
# ========================================
import tensorflow as tf
import numpy as np
import os

# Import local modules
from SRC.Architecture import make_u_model  # Model architecture
from SRC.Training import train_model       # Training loop
from SRC.PostProcessing import post_processing_loss, post_processing_solution

# ========================================
# Set Random Seed for Reproducibility
# ========================================
# Ensures that the results are reproducible by fixing the random seed
tf.random.set_seed(1234)
np.random.seed(1234)
tf.keras.utils.set_random_seed(1234)

# ========================================
# Set Global Data Type to float32
# ========================================
# Configures TensorFlow to use float32 throughout the training process
dtype = 'float32'
tf.keras.backend.set_floatx(dtype)

# ========================================
# Parameters and Problem Settings
# ========================================
# Define problem-specific parameters and create the necessary directories
iterations = 1000            # Number of training iterations
Nmodes = [50, 50]            # Number of Fourier modes in each dimension
nx, ny = 50, 50              # Grid resolution in the x and y directions
N = [nx, ny]                 # Integration points

# Validation settings (if used), with 17% more points for validation
Nval = [int(nx * 1.17), int(ny * 1.17)]

# Create a directory for saving the solution if it doesn't already exist
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, 'solution')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

# ========================================
# Problem Definition: Import Problem-Specific Functions
# ========================================
# Import the specific problem functions (v1_exact, v2_exact, gradbit_exact, F1, F2, etc.)
from Problems.Problem1 import *

# ========================================
# Create Neural Network Model (u_model)
# ========================================
# The neural network is defined using the architecture from make_u_model
# n_neurons: number of neurons per hidden layer
# n_layers: number of hidden layers
v_model = make_u_model(n_neurons=20, n_layers=5, dimensions=dimensions, dtype=dtype)

# ========================================
# Define the Solution and RHS (Right-Hand Side) Functions
# ========================================
# The solution list contains the exact components for comparison (v1_exact, v2_exact, gradbit_exact)
solution = [v1_exact, v2_exact, gradbit_exact]

# The rhs list contains the right-hand side components of the problem (F1 and F2)
rhs = [F1, F2]

# Additional solution-related objects (if needed)
sol_objects = [v1_exact, v2_exact, gradbit_exact, dimensions]

# ========================================
# Model Training
# ========================================
# Train the neural network using the train_model function

history, loss_model = train_model(v_model, N, Nmodes, epsilon, mu, rhs, solution,
    iterations, Nval, dimensions=dimensions, dtype=dtype)


# =================================
# POSTPROCESSING SOLUTION AND LOSS
# =================================

post_processing_solution(v_model,sol_objects,dtype)

np.save(f'{final_directory}/my_history.npy',history.history)

post_processing_loss(history.history,nx,ny,150,final_directory)

