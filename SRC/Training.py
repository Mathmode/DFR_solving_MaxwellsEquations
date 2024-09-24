# -*- coding: utf-8 -*-
"""
=============================================================
Created on:    January 2023
Authors:      Jaime Taylor and Manuela Bastidas
Description:  "Neural Networks for Solving PDEs in H_0(curl) Space"
=============================================================
"""

import tensorflow as tf
import numpy as np

from SRC.Architecture import make_loss_model

# ========================================
# Training Loop for a H0(curl) problem
# ========================================
# This function trains the neural network model 

# The tricky_loss function returns the value to be minimized, while sol_err calculates
# the relative error in the appropriate norm.


def train_model(v_model, N, Nmodes, epsilon, mu, rhs, solution, iterations, 
                Nval, dimensions=[[0, np.pi], [0, np.pi]], dtype='float32'):
    """
    Trains the model for the specified number of iterations using the given options.
    
    Parameters:
        v_model: The neural network model to be trained.
        N: Number of integration points.
        Nmodes: Number of Fourier modes.
        epsilon, mu: Material parameters.
        rhs: The right-hand side of the equation.
        solution: The exact solution (for comparison).
        iterations: Number of iterations to train.
        Nval: Validation set size (if using validation).
        dimensions: Domain dimensions (default is [[0, pi], [0, pi]]).
        dtype: Data type for training (default is 'float32').
    
    Returns:
        history: The training history.
        loss_model: The trained loss model.
    """
    
    # Create the loss model
    loss_model = make_loss_model(v_model, N, Nmodes, epsilon, mu, rhs,
                                    solution, dimensions=dimensions, dtype='float32')
    
    # Custom loss function: only the first output (minimized loss) is used for training
    def tricky_loss(y_true, y_pred):
        return y_pred[0]
    
    # Custom metric: the second output is used for calculating relative error
    def sol_err(y_true, y_pred):
        return y_pred[1]
    
    metrics = [sol_err]  # Use relative error as the primary metric

    # ========================================
    # Optimizer 
    # ========================================
    optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
    
    # Adaptive learning rate callbacks can be added here if needed
    callbacks = []  # List of callbacks to modify training behavior

    # ========================================
    # Compile the Model
    # ========================================
    # Compile the loss model with the selected optimizer, custom loss function, and metrics
    loss_model.compile(optimizer=optimizer, loss=tricky_loss, metrics=metrics)

    # ========================================
    # Train the Model
    # ========================================
    # Start the training loop, fitting the model for the specified number of iterations
    # A single input/output pair of constants is used because the model learns through the loss model
    history = loss_model.fit(
        x=tf.constant([1.]),  # Dummy input
        y=tf.constant([1.]),  # Dummy output
        epochs=iterations,    # Number of iterations
        batch_size=1,         # Batch size of 1 (single training point)
        callbacks=callbacks    # List of callbacks (empty by default)
    )
    
    return history, loss_model  # Return the training history and the trained loss model
