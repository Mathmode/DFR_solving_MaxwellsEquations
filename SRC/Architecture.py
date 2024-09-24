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

from SRC.Loss import cross_loss, cross_loss_3D

# ========================================
# Create the Candidate Solution as a Neural Network
# ========================================
# This function constructs a fully connected feed-forward neural network 
# with a specified number of layers, neurons, and activation functions. 
# The output layer applies a custom cutoff to enforce H_0(curl) boundary conditions.

def make_u_model(n_neurons=20, n_layers=5, activation='tanh', dimout=2, 
                 dimensions=[[0, np.pi], [0, np.pi]], dtype='float32'):
    """
    Constructs a neural network model with custom cutoff layers.
    
    Parameters:
        n_neurons: Number of neurons per hidden layer
        n_layers: Total number of layers
        activation: Activation function (e.g., 'tanh')
        dimout: Output dimension (codomain of the function)
        dimensions: Domain dimensions (default is [[0, pi], [0, pi]])
        dtype: Data type of the layers (default is 'float32')
    
    Returns:
        A compiled Keras model.
    """

    # Input layer: shape corresponds to the number of dimensions
    xvals = tf.keras.layers.Input(shape=(len(dimensions),), name='x_input', dtype=dtype)

    # First hidden layer
    l1 = tf.keras.layers.Dense(n_neurons, activation=activation, name='l1', dtype=dtype)(xvals)

    # Create additional hidden layers
    for i in range(n_layers - 3):
        name = 'l' + str(i + 2)
        l1 = tf.keras.layers.Dense(n_neurons, activation=activation, name=name, dtype=dtype)(l1)

    # Last hidden layer
    l1 = tf.keras.layers.Dense(n_neurons, activation=activation, name='last_hidden', dtype=dtype)(l1)

    # Output layer before applying cutoff
    proto_output = tf.keras.layers.Dense(dimout, name="proto_output", dtype=dtype)(l1)

    # Apply cutoff based on dimension (2D or 3D)
    if len(dimensions) == 2:
        bare_output = cross_cutoff(dimensions=dimensions)([xvals, proto_output])
    elif len(dimensions) == 3:
        bare_output = cross_cutoff_3d(dimensions=dimensions)([xvals, proto_output])

    # Create and compile the model
    u_model = tf.keras.Model(inputs=xvals, outputs=bare_output, name='u_model')
    
    # Model summary for debugging
    u_model.summary()

    return u_model

# ========================================
# Cutoff Layer for 2D: Enforcing H_0(curl) Boundary Conditions
# ========================================
# This custom layer applies a cutoff to the neural network output to enforce 
# H_0(curl) boundary conditions in a 2D domain (e.g., [0, pi] x [0, pi]).

class cross_cutoff(tf.keras.layers.Layer):
    def __init__(self, dimensions=[[0, np.pi], [0, np.pi]], **kwargs):
        super(cross_cutoff, self).__init__()
        [[self.a1, self.b1], [self.a2, self.b2]] = dimensions

    def call(self, inputs):
        xvals, vvals = inputs
        x, y = tf.unstack(xvals, axis=-1)
        vx, vy = tf.unstack(vvals, axis=-1)
        
        # Apply cutoff based on boundary values
        VY = (x - self.a1) * (self.b1 - x) * vy
        VX = (y - self.a2) * (self.b2 - y) * vx
        
        return tf.stack([VX, VY], axis=-1)

# ========================================
# Cutoff Layer for 3D: Enforcing H_0(curl) Boundary Conditions
# ========================================
# This custom layer extends the 2D cutoff logic to a 3D domain, enforcing 
# H_0(curl) boundary conditions on the output of the neural network.

class cross_cutoff_3d(tf.keras.layers.Layer):
    def __init__(self, dimensions=3, **kwargs):
        super(cross_cutoff_3d, self).__init__()
        [self.a1, self.b1] = dimensions[0]
        [self.a2, self.b2] = dimensions[1]
        [self.a3, self.b3] = dimensions[2]

    def call(self, inputs):
        xvals, vvals = inputs
        x, y, z = tf.unstack(xvals, axis=-1)
        vx, vy, vz = tf.unstack(vvals, axis=-1)
        
        # Apply 3D boundary conditions
        VY = (z - self.a3) * (self.b3 - z) * (x - self.a1) * (self.b1 - x) * vy
        VX = (z - self.a3) * (self.b3 - z) * (y - self.a2) * (self.b2 - y) * vx
        VZ = (y - self.a2) * (self.b2 - y) * (x - self.a1) * (self.b1 - x) * vz
        
        return tf.stack([VX, VY, VZ], axis=-1)



# ========================================
# Create the Loss Function for Training
# ========================================
# This function creates a loss model that will be used for training the neural network.
# The loss function depends on the solution model (v_model), parameters such as N, epsilon, 
# and mu, and the right-hand side (rhs) of the equation. It applies different loss calculation 
# strategies depending on whether the dimension is 2D or 3D.

def make_loss_model(v_model, N, Nmodes, epsilon, mu, rhs, solution, dimensions=[[0, np.pi], [0, np.pi]], dtype='float32'):
    """
    Creates a loss model that calculates the loss based on the neural network solution 
    and the desired norm in the specified function space.
    
    Parameters:
        v_model: The neural network model that represents the candidate solution.
        N: Number of integration points/modes (e.g., nx, ny).
        epsilon: Coefficient for the PDE.
        mu: Coefficient for the PDE.
        rhs: The right-hand side of the equation.
        solution: The exact solution or a reference function for comparison.
        space: The function space corresponding to the norm being calculated (e.g., "Hcurl").
        dimensions: The domain dimensions (default is [[0, pi], [0, pi]]).
        dtype: Data type for layers (default is 'float32').
    
    Returns:
        A Keras model that calculates the loss.
    """
    
    # Input layer: a dummy input used to drive the loss calculation
    xvals = tf.keras.layers.Input(shape=(1,), name="dummy input", dtype=dtype)
    
    # Choose the appropriate loss calculation method based on the dimensionality
    if len(dimensions) == 2:
        # 2D loss calculation using cross_loss layer
        l1 = cross_loss(v_model, N, Nmodes, epsilon, mu, rhs, solution, dimensions=dimensions)(xvals)
        # val1 = cross_loss(v_model, Nval, epsilon, mu, rhs, solution, dimensions=dimensions)(xvals)
    else:
        # 3D loss calculation using cross_loss_3d layer
        l1 = cross_loss_3D(v_model, N, epsilon, mu, rhs, solution, dimensions=dimensions)(xvals)
        # val1 = cross_loss(v_model, Nval, epsilon, mu, rhs, solution, dimensions=dimensions)(xvals)
    
    #outputs = tf.concat([l1,val1],axis=-1)
    # Create a Keras model for the loss calculation
    loss_model = tf.keras.Model(inputs=xvals, outputs=l1, name="loss_model")
    
    # Print the model summary for debugging
    loss_model.summary()
    
    return loss_model




