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

a1 = 0
b1 = np.pi
a2 = 0
b2 = np.pi

# Dimensions are [0,pi]^2 everywhere and this modifies that
dimensions = [[a1, b1], [a2, b2]]

def epsilon(X, Y):
 return tf.ones_like(X,dtype='float32')

def F1(x, y):
 return (y - b2)*(y - a2)*x - a1 - b1

def F2(x, y):
 return (x - b1)*(x - a1)*y - a2 - b2

def mu(X, Y):
 return tf.ones_like(X,dtype='float32')

def v1_exact(x, y):
 return x*(a2 - y)*(b2 - y)

def v2_exact(x, y):
 return y*(a1 - x)*(b1 - x)

def gradbit_exact(x, y):
 return -((b2 + a2)*x - y*(a1 + b1))

# v_exact is by default evaluating in Forward mode
def v_exact(inputs, training=True):
    # Split the input tensor into two tensors along the last dimension
    X, Y = tf.unstack(inputs, axis=-1)
    
    # Compute v1 and v2 by applying two other functions (v1_exact and v2_exact)
    # to the X and Y tensors
    v1 = v1_exact(X, Y)
    v2 = v2_exact(X, Y)
    
    # Combine the v1 and v2 tensors into a single tensor along the 
    # last dimension and return the result
    return tf.stack([v1, v2], axis=-1)
