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

# ========================================
# Normalization Constants for Eigenvectors in X(Q)
# ========================================
# This function calculates the normalization constant for eigenvectors 
# in X(Q) for given k1, k2 values.
def evs_curl(k1, k2, d1=np.pi, d2=np.pi):
    """
    Computes the normalization constants for eigenvectors in X(Q).
    
    Parameters:
        k1, k2: Mode indices
        d1, d2: Domain sizes (default is pi)
    
    Returns:
        Normalization constant for the eigenvectors.
    """
    if k1 == 0 and k2 == 0:
        ans = 0.
    elif k1 == 0 or k2 == 0:
        k = k1 + k2
        ans = (np.pi**2 * k**2 * (np.pi**2 * k**2 + d1**2) * d2 / (2 * d1**3))**-1
    else:
        ans = (np.pi**2 * (((np.pi**2 * k2**2 + d2**2) * d1**2 + np.pi**2 * k1**2 * d2**2)
               * (d1**2 * k2**2 + d2**2 * k1**2) / (4 * d1**3 * d2**3)))**-1
    return ans

# ========================================
# Normalization Constants for Eigenvectors in DH^1_0(Q)
# ========================================
# This function calculates the normalization constants for eigenvectors in DH^1_0(Q)
def evs_DH1(k1, k2, d1=np.pi, d2=np.pi):
    """
    Computes the normalization constants for eigenvectors in DH^1_0(Q).
    
    Parameters:
        k1, k2: Mode indices
        d1, d2: Domain sizes (default is pi)
    
    Returns:
        Normalization constant for the eigenvectors.
    """
    if k1 == 0 and k2 == 0:
        ans = 0.
    else:
        ans = (np.pi**2 * (d1**2 * k2**2 + d2**2 * k1**2) / (4 * d1 * d2))**-1
    return ans

# ========================================
# Custom Layer for Evaluating H_0(curl)* Norm and H(curl) Error
# ========================================
# This class defines a custom Keras layer to evaluate the H_0(curl)* norm,
# and calculate the H(curl) error of a candidate solution (v_model). It uses 
# Fourier-based methods to solve the equation curl(mu^-1 curl(v)) + epsilon v = F 
# in the H_0(curl) space.

class cross_loss(tf.keras.layers.Layer):
    def __init__(self, v_model, N, Nmodes, epsilon, mu, rhs, solution,
                 dimensions=[[0, np.pi], [0, np.pi]], dtype='float32', **kwargs):
        """
        Initializes the custom loss layer with necessary parameters.
        
        Parameters:
            v_model: The candidate solution neural network.
            N: Number of integration points in x and y directions (nx, ny).
            Nmodes: Number of Fourier modes in x and y directions (nmx, nmy).
            epsilon, mu: Material parameters as functions.
            rhs: Right-hand side components (F1, F2).
            solution: Exact solution components (v1_exact, v2_exact, curlv_exact).
            dimensions: Domain dimensions (default is [[0, pi], [0, pi]]).
            dtype: Data type (default is 'float32').
        """
        super(cross_loss, self).__init__()

        # Set domain dimensions and compute domain sizes
        [[a1, b1], [a2, b2]] = dimensions
        d1 = b1 - a1
        d2 = b2 - a2
        
        # Integration points and Fourier modes
        [nx, ny] = N
        [nmx, nmy] = Nmodes
        [F1, F2] = rhs
        [v1_exact, v2_exact, curlv_exact] = solution

        # Sampling points for numerical integration in x and y directions
        x = tf.constant([(i + 0.5) / nx * d1 + a1 for i in range(nx)], dtype=dtype)
        y = tf.constant([(i + 0.5) / ny * d2 + a2 for i in range(ny)], dtype=dtype)

        # Create meshgrid for the sampling points
        self.X, self.Y = tf.meshgrid(x, y)

        # Evaluate forcing terms and material parameters on the mesh
        self.F1 = F1(self.X, self.Y)
        self.F2 = F2(self.X, self.Y)
        self.mu = mu(self.X, self.Y)
        self.epsilon = epsilon(self.X, self.Y)

        # Compute factors that appear in the gradient terms (kx, ky for Fourier modes)
        self.kx = tf.constant([k1 * np.pi / d1 for k1 in range(nmx)], dtype=dtype)
        self.ky = tf.constant([k1 * np.pi / d2 for k1 in range(nmy)], dtype=dtype)

        # Normalization constants for DH^1_0 and X0
        self.evs_DH1 = tf.constant([[evs_DH1(k1, k2, d1=d1, d2=d2) for k1 in range(nmx)] for k2 in range(nmy)], dtype=dtype)
        self.evs_X0 = tf.constant([[evs_curl(k1, k2, d1=d1, d2=d2) for k1 in range(nmx)] for k2 in range(nmy)], dtype=dtype)

        # k^2 for curl terms in Fourier modes
        self.ksq_curl = tf.constant([[((k1 * np.pi / d1)**2 + (k2 * np.pi / d2)**2) for k1 in range(nmx)] for k2 in range(nmy)], dtype=dtype)

        # Store the candidate solution neural network
        self.v_model = v_model

        # Reshape the meshgrid for integration
        self.xlist = tf.reshape(self.X, [nx * ny])
        self.ylist = tf.reshape(self.Y, [nx * ny])

        # Store the number of points and modes
        self.nx = nx
        self.ny = ny
        self.nmx = nmx
        self.nmy = nmy

        # Evaluate the exact solution on the mesh
        self.v1_exact = v1_exact(self.X, self.Y)
        self.v2_exact = v2_exact(self.X, self.Y)
        self.curlv_exact = curlv_exact(self.X, self.Y)

        # Compute the Hcurl norm of the exact solution
        self.norm = tf.reduce_mean(self.v1_exact**2 + self.v2_exact**2 + self.curlv_exact**2)

        # Create the DCT and DST matrices for Fourier Transforms
        self.dct2_x = d1 * tf.stack([tf.math.cos((x - a1) * k1 * np.pi / d1) for k1 in range(nmx)], axis=-1) / nx
        self.dct2_y = d2 * tf.stack([tf.math.cos((y - a2) * k1 * np.pi / d2) for k1 in range(nmy)], axis=-1) / ny
        self.dst2_x = d1 * tf.stack([tf.math.sin((x - a1) * k1 * np.pi / d1) for k1 in range(nmx)], axis=-1) / nx
        self.dst2_y = d2 * tf.stack([tf.math.sin((y - a2) * k1 * np.pi / d2) for k1 in range(nmy)], axis=-1) / ny

    def call(self, inputs):
        """
        Computes the dual norm of the residuals and the relative H_0(curl) error.
        
        Returns:
            A tensor with two values:
                - sqrt of the dual norm squared of the residual on DH^1_0 and X0.
                - sqrt of the relative H_0(curl) error.
        """

        # Compute the curl of the candidate solution using automatic differentiation
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.xlist)
            t1.watch(self.ylist)
            xylist = tf.stack([self.xlist, self.ylist], axis=-1)
            v = self.v_model(xylist, training=True)
            v1, v2 = tf.unstack(v, axis=-1)
        curlv = t1.gradient(v1, self.ylist) - t1.gradient(v2, self.xlist)
        del t1

        # Reshape v1, v2, and curlv to the meshgrid
        v1 = tf.reshape(v1, [self.ny, self.nx])
        v2 = tf.reshape(v2, [self.ny, self.nx])
        curlv = tf.reshape(curlv, [self.ny, self.nx])

        # Compute epsilon * v - F
        eps_u_mins_f_1 = self.epsilon * v1 - self.F1
        eps_u_mins_f_2 = self.epsilon * v2 - self.F2

        # Perform DCT and DST for x and y components of epsilon * v - F
        SyCx_epsuF_1 = tf.einsum("iI,jJ,ji->JI", self.dct2_x, self.dst2_y, eps_u_mins_f_1)
        SxCy_epsuF_2 = tf.einsum("iI,jJ,ji->JI", self.dst2_x, self.dct2_y, eps_u_mins_f_2)

        # Test the residuals against the basis of DH^1_0 (high-order term)
        h1_test = tf.einsum("i,ij->ij", self.ky, SxCy_epsuF_2) + tf.einsum("j,ij->ij", self.kx, SyCx_epsuF_1)

        # Test the residuals against the basis of X0 (low-order term)
        curl_test_low = (-tf.einsum("j,ij->ij", self.kx, SxCy_epsuF_2)
                         + tf.einsum("i,ij->ij", self.ky, SyCx_epsuF_1))

        # Perform DCT on curl(v) / mu
        CxCyCurl = tf.einsum("iI,jJ,ji->JI", self.dct2_x, self.dct2_y, curlv / self.mu) * self.ksq_curl

        # Compute the dual norm squared on DH^1_0
        h1_contribution = tf.reduce_sum(tf.square(h1_test) * self.evs_DH1)

        # Compute the dual norm squared on X0
        curl_contribution = tf.reduce_sum(tf.square(curl_test_low + CxCyCurl) * self.evs_X0)

        # Compute the relative H_0(curl) error
        curl_er = tf.reduce_mean(tf.square(v1 - self.v1_exact) + tf.square(v2 - self.v2_exact) +
                                 tf.square(curlv - self.curlv_exact)) / self.norm

        # Return the dual norm and relative error
        return tf.stack([tf.sqrt(h1_contribution + curl_contribution), curl_er**0.5])



# ========================================
# Normalization Constants for Eigenvectors in TE and TM Modes
# ========================================

def evs_te(k1, k2, k3, d1=np.pi, d2=np.pi, d3=np.pi):
    """
    Computes normalization constants for eigenvectors in the TE (Transverse Electric) mode.
    
    Parameters:
        k1, k2, k3: Mode indices
        d1, d2, d3: Domain sizes (default is pi)
    
    Returns:
        Normalization constant for the TE mode.
    """
    if k2 == 0 or k3 == 0:
        ans = 0
    else:
        ans = (np.pi**4 * (d1**2 * d2**2 * k3**2 + d1**2 * d3**2 * k2**2 + d2**2 * d3**2 * k1**2) *
               (d2**2 * k3**2 + d3**2 * k2**2) *
               (np.pi**2 * d1**2 * d2**2 * k3**2 + np.pi**2 * d1**2 * d3**2 * k2**2 +
                np.pi**2 * d2**2 * d3**2 * k1**2 + d1**2 * d2**2 * d3**2) /
               (8 * d1**3 * d2**5 * d3**5))**-1
    return ans


def evs_tm(k1, k2, k3, d1=np.pi, d2=np.pi, d3=np.pi):
    """
    Computes normalization constants for eigenvectors in the TM (Transverse Magnetic) mode.
    
    Parameters:
        k1, k2, k3: Mode indices
        d1, d2, d3: Domain sizes (default is pi)
    
    Returns:
        Normalization constant for the TM mode.
    """
    if k1 == 0 or (k2 == 0 and k3 == 0):
        ans = 0
    else:
        ans = (np.pi**2 * (d2**2 * k3**2 + d3**2 * k2**2) *
               (np.pi**2 * d1**2 * d2**2 * k3**2 + np.pi**2 * d1**2 * d3**2 * k2**2 +
                np.pi**2 * d2**2 * d3**2 * k1**2 + d1**2 * d2**2 * d3**2) /
               (8 * d1 * d2**3 * d3**3))**-1
    return ans

# ========================================
# Normalization Constants for Eigenvectors in DH^1_0(Q) in H_0(curl)
# ========================================
def evs_DH1_3D(k1, k2, k3, d1=np.pi, d2=np.pi, d3=np.pi):
    """
    Computes normalization constants for eigenvectors in DH^1_0(Q) in H_0(curl) space.
    
    Parameters:
        k1, k2, k3: Mode indices
        d1, d2, d3: Domain sizes (default is pi)
    
    Returns:
        Normalization constant for the eigenvectors in DH^1_0(Q).
    """
    if k1 == 0 or k2 == 0 or k3 == 0:
        ans = 0
    else:
        ans = (np.pi**2 * ((d1**2 * k2**2 + d2**2 * k1**2) * d3**2 + d1**2 * d2**2 * k3**2) /
               (8 * d1 * d2 * d3))**-1
    return ans

# ========================================
# Custom Layer for Evaluating the Loss in 3D (mu**-1 curl(u) curl(v) + epsilon u v - Fv = 0)
# ========================================

class cross_loss_3D(tf.keras.layers.Layer):
    """
    Custom TensorFlow Keras layer for evaluating the 3D loss function:
    mu**-1 curl(u) curl(v) + epsilon u v - Fv = 0.
    
    This layer calculates both the dual norm of the residual and the relative H_0(curl) error.
    """
    def __init__(self, v_model, N, epsilon, mu, rhs, solution, dimensions=[[0, np.pi]]*3, **kwargs):
        """
        Initializes the layer with required parameters.
        
        Parameters:
            v_model: The neural network model representing the candidate solution.
            N: Number of integration points in x, y, z directions.
            epsilon, mu: Material parameter functions.
            rhs: The right-hand side of the equation (F1, F2, F3).
            solution: Exact solution (v1_exact, v2_exact, v3_exact, curlv_exact).
            dimensions: The domain dimensions.
        """
        super(cross_loss_3D, self).__init__()
        
        # Unpacking the dimensions and integration points
        [nx, ny, nz] = N
        [F1, F2, F3] = rhs
        [v1_exact, v2_exact, v3_exact, curlv_exact] = solution
        
        [[a1, b1], [a2, b2], [a3, b3]] = dimensions
        d1 = b1 - a1
        d2 = b2 - a2
        d3 = b3 - a3
        
        # Creating the sampling points for x, y, and z directions
        x = tf.constant([(i + 0.5) / nx * d1 + a1 for i in range(nx)], dtype='float32')
        y = tf.constant([(i + 0.5) / ny * d2 + a2 for i in range(ny)], dtype='float32')
        z = tf.constant([(i + 0.5) / nz * d3 + a3 for i in range(nz)], dtype='float32')
        self.X, self.Y, self.Z = tf.meshgrid(x, y, z)
        
        # Transposing the meshgrid
        self.X = tf.transpose(self.X, [2, 0, 1])
        self.Y = tf.transpose(self.Y, [2, 0, 1])
        self.Z = tf.transpose(self.Z, [2, 0, 1])
        
        # Evaluating forcing terms and material parameters on the mesh
        self.F1 = F1(self.X, self.Y, self.Z)
        self.F2 = F2(self.X, self.Y, self.Z)
        self.F3 = F3(self.X, self.Y, self.Z)
        self.mu = mu(self.X, self.Y, self.Z)
        self.epsilon = epsilon(self.X, self.Y, self.Z)
        
        # Calculating the Fourier mode coefficients
        self.kx = tf.constant([k1 * np.pi / d1 for k1 in range(nx)], dtype='float32')
        self.ky = tf.constant([k1 * np.pi / d2 for k1 in range(ny)], dtype='float32')
        self.kz = tf.constant([k1 * np.pi / d3 for k1 in range(nz)], dtype='float32')
        
        # Normalization constants for DH^1_0 and curl terms
        self.evs_DH1 = tf.constant([[[evs_DH1_3D(k1, k2, k3, d1=d1, d2=d2, d3=d3) 
                                     for k1 in range(nx)] for k2 in range(ny)] for k3 in range(nz)], dtype='float32')
        
        self.ksq_curl = tf.constant([[[((k1 * np.pi / d1)**2 + (k2 * np.pi / d2)**2 + (k3 * np.pi / d3)**2) 
                                       for k1 in range(nx)] for k2 in range(ny)] for k3 in range(nz)], dtype='float32')
        
        # Normalization constants for TE and TM modes
        self.evs_te = tf.constant([[[evs_te(k1, k2, k3, d1=d1, d2=d2, d3=d3) 
                                    for k1 in range(nx)] for k2 in range(ny)] for k3 in range(nz)], dtype='float32')
        
        self.evs_tm = tf.constant([[[evs_tm(k1, k2, k3, d1=d1, d2=d2, d3=d3) 
                                    for k1 in range(nx)] for k2 in range(ny)] for k3 in range(nz)], dtype='float32')
        
        # Storing the neural network model and reshaping inputs for calculation
        self.v_model = v_model
        self.xlist = tf.reshape(self.X, [nx * ny * nz])
        self.ylist = tf.reshape(self.Y, [nx * ny * nz])
        self.zlist = tf.reshape(self.Z, [nx * ny * nz])
        
        # Exact solution components
        self.v1_exact = v1_exact(self.X, self.Y, self.Z)
        self.v2_exact = v2_exact(self.X, self.Y, self.Z)
        self.v3_exact = v3_exact(self.X, self.Y, self.Z)
        self.curlv_exact_1, self.curlv_exact_2, self.curlv_exact_3 = curlv_exact(self.X, self.Y, self.Z)
        
        # Norm of the exact solution
        norm_low = tf.reduce_mean(self.v1_exact**2 + self.v2_exact**2 + self.v3_exact**2)
        norm_high = tf.reduce_mean(tf.square(self.curlv_exact_1) + tf.square(self.curlv_exact_2) + tf.square(self.curlv_exact_3))
        self.norm = norm_low + norm_high
        
        # DCT and DST matrices for Fourier transforms
        self.dct2_x = tf.stack([tf.math.cos((x - a1) * k1 * np.pi / d1) for k1 in range(nx)], axis=-1) / nx
        self.dct2_y = tf.stack([tf.math.cos((y - a2) * k1 * np.pi / d2) for k1 in range(ny)], axis=-1) / ny
        self.dct2_z = tf.stack([tf.math.cos((z - a3) * k1 * np.pi / d3) for k1 in range(nz)], axis=-1) / nz
        self.dst2_x = tf.stack([tf.math.sin((x - a1) * k1 * np.pi / d1) for k1 in range(nx)], axis=-1) / nx
        self.dst2_y = tf.stack([tf.math.sin((y - a2) * k1 * np.pi / d2) for k1 in range(ny)], axis=-1) / ny
        self.dst2_z = tf.stack([tf.math.sin((z - a3) * k1 * np.pi / d3) for k1 in range(nz)], axis=-1) / nz
    
    def call(self, inputs):
        """
        Calculates the dual norm and relative H_0(curl) error.
        
        Returns:
            A tensor with four values:
            - Dual norm contribution from DH^1_0.
            - Dual norm contribution from curl terms.
            - Total dual norm.
            - Relative H_0(curl) error.
        """
        
        # Compute the curl of the candidate solution
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.xlist)
            t1.watch(self.ylist)
            t1.watch(self.zlist)
            xyzlist = tf.stack([self.xlist, self.ylist, self.zlist], axis=-1)
            v = self.v_model(xyzlist, training=True)
            v1, v2, v3 = tf.unstack(v, axis=-1)
        
        # Compute the curl components
        v12 = t1.gradient(v1, self.ylist)
        v21 = t1.gradient(v2, self.xlist)
        v13 = t1.gradient(v1, self.zlist)
        v31 = t1.gradient(v3, self.xlist)
        v23 = t1.gradient(v2, self.zlist)
        v32 = t1.gradient(v3, self.ylist)
        
        curlv1 = v32 - v23
        curlv2 = v13 - v31
        curlv3 = v21 - v12
        
        del t1
        
        # Reshape the values
        v1 = tf.reshape(v1, [self.nz, self.ny, self.nx])
        v2 = tf.reshape(v2, [self.nz, self.ny, self.nx])
        v3 = tf.reshape(v3, [self.nz, self.ny, self.nx])
        curlv1 = tf.reshape(curlv1, [self.nz, self.ny, self.nx])
        curlv2 = tf.reshape(curlv2, [self.nz, self.ny, self.nx])
        curlv3 = tf.reshape(curlv3, [self.nz, self.ny, self.nx])
        
        # Compute epsilon * v - F
        eps_u_mins_f_1 = self.epsilon * v1 - self.F1
        eps_u_mins_f_2 = self.epsilon * v2 - self.F2
        eps_u_mins_f_3 = self.epsilon * v3 - self.F3
        
        # Fourier transforms (DCT and DST) for H1 modes
        CxSySz_l1 = tf.einsum("iI,jJ,kK,kji->KJI", self.dct2_x, self.dst2_y, self.dst2_z, eps_u_mins_f_1)
        SxCySz_l2 = tf.einsum("iI,jJ,kK,kji->KJI", self.dst2_x, self.dct2_y, self.dst2_z, eps_u_mins_f_2)
        SxSyCz_l3 = tf.einsum("iI,jJ,kK,kji->KJI", self.dst2_x, self.dst2_y, self.dct2_z, eps_u_mins_f_3)
        
        # Contribution from DH^1_0
        h1_1 = tf.einsum("I,KJI->KJI", self.kx, CxSySz_l1)
        h1_2 = tf.einsum("J,KJI->KJI", self.ky, SxCySz_l2)
        h1_3 = tf.einsum("K,KJI->KJI", self.kz, SxSyCz_l3)
        h1_contribution = tf.reduce_sum(self.evs_DH1 * (h1_1 + h1_2 + h1_3)**2)
        
        # TM modes contribution
        TM_L_2 = -tf.einsum("K,KJI->KJI", self.kz, SxCySz_l2)
        TM_L_3 = tf.einsum("J,KJI->KJI", self.ky, SxSyCz_l3)
        SxCyCz_h1 = tf.einsum("iI,jJ,kK,kji->KJI", self.dst2_x, self.dct2_y, self.dct2_z, curlv1 / self.mu)
        CxSyCz_h1 = tf.einsum("iI,jJ,kK,kji->KJI", self.dct2_x, self.dst2_y, self.dct2_z, curlv2 / self.mu)
        CxCySz_h1 = tf.einsum("iI,jJ,kK,kji->KJI", self.dct2_x, self.dct2_y, self.dst2_z, curlv3 / self.mu)
        TM_H_1 = tf.einsum("J,KJI->KJI", self.ky**2, SxCyCz_h1) + tf.einsum("K,KJI->KJI", self.kz**2, SxCyCz_h1)
        TM_H_2 = -tf.einsum("I,J,KJI->KJI", self.kx, self.ky, CxSyCz_h1)
        TM_H_3 = -tf.einsum("I,K,KJI->KJI", self.kx, self.kz, CxCySz_h1)
        tm_tot = TM_H_1 + TM_H_2 + TM_H_3 + TM_L_2 + TM_L_3
        tm_contribution = tf.reduce_sum(tf.square(tm_tot) * self.evs_tm)
        
        # TE modes contribution
        TE_L_1 = tf.einsum("J,KJI->KJI", self.ky**2, CxSySz_l1) + tf.einsum("K,KJI->KJI", self.kz**2, CxSySz_l1)
        TE_L_2 = -tf.einsum("I,J,KJI->KJI", self.kx, self.ky, SxCySz_l2)
        TE_L_3 = -tf.einsum("I,K,KJI->KJI", self.kx, self.kz, SxSyCz_l3)
        CxSyCz_h_2 = tf.einsum("iI,jJ,kK,kji->KJI", self.dct2_x, self.dst2_y, self.dct2_z, curlv2 / self.mu)
        CxCySz_h_3 = tf.einsum("iI,jJ,kK,kji->KJI", self.dct2_x, self.dct2_y, self.dst2_z, curlv3 / self.mu)
        TE_H = self.ksq_curl * (tf.einsum("K,KJI->KJI", self.kz, CxSyCz_h_2) - tf.einsum("J,KJI->KJI", self.ky, CxCySz_h_3))
        TE_tot = TE_H + TE_L_1 + TE_L_2 + TE_L_3
        TE_contribution = tf.reduce_sum(tf.square(TE_tot) * self.evs_te)
        
        # Total curl contribution
        curl_contribution = TE_contribution + tm_contribution
        
        # Calculate the low-order and high-order errors
        low_er = tf.reduce_mean(tf.square(v1 - self.v1_exact) + tf.square(v2 - self.v2_exact) + tf.square(v3 - self.v3_exact))
        high_er = tf.reduce_mean(tf.square(curlv1 - self.curlv_exact_1) + tf.square(curlv2 - self.curlv_exact_2) + tf.square(curlv3 - self.curlv_exact_3))
        curl_er = tf.sqrt((low_er + high_er) / self.norm)
        
        # Return the dual norm contributions and error
        return tf.stack([tf.sqrt(h1_contribution + curl_contribution), curl_er])
