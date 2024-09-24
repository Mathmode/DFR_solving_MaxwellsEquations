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
import matplotlib.pyplot as plt


def post_processing_solution(v_model, sol_objects, dtype='float32'):
    """
    Post-processes the solution obtained from the neural network model by comparing
    it with the exact solution and plotting the results.

    Parameters:
        v_model: Trained neural network model representing the approximate solution.
        sol_objects: A list containing the exact solution components and domain dimensions:
            - v1_exact: Exact solution component in the x-direction.
            - v2_exact: Exact solution component in the y-direction.
            - curlv_exact: Exact curl of the solution.
            - dimensions: Domain dimensions.
        dtype: Data type to be used for computations (default is 'float32').

    Returns:
        None. The function displays several plots comparing the approximate and exact solutions.
    """

    # Unpack the exact solution components and domain dimensions
    v1_exact, v2_exact, curlv_exact, dimensions = sol_objects

    # Set the number of test points in x and y directions for plotting
    nxtest = 20
    nytest = 20

    # Create test points in the x and y directions within the domain
    xtest = tf.constant([(i + 0.5) * np.pi / nxtest for i in range(nxtest)], dtype=dtype)
    ytest = tf.constant([(i + 0.5) * np.pi / nytest for i in range(nytest)], dtype=dtype)

    # Create a meshgrid for the test points
    X, Y = tf.meshgrid(xtest, ytest)

    # Evaluate the exact solution components at the test points
    v1_e = v1_exact(X, Y)
    v2_e = v2_exact(X, Y)

    # Flatten the meshgrid for input into the model
    xx = tf.reshape(X, [nxtest * nytest])
    yy = tf.reshape(Y, [nxtest * nytest])

    # Use TensorFlow's GradientTape to compute gradients (for curl calculation)
    with tf.GradientTape(persistent=True) as t1:
        # Watch the input variables for gradient computation
        t1.watch(xx)
        t1.watch(yy)

        # Stack the input coordinates and pass them through the model to get approximate solutions
        vap = v_model(tf.stack([xx, yy], axis=-1))
        v1_a, v2_a = tf.unstack(vap, axis=-1)

    # Compute the approximate curl using gradients of the approximate solution components
    curl_a = tf.reshape(t1.gradient(v1_a, yy) - t1.gradient(v2_a, xx), [nytest, nxtest])

    # Delete the GradientTape object to free resources
    del t1

    # Reshape the approximate solution components back to the meshgrid shape
    v1_a = tf.reshape(v1_a, [nytest, nxtest])
    v2_a = tf.reshape(v2_a, [nytest, nxtest])

    # ========================================
    # Plotting the Approximate Solution
    # ========================================

    # Compute the magnitude of the approximate vector field
    M = tf.sqrt(v1_a**2 + v2_a**2)

    # Create a filled contour plot of the magnitude with a quiver plot of the vector field
    plt.contourf(X, Y, M, cmap='coolwarm', alpha=0.5)
    plt.quiver(X, Y, v1_a, v2_a, cmap='coolwarm')

    # Add a colorbar with formatting
    cbar = plt.colorbar(format='%.1f')
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=14)

    # Set plot titles and labels
    plt.title(r'$\mathbf{E}$ (Approximate Solution)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # ========================================
    # Plotting the Exact Solution
    # ========================================

    # Compute the magnitude of the exact vector field
    M = np.sqrt(v1_e**2 + v2_e**2)

    # Create a filled contour plot of the magnitude with a quiver plot of the vector field
    plt.contourf(X, Y, M, cmap='coolwarm', alpha=0.5)
    plt.quiver(X, Y, v1_e, v2_e, cmap='coolwarm')

    # Add a colorbar with formatting
    cbar = plt.colorbar(format='%.1f')
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=14)

    # Set plot titles and labels
    plt.title(r'$\mathbf{E}^*$ (Exact Solution)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # ========================================
    # Plotting the Error Between Approximate and Exact Solutions
    # ========================================

    # Compute the pointwise error between the approximate and exact solutions
    err_sol = np.sqrt(tf.square(v1_e - v1_a) + tf.square(v2_e - v2_a))

    # Create a filled contour plot of the error
    plt.contourf(X, Y, err_sol, cmap='coolwarm', alpha=0.9, levels=20)

    # Add a colorbar with scientific notation formatting
    cbar = plt.colorbar(format='%.1e')
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=14)

    # Set plot titles and labels
    plt.title(r'$||\mathbf{E} - \mathbf{E}^*||_{\mathbb{R}^2}$', fontsize=22)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # ========================================
    # Plotting the Approximate Curl of the Solution
    # ========================================

    # Create a filled contour plot of the approximate curl
    plt.contourf(X, Y, curl_a, cmap='coolwarm')

    # Add a colorbar with formatting
    cbar = plt.colorbar(format='%.f')
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=14)

    # Set plot titles and labels
    plt.title(r'$\mathrm{curl}(\mathbf{E})$ (Approximate)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # ========================================
    # Plotting the Exact Curl of the Solution
    # ========================================

    # Evaluate the exact curl at the test points
    curl_exact = curlv_exact(X, Y)

    # Create a filled contour plot of the exact curl
    plt.contourf(X, Y, curl_exact, cmap='coolwarm')

    # Add a colorbar with formatting
    cbar = plt.colorbar(format='%.f')
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=14)

    # Set plot titles and labels
    plt.title(r'$\mathrm{curl}(\mathbf{E}^*)$ (Exact)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # ========================================
    # Plotting the Error in the Curl Between Approximate and Exact Solutions
    # ========================================

    # Compute the absolute error in the curl
    err_curl = tf.math.abs(curl_a - curl_exact)

    # Create a filled contour plot of the curl error
    plt.contourf(X, Y, err_curl, cmap='coolwarm', alpha=0.9, levels=20)

    # Add a colorbar with scientific notation formatting
    cbar = plt.colorbar(format='%.1e')
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=14)

    # Set plot titles and labels
    plt.title(r'$|\mathrm{curl}(\mathbf{E}) - \mathrm{curl}(\mathbf{E}^*)|$', fontsize=22)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def post_processing_loss(history, nx, ny, n_points, folder):
    """
    Plots the loss and error versus iterations from the training history.

    Parameters:
        history (dict): Training history containing 'loss' and 'sol_err'.
        nx (int): Number of points in the x-direction (for display purposes).
        ny (int): Number of points in the y-direction (for display purposes).
        n_points (int): Number of points to display in the plots.
        folder (str): Folder path where the plots are saved (not used here).
    """

    # -----------------------------------
    # PLOT LOSS VS ITERATIONS
    # -----------------------------------

    # Extract loss and error from the training history
    loss_train = np.array(history['loss'])
    error_train = np.array(history['sol_err']) * 100  # Convert error to percentage
    
    # Determine the number of data points to plot
    ndata = len(loss_train) - 1
    n_points = min(n_points, ndata)
    
    # Create the figure and axis for the loss plot
    fig, ax = plt.subplots()
    
    # Select indices for plotting using logarithmic spacing
    idx_data = np.logspace(0, np.log10(ndata), n_points, dtype=int)

    # Plot the loss data
    ax.scatter(idx_data, [loss_train[i] for i in idx_data],
               s=40, c='b', marker='+', linewidth=2)
    
    # Set axis properties
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$\mathcal{L}(\mathbf{E})$')
    
    # Add legend
    ax.legend(['Loss Training'])
    
    # Add grid lines
    ax.grid(which='major', axis='both', linestyle=':', color='gray')
    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # PLOT ERROR VS ITERATIONS
    # -----------------------------------

    # Create the figure and axis for the error plot
    fig, ax = plt.subplots()
    
    # Select indices for plotting using logarithmic spacing
    idx_data = np.logspace(np.log10(1), np.log10(ndata), n_points, dtype=int)

    # Plot the error data
    ax.scatter(idx_data, [error_train[i] for i in idx_data],
               s=40, c='r', marker='x', linewidth=2)

    # Set axis properties
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$H(\mathrm{curl})$ error (relative %)')
    
    # Add legend
    ax.legend(['Error Training'])
    
    # Add grid lines
    ax.grid(which='major', axis='both', linestyle=':', color='gray')
    plt.tight_layout()
    plt.show()

    
   