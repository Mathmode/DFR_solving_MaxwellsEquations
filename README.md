# Deep Fourier Residual method for solving time-harmonic Maxwell's equations

This repository contains the source code implementation associated with the paper titled **"Deep Fourier Residual method for solving time-harmonic Maxwell's equations."** The method detailed in the paper introduces THE deep Fourier residual networks to solve the time-harmonic equations in 2D and 3D

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Examples](#examples)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Abstract

Solving PDEs with machine learning techniques has become a popular alternative to conventional methods. In this context, Neural networks (NNs) are among the most commonly used machine learning tools, and in those models, the choice of an appropriate loss function is critical. In general, the main goal is to guarantee that minimizing the loss during training translates to minimizing the error in the solution at the same rate. In this work, we focus on the time-harmonic Maxwell's equations, whose weak formulation takes H(curl) as the space of test functions. We propose a NN in which the loss function is a computable approximation of the dual norm of the weak-form PDE residual. To that end, we employ the Helmholtz decomposition of the space H(curl) and construct an orthonormal basis for this space in two and three spatial dimensions. Here, we use the Discrete Sine/Cosine Transform to accurately and efficiently compute the discrete version of our proposed loss function. Moreover, in the numerical examples we show a high correlation between the proposed loss function and the H(curl)-norm of the error, even in problems with low-regularity solutions.

## References 

**Deep Fourier Residual method for solving time-harmonic Maxwell's equations**
https://arxiv.org/abs/2305.09578

**A Deep Fourier Residual Method for solving PDEs using Neural Networks**
https://arxiv.org/abs/2210.14129

**Basic DFR code on keras-core** 
https://github.com/Mathmode/PINNSandDFR_kerascore


## Requirements

- Python 3.10.10
- TensorFlow 2.10
- NumPy

## Examples

We let $\Omega =[0,\pi]^2$ and consider the variational form: find $E \in H(curl,\Omega)$ satisfying

$$ \int_\Omega  curl(E) \cdot curl(\phi)+ E \cdot \phi \, dx = \int_\Omega \tilde{J} \cdot \phi \, dx \qquad \forall \phi \in H(curl,\Omega) $$

with, $\tilde{J}$ is chosen such that the exact solution is $E^{\mathrm{exact}}(x,y) = (xy(y-\pi),xy(x-\pi))^t$. 

## Authors 

Prof. Dr. Jamie M. Taylor. CUNEF Universidad, Madrid, Spain. (jamie.taylor@cunef.edu) 

Prof. Dr. Manuela Bastidas. University of the Basque Country (UPV/EHU), Leioa, Spain. / Universidad Nacional de Colombia, Medellín, Colombia. (manumnlb@gmail.com)

## Acknowledgments

Jamie Taylor have received funding from the Spanish Ministry of Science and Innovation projects with references TED2021-132783B-I00, PID2019-108111RB-I00 (FEDER/AEI), and PDC2021-121093-I00 (MCIN / AEI / 10.13039 / 501100011033 / Next Generation EU), the ``BCAM Severo Ochoa'' accreditation of excellence CEX2021-001142-S / MICIN / AEI / 10.13039 / 501100011033; the Spanish Ministry of Economic and Digital Transformation with Misiones Project IA4TES (MIA.2021.M04.008 / NextGenerationEU PRTR); and the Basque Government through the BERC 2022-2025 program, the Elkartek project BEREZ-IA (KK-2023 / 00012), and the Consolidated Research Group MATHMODE (IT1456-22) given by the Department of Education. 
