# Coupled regularized sample covariance matrix estimator for multiple classes
This repository contains MATLAB, Python, and R code for the paper E. Raninen and E. Ollila, "Coupled Regularized Sample Covariance Matrix Estimator for Multiple Classes," in IEEE Transactions on Signal Processing, vol. 69, pp. 5681-5692, 2021, doi: [10.1109/TSP.2021.3118546](https://dx.doi.org/10.1109/TSP.2021.3118546).

## Abstract

The estimation of covariance matrices of multiple classes with limited training data is a difficult problem. The sample covariance matrix (SCM) is known to perform poorly when the number of variables is large compared to the available number of samples. In order to reduce the mean squared error (MSE) of the SCM, regularized (shrinkage) SCM estimators are often used. In this work, we consider regularized SCM (RSCM) estimators for multiclass problems that couple together two different target matrices for regularization: the pooled (average) SCM of the classes and the scaled identity matrix. Regularization toward the pooled SCM is beneficial when the population covariances are similar, whereas regularization toward the identity matrix guarantees that the estimators are positive definite. We derive the MSE optimal tuning parameters for the estimators as well as propose a method for their estimation under the assumption that the class populations follow (unspecified) elliptical distributions with finite fourth-order moments. The MSE performance of the proposed coupled RSCMs are evaluated with simulations and in a regularized discriminant analysis (RDA) classification set-up on real data. The results based on three different real data sets indicate comparable performance to cross-validation but with a significant speed-up in computation time. 

## Authors
- Elias Raninen, Doctoral Candidate, Department of Signal Processing and Acoustics, Aalto University.
- Esa Ollila, Professor, Department of Signal Processing and Acoustics, Aalto University.
