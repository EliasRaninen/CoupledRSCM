"""
@author: Elias Raninen www.github.com/EliasRaninen

This version: March 2022
"""
import numpy as np
import pandas as pd
import multiclass_rscm as pool
np.random.seed(42)

# Define simulation setup

nmc = 4000  # number of Monte Carlo trials

SETUP = 'C'  # choose 'A', 'B', or 'C'

if SETUP == 'A':
    # Setup A: AR(1) covariance matrices
    p = 200  # dimension
    n = [25, 50, 75, 100]  # number of samples
    K = 4  # number of classes
    df = [8, 8, 8, 8]  # degrees of freedom of multivariate t
    rho = [0.2, 0.3, 0.4, 0.5]
    C = [pool.ARcov(rho[0], p),  # covariance matrices
         pool.ARcov(rho[1], p),
         pool.ARcov(rho[2], p),
         pool.ARcov(rho[3], p)]
elif SETUP == 'B':
    # Setup B: compound symmetry
    p = 200  # dimension
    K = 4  # number of classes
    n = [25, 50, 75, 100]  # number of samples
    df = [8, 8, 8, 8]  # degrees of freedom of multivariate t
    rho = [0.2, 0.3, 0.4, 0.5]
    C = [pool.CScov(rho[0], p),  # covariance matrices
         pool.CScov(rho[1], p),
         pool.CScov(rho[2], p),
         pool.CScov(rho[3], p)]
elif SETUP == 'C':
    # Setup C: mixed case
    p = 200  # dimension
    n = [100, 100, 100, 100]  # number of samples
    K = 4  # number of classes
    df = [12, 8, 12, 8]  # degrees of freedom of multivariate t
    rho = [0.6, 0.6, 0.1, 0.1]
    C = [pool.ARcov(rho[0], p),
         pool.ARcov(rho[1], p),
         pool.CScov(rho[2], p),
         pool.CScov(rho[3], p)]
else:
    print("SETUP must be either 'A', 'B', or 'C'")


# Run simulation

print('Running simulation: SETUP ' + SETUP + ', averaging over '
      + str(nmc) + ' Monte Carlo trials.')

# class means
mu = [np.random.normal(1, 0, p),
      np.random.normal(1, 0, p),
      np.random.normal(1, 0, p),
      np.random.normal(1, 0, p)]

mse_mc = dict()
mse_mc['poly'] = np.zeros((nmc, K))
mse_mc['polyave'] = np.zeros((nmc, K))
mse_mc['polys'] = np.zeros((nmc, K))
mse_mc['polysave'] = np.zeros((nmc, K))
mse_mc['lin2'] = np.zeros((nmc, K))
for mc in range(nmc):
    if mc % 50 == 0:
        print(mc, '/', nmc)
    dataset = pool.generate_dataset(mu, C, df, n, p, K)
    params = pool.estimate_parameters(dataset)

    # LINPOOL2
    out = pool.linpool(params, 'LIN2')
    LIN2 = out['Sigmas']

    # POLY
    out = pool.rscmpool(params, averaging=False, compute_inverse=False)
    POLY = out['Sigmas']

    # POLY-Ave
    out = pool.rscmpool(params, averaging=True, compute_inverse=False)
    POLYAve = out['Sigmas']

    # POLYs
    out = pool.rscmpools(params, T='Sp', averaging=False, compute_inverse=False)
    POLYs = out['Sigmas']

    # POLYs-Ave
    out = pool.rscmpools(params, T='Sp', averaging=True, compute_inverse=False)
    POLYsAve = out['Sigmas']

    for k in range(K):
        mse_mc['poly'][mc, k] = pool.NSE(POLY[k], C[k])
        mse_mc['polyave'][mc, k] = pool.NSE(POLYAve[k], C[k])
        mse_mc['polys'][mc, k] = pool.NSE(POLYs[k], C[k])
        mse_mc['polysave'][mc, k] = pool.NSE(POLYsAve[k], C[k])
        mse_mc['lin2'][mc, k] = pool.NSE(LIN2[k], C[k])


# Print results

def meanmse(a):
    """ function for returning mean of normalized squared error (x10) and their sum """
    return np.append(a.mean(axis=0), a.sum(axis=1).mean())*10


def stdmse(a):
    """ function for returning std of normalized squared error (x10) and their sum """
    return np.append(a.std(axis=0), a.sum(axis=1).std())*10


mse = dict()
mse['lin2'] = meanmse(mse_mc['lin2'])
mse['poly'] = meanmse(mse_mc['poly'])
mse['polys'] = meanmse(mse_mc['polys'])
mse['polyave'] = meanmse(mse_mc['polyave'])
mse['polysave'] = meanmse(mse_mc['polysave'])

std = dict()
std['lin2'] = stdmse(mse_mc['lin2'])
std['poly'] = stdmse(mse_mc['poly'])
std['polys'] = stdmse(mse_mc['polys'])
std['polyave'] = stdmse(mse_mc['polyave'])
std['polysave'] = stdmse(mse_mc['polysave'])

# print results
print('MSE (x10):')
print(pd.DataFrame(mse, index=['Class 1', 'Class 2',
                               'Class 3', 'Class 4', 'Sum']).T.round(1))
print('')
print('Standard deviation (x10):')
print(pd.DataFrame(std, index=['Class 1', 'Class 2',
                               'Class 3', 'Class 4', 'Sum']).T.round(1))
