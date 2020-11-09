# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:41:13 2020

@author: Elias Raninen www.github.com/EliasRaninen
"""
import numpy as np
import multirscmpooling as pool
np.random.seed(0)

# testing setup
# p = 3
# n = [10,9,8,7]
# rho = [0.3, 0.4, 0.5, 0.6]
# C = [pool.ARcov(rho[0],p),
#      pool.ARcov(rho[1],p),
#      pool.ARcov(rho[2],p),
#      pool.ARcov(rho[3],p)]

"""Define simulation setup"""
nmc = 4000 # number of Monte Carlo trials


# Setup A: AR(1) covariance matrices
# p = 200 # dimension
# n = [25, 50, 75, 100] # number of samples
# K = 4 # number of classes
# df = [8, 8, 8, 8] # degrees of freedom of multivariate t
# rho = [0.2, 0.3, 0.4, 0.5]
# C = [pool.ARcov(rho[0],p), # covariance matrices
#       pool.ARcov(rho[1],p),
#       pool.ARcov(rho[2],p),
#       pool.ARcov(rho[3],p)]

# Setup B: compound symmetry
# p = 200 # dimension
# K = 4 # number of classes
# n = [25, 50, 75, 100] # number of samples
# df = [8, 8, 8, 8] # degrees of freedom of multivariate t
# rho = [0.2, 0.3, 0.4, 0.5]
# C = [pool.CScov(rho[0],p), # covariance matrices
#       pool.CScov(rho[1],p),
#       pool.CScov(rho[2],p),
#       pool.CScov(rho[3],p)]

# Setup C: mixed case
p = 200 # dimension
n = [100,100,100,100] # number of samples
K = 4 # number of classes
df = [12, 8, 12, 8] # degrees of freedom of multivariate t
rho = [0.6, 0.6, 0.1, 0.1]
C = [pool.ARcov(rho[0],p),
      pool.ARcov(rho[1],p),
      pool.CScov(rho[2],p),
      pool.CScov(rho[3],p)]


"""Run simulation"""

# class means
mu = [np.random.normal(1,0,p),
      np.random.normal(1,0,p),
      np.random.normal(1,0,p),
      np.random.normal(1,0,p)]

mse_mc = dict()
mse_mc['poly'] = np.zeros((nmc,K))
mse_mc['polyave'] = np.zeros((nmc,K))
mse_mc['polys'] = np.zeros((nmc,K))
mse_mc['polysave'] = np.zeros((nmc,K))
mse_mc['lin2'] = np.zeros((nmc,K))
for mc in range(nmc):
    if mc % 50 == 0:
        print(mc,'/',nmc)
    dataset = pool.generate_dataset(mu,C,df,n,p,K)
    params = pool.estimate_parameters(dataset)
    
    # LINPOOL2
    out = pool.linpool(params,'LIN2')
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
        mse_mc['poly'][mc,k] = pool.NSE(POLY[k],C[k])
        mse_mc['polyave'][mc,k] = pool.NSE(POLYAve[k],C[k])
        mse_mc['polys'][mc,k] = pool.NSE(POLYs[k],C[k])
        mse_mc['polysave'][mc,k] = pool.NSE(POLYsAve[k],C[k])
        mse_mc['lin2'][mc,k] = pool.NSE(LIN2[k],C[k])

"""Print results"""

# function for returning mean of normalized squared error and their sum
def meanmse(a):
    return (np.append(a.mean(axis=0), a.sum(axis=1).mean())*10)
# function for returning std of normalized squared error and their sum
def stdmse(a):
    return (np.append(a.std(axis=0), a.sum(axis=1).std())*10)
def printresult(name,mse,std):
    print(("{:<12} {m[0]:<{prec}} ({d[0]:<{prec}}) {m[1]:<{prec}} ({d[1]:<{prec}}) {m[2]:<{prec}} ({d[2]:<{prec}}) {m[3]:<{prec}} ({d[3]:<{prec}}) {m[4]:<{prec}} ({d[4]:<{prec}})").format(name,m=mse.tolist(),d=std.tolist(), prec='6.3'))
    
mse = dict()
mse['poly']     = meanmse(mse_mc['poly'])
mse['polyave']  = meanmse(mse_mc['polyave'])
mse['polys']    = meanmse(mse_mc['polys'])
mse['polysave'] = meanmse(mse_mc['polysave'])
mse['lin2']     = meanmse(mse_mc['lin2'])

std = dict()
std['poly']     = stdmse(mse_mc['poly'])
std['polyave']  = stdmse(mse_mc['polyave'])
std['polys']    = stdmse(mse_mc['polys'])
std['polysave'] = stdmse(mse_mc['polysave'])
std['lin2']     = stdmse(mse_mc['lin2'])

# print results
print(("{:<12} "+"{:<16}"*5).format('Method','class1','class2','class3','class4','sum'))
printresult('Lin2:',mse['lin2'],std['lin2'])
printresult('Poly:',mse['poly'],std['poly'])
printresult('Polys:',mse['polys'],std['polys'])
printresult('PolyAve:',mse['polyave'],std['polyave'])
printresult('PolysAve:',mse['polysave'],std['polysave'])