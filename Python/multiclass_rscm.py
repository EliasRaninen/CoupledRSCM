"""
@author: Elias Raninen, www.github.com/EliasRaninen

This version: March 2022
"""

import pandas as pd
import numpy as np
from scipy.stats import stats
from numpy import linalg


def CScov(rho, p):
    """Generate a compound symmetry (CS) covariance matrix.

    Returns a covariance matrix with compound symmetry structure

    Parameters
    ----------
    rho : int
          Correlation of variables (off-diagonal elements of covariance
          matrix).

    p : int
        Dimension.

    Returns
    -------
    C : ndarray of shape (p, p)
    """
    return np.ones(p)*rho + (1-rho)*np.eye(p)


def ARcov(rho, p):
    """Generate a first-order autoregressive (AR1) covariance matrix.

    Returns a covariance matrix with autoregressive (AR1) structure.

    Parameters
    ----------
    rho : int
          Parameter of the model.

    p : int
        Dimension.

    Returns
    -------
    C : ndarray of shape (p, p)
    """
    M = np.tile(np.arange(p), (p, 1))
    return rho ** abs(M - M.transpose())


def generate_multivariate_t_samples(mu, Sigma, df, n, p):
    """Generate multivariate t samples with mean mu and covariance matrix Sigma.
    
    Parameters
    ----------
    
    mu : array-like of shape (p,)
         Mean of the class.

    Sigma : array-like of shape (p, p)
            Covariance matrix.

    df : array-like of shape (K,)
         Degrees of freedom.

    n : int
        Number of samples.

    p : int
        Dimension.

    Return
    ------
    X : ndarray of shape (n, p)
        Array of samples
    """
    assert df > 2

    # N(0,Sigma*(df-2)/df)
    Z = np.random.multivariate_normal(np.zeros(p), Sigma*(df-2)/df, n)
    tau = np.random.chisquare(df, (n, 1))

    # multivariate t samples with covariance matrix Sigma and mean mu
    X = Z*np.sqrt(df/tau) + mu
    return X


def scov(X):
    """Compute the sample covariance matrix (SCM).

    Parameters:
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of samples.

    Returns:
    --------
    C : array-like of shape (n_features, n_features)
        Sample covariance matrix.
    """
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    return np.dot(Xc.T, Xc)/(n-1)
    # return np.cov(X.transpose())


def sscov(X):
    """Compute the spatial sign covariance matrix (SSCM).

    Parameters:
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of samples.

    Returns:
    --------
    C : array-like of shape (n_features, n_features)
        Spatial sign covariance matrix.
    """
    X = np.array(X)
    n = X.shape[0]
    m = spatialmedian(X)
    Xc = X - m
    xnorm = np.sqrt((Xc**2).sum(axis=1, keepdims=True))
    U = Xc / xnorm
    return np.dot(U.T, U)/n


def spatialmedian(X):
    """Compute the spatial median (geometric median)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of samples.

    Returns
    -------
    mu : array-like of shape (1, n_features)
         Spatial median.
    """
    X = np.array(X)
    m = X.mean(axis=0)
    iterMAX = 2000
    for iteration in range(iterMAX):
        m0 = m

        d = X - m
        dn = (np.sqrt((d**2).sum(axis=1, keepdims=True)))
        num = (X/dn).sum(axis=0, keepdims=True)
        den = (1/dn).sum(axis=0, keepdims=True)
        m = num/den

        crit = linalg.norm(m0-m)/linalg.norm(m0)
        if crit < 1e-6:
            return m
    if iteration == iterMAX:
        print("spatial median: slow convergence. crit:", crit)
    return m


def NSE(M, Sigma):
    """Computes the normalized squared error between M and Sigma.

    Parameters
    ----------
    M : array-like of shape (n_features, n_features)
        Estimate.

    Sigma : array-like of shape (n_features, n_features)
            True matrix.

    Returns
    -------
    out : numpy.float64
          Normalized squared error.
    """
    return linalg.norm(M-Sigma, 'fro')**2 / linalg.norm(Sigma, 'fro')**2


def estimate_gamma(X):
    """Estimate the sphericity parameter.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of samples.

    Returns
    -------
    gam : numpy.float64
          Estimate of the sphericity.
    """
    n, p = X.shape
    SSCM = sscov(X)
    gam = n/(n-1)*(p*np.trace(linalg.matrix_power(SSCM, 2))-p/n)
    return np.clip(gam, 1, p)


def estimate_eta(SCM):
    """Estimate the scale parameter

    Parameters
    ----------
    SCM : array-like of shape (n_features, n_features)
          The sample covariance matrix.

    Returns
    -------
    eta : numpy.float64
          The estimate of the scale.
    """
    return np.trace(SCM)/SCM.shape[0]


def estimate_kappa(X):
    """Estimate the elliptical kurtosis

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Array of samples.

    Returns
    -------
    kappa : numpy.float64
            estimate of the elliptical kurtosis
    """
    p = X.shape[1]
    kurt = np.maximum(stats.kurtosis(X).mean()/3, -2/(p+2))
    return kurt


def estimate_parameters(dataset):
    """Create dictionary of parameter estimates.

    Parameters
    ----------
    dataset : pandas.DataFrame
              Last column is dataset['class'], which indicates the class of the
              sample.

    Returns
    -------
    params : dict
             A dictionary of different population parameters.
    """
    classes = pd.unique(dataset['class'])
    K = len(classes)  # number of classes
    n = np.zeros(K)  # number of samples
    p = dataset.shape[1]-1
    params = dict()
    params['SCM'] = list()
    params['SSCM'] = list()
    params['gamma'] = np.zeros(K)
    params['eta'] = np.zeros(K)
    params['kappa'] = np.zeros(K)
    Sp = np.zeros((p, p))
    for k in range(K):
        Xk = (dataset.loc[dataset['class'] == k]).drop('class', axis=1)
        n[k] = Xk.shape[0]
        SCM = scov(Xk)
        Sp += n[k]*SCM
        params['SCM'].append(SCM)
        params['SSCM'].append(sscov(Xk))
        params['eta'][k] = estimate_eta(SCM)
        params['gamma'][k] = estimate_gamma(Xk)
        params['kappa'][k] = estimate_kappa(Xk)

        # store centered data for computing inverse if total number of samples
        # less than p
        if dataset.shape[0] < p:
            Xkc = Xk - Xk.mean(axis=0)
            if k == 0:
                X = np.copy(Xkc)
            else:
                X = np.vstack((X, Xkc))

    if dataset.shape[0] < p:
        params['centered_data'] = X

    params['Sp'] = Sp/n.sum()
    params['n'] = n.astype(int)
    params['p'] = p
    params['K'] = K
    params['PI'] = n/n.sum()
    params['tr_Ck2'] = p*params['eta']**2*params['gamma']
    params['trCk_2'] = p**2*params['eta']**2
    tau1 = 1/(n-1)+params['kappa']/n
    tau2 = params['kappa']/n
    params['Etr_Sk2'] = tau1*params['trCk_2']+(1+tau1+tau2)*params['tr_Ck2']
    params['EtrSk_2'] = 2*tau1*params['tr_Ck2']+(1+tau2)*params['trCk_2']
    params['MSE_Sk'] = (tau1*(params['trCk_2'] + params['tr_Ck2'])
                        + tau2*params['tr_Ck2'])

    params['trCiCj'] = np.zeros((K, K))
    params['trCitrCj'] = np.zeros((K, K))

    for k in range(K):
        jstart = k+1
        if jstart <= K:
            for j in range(jstart, K):
                params['trCitrCj'][k, j] = (
                        p**2*params['eta'][k]*params['eta'][j]
                        )
                params['trCitrCj'][j, k] = params['trCitrCj'][k, j]
                params['trCiCj'][k, j] = (np.trace(
                        np.dot(params['SSCM'][k], params['SSCM'][j])
                        ) * params['trCitrCj'][k, j])
                params['trCiCj'][j, k] = params['trCiCj'][k, j]

    params['trCiCj'][range(K), range(K)] = params['tr_Ck2'].reshape(K)
    params['trCitrCj'][range(K), range(K)] = params['trCk_2'].reshape(K)

    params['EtrSiSj'] = np.copy(params['trCiCj'])
    params['EtrSiSj'][range(K), range(K)] = params['Etr_Sk2'].reshape(K)
    params['EtrSitrSj'] = np.copy(params['trCitrCj'])
    params['EtrSitrSj'][range(K), range(K)] = params['EtrSk_2'].reshape(K)

    PIiPIj = np.outer(params['PI'], params['PI'])
    PImat = np.tile(params['PI'].transpose(), (K, 1))
    params['Etr_S2'] = (PIiPIj*params['EtrSiSj']).sum()
    params['EtrS_2'] = (PIiPIj*params['EtrSitrSj']).sum()
    params['EtrSkS'] = (PImat*params['EtrSiSj']).sum(axis=1)
    params['EtrCkS'] = (PImat*params['trCiCj']).sum(axis=1)
    params['EtrSktrS'] = (PImat*params['EtrSitrSj']).sum(axis=1)
    params['EtrCktrS'] = (PImat*params['trCitrCj']).sum(axis=1)
    return params


def rscmpool(params, averaging=False, compute_inverse=False):
    """Compute regularization parameters for the shrinkage covariance matrix
    estimator proposed E. Raninen and E. Ollila, “Coupled regularized sample
    covariance matrix estimator for multiple classes,” IEEE Transactions on
    Signal Processing, vol.  69, pp. 5681–5692, Oct. 2021.

    Parameters
    ----------
    params : dict
             A dictionary of parameter estimates computed by the function
             estimate_parameters().
    averaging : bool
                Default is False. If True, the tuning parameters alpha and beta
                are averaged over the classes and the average value is used in
                computing the covariance estimates.

    compute_inverse : bool
                      Default is False. If True, the inverses of the covariance
                      estimates are computed.

    Returns
    -------
    tuningparameters : ndarray of shape (2, n_classes, 1)
                       The estimated tuning parameters alpha and beta.

    Notes
    -----

    The coupled regularized sample covariance matrix estimator M_k for class k,
    where k = 1,2,...,K, is defined as

    M_k = alpha*C_k + (1-alpha)*trace(C_k)/n_features*I, where
    C_k = beta*SCM_k + (1-beta)*Spool,

    and where alpha and beta are tuning parameters, SCM_k is the sample
    covariance matrix of class k, I is the identity matrix, Spool is the pooled
    SCM defined as

    Spool = sum_{i=1}^K pi_k * SCM_k, where
    pi_k = n_samples_of_class_k / total_samples.

    References
    ----------

    E. Raninen and E. Ollila, “Coupled regularized sample covariance matrix
    estimator for multiple classes,” IEEE Transactions on Signal Processing,
    vol.  69, pp. 5681–5692, Oct. 2021.

    """
    p = params['p']  # dimension
    K = params['K']  # number of classes

    S2 = params['Etr_S2']
    IS2 = params['EtrS_2']/p
    SI2 = S2 - IS2

    Sk2 = params['EtrSiSj'].diagonal()
    ISk2 = params['EtrSitrSj'].diagonal()/p
    SkI2 = Sk2 - ISk2

    SkS = params['EtrSkS']
    ISkIS = params['EtrSktrS']/p
    SkISI = SkS - ISkIS

    SCk = params['EtrCkS']
    ISCk = params['EtrCktrS']/p
    SICk = SCk - ISCk

    Ck2 = params['trCiCj'].diagonal()
    SkCk = np.copy(Ck2)
    ISkCk = params['trCitrCj'].diagonal()/p
    SkICk = SkCk - ISkCk

    # coefficients for the mse polynomial
    C = dict()
    C["22"] = SkI2 + SI2 - 2*SkISI
    C["21"] = 2*(SkISI - SI2)
    C["12"] = 0
    C["20"] = SI2
    C["02"] = ISk2 + IS2 - 2*ISkIS
    C["11"] = -2*(SkICk - SICk)
    C["10"] = -2*SICk
    C["01"] = 2*(ISkIS - ISkCk - IS2 + ISCk)
    C["00"] = IS2 + Ck2 - 2*ISCk

    def msefunction(a, b, C):
        """Estimate the mse of the estimator with given the tuning parameters
        alpha and beta.

        Parameters
        ----------
        a : ndarray of shape (len(a),)
            Tuning parameter alpha.

        b : ndarray of shape (len(b),)
            Tuning parameter beta.

        C : dict
            Contains the coefficients of the mse polynomial.
        """
        return (a**2*b**2*C["22"]
                + a**2*b*C["21"]
                + a**2*C["20"]
                + b**2*C["02"]
                + a*b*C["11"]
                + a*C["10"]
                + b*C["01"]
                + C["00"])

    alpha_arr = np.linspace(0, 1, 21)
    beta_arr = np.linspace(0, 1, 21)
    mse_estimates = np.zeros((K, len(alpha_arr), len(beta_arr)))
    for i in range(len(alpha_arr)):
        for j in range(len(beta_arr)):
            mse_estimates[:, i, j] = msefunction(alpha_arr[i],
                                                 beta_arr[j],
                                                 C).reshape(4)

    min_value_of_class = np.zeros(K)
    alpha = np.zeros(K)
    beta = np.zeros(K)
    for k in range(K):
        min_value_of_class[k] = np.amin(mse_estimates[k, :, :])
        index_of_min_value = np.where(min_value_of_class[k]
                                      == mse_estimates[k, :, :])
        alpha[k] = alpha_arr[index_of_min_value[0][0]]
        beta[k] = beta_arr[index_of_min_value[1][0]]

    def opt_al(b):
        """Optimize alpha given beta"""
        return -1/2*(b*C["11"]+C["10"])/(b**2*C["22"]+b*C["21"]+C["20"])

    def opt_be(a):
        """optimize beta given alpha"""
        return -1/2*(a**2*C["21"]+a*C["11"]+C["01"])/(a**2*C["22"]+C["02"])

    iterMAX = 1000
    for iteration in range(iterMAX):
        alpha0 = np.copy(alpha)
        beta0 = np.copy(beta)
        alpha = np.clip(opt_al(beta), 0, 1)
        beta = np.clip(opt_be(alpha), 0, 1)
        crit = linalg.norm(alpha-alpha0 + beta-beta0)/linalg.norm(alpha0+beta0)
        if crit < 1e-8:
            break
    if iteration == iterMAX:
        print('rscmpool: slow convergence.')

    # averaging of the tuning parameters
    if averaging:
        alpha = alpha.mean()*np.ones(K)
        beta = beta.mean()*np.ones(K)

    # covariance matrix estimates
    out = dict()
    if not compute_inverse:
        Sigmas = list()
        for k in range(K):
            Sb = beta[k]*params['SCM'][k] + (1-beta[k])*params['Sp']
            Sigmas.append(alpha[k]*Sb + (1-alpha[k])*np.trace(Sb)/p*np.eye(p))
        out['Sigmas'] = Sigmas
    elif compute_inverse:
        temp = compute_inverse_rscm(params, alpha, beta, 'Sb')
        out['Sigmas'] = temp[0]
        out['iSigmas'] = temp[1]

    out['alpha'] = alpha
    out['beta'] = beta
    return out


def rscmpools(params, T='Sp', averaging=False, compute_inverse=False):
    """Compute regularization parameters for the shrinkage covariance matrix
    estimator proposed in E. Raninen and E. Ollila (2020). This is the
    streamlined analytical version of the estimator.

    Parameters
    ----------
    params : dict
             A dictionary of parameter estimates computed by the function
             estimate_parameters().

    T : str
        Default 'Sp'. If 'Sp', the pooled SCM is used for scaling the identity 
        matrix target. If 'Sk', the SCM is used for scaling the identity matrix
        target.

    averaging : bool
                Default is False. If True, the tuning parameters alpha and beta
                are averaged over the classes and the average value is used in
                computing the covariance estimates.

    compute_inverse : bool
                      Default is False. If True, the inverses of the covariance
                      estimates are computed.

    Returns
    -------
    tuningparameters : ndarray of shape (2, n_classes, 1)
                       The estimated tuning parameters alpha and beta.
    """
    p = params['p']  # dimension
    K = params['K']  # number of classes

    S2 = params['Etr_S2']
    IS2 = params['EtrS_2']/p

    Sk2 = params['EtrSiSj'].diagonal()
    ISk2 = params['EtrSitrSj'].diagonal()/p

    SkS = params['EtrSkS']
    ISkIS = params['EtrSktrS']/p

    SCk = params['EtrCkS']
    ISCk = params['EtrCktrS']/p

    Ck2 = params['trCiCj'].diagonal()
    SkCk = np.copy(Ck2)
    ISkCk = params['trCitrCj'].diagonal()/p

    if T == 'Sk':
        ISkIT = ISk2
        ISIT = ISkIS
        IT2 = ISk2
        ITCk = ISkCk
    elif T == 'Sp':
        ISkIT = ISkIS
        ISIT = IS2
        IT2 = IS2
        ITCk = ISCk
    else:
        print("Target can only be T={'Sk', 'Sp'}. \
               Using the pooled SCM as target, i.e., T='Sp'.")
        T = 'Sp'
        ISkIT = ISkIS
        ISIT = IS2
        IT2 = IS2
        ITCk = ISCk

    # Coefficients of MSE polynomial
    B = dict()
    B["22"] = Sk2 + S2 - 2*SkS
    B["21"] = 2*(SkS - S2 - ISkIT + ISIT)
    B["20"] = S2 + IT2 - 2*ISIT
    B["11"] = 2*(ISkIT - SkCk - ISIT + SCk)
    B["10"] = 2*(ISIT - SCk - IT2 + ITCk)
    B["00"] = IT2 + Ck2 - 2*ITCk

    # MSE polynomial
    def msefunction(a, b, B):
        return (a**2*b**2*B["22"] + a**2*b*B["21"] + a**2*B["20"] 
                + a*b*B["11"] + a*B["10"] + B["00"])

    # initialize alpha and beta candidates and their estimated mse
    al = np.zeros((5, K))
    be = np.zeros((5, K))
    mse = np.zeros((5, K))

    # when (alpha,beta) is in the interior of [0,1] X [0,1]
    al[0, ] = np.clip((2*B["10"]*B["22"]-B["11"]*B["21"])
                      / (B["21"]**2-4*B["20"]*B["22"]), 0, 1)
    be[0, ] = np.clip((2*B["11"]*B["20"]-B["10"]*B["21"])
                      / (2*B["10"]*B["22"]-B["11"]*B["21"]), 0, 1)
    mse[0, ] = msefunction(al[0, ], be[0, ], B)

    # at the borders of [0,1] X [0,1]
    al[1, ] = 0
    be[1, ] = 0  # when al = 0, the estimator doesn't depend on be
    mse[1, ] = msefunction(al[1, ], be[1, ], B)

    al[2, ] = 1
    be[2, ] = np.clip((-1/2)*(B["21"]+B["11"])/B["22"], 0, 1)  # al=1
    mse[2, ] = msefunction(al[2, ], be[2, ], B)

    al[3, ] = np.clip((-1/2)*B["10"]/B["20"], 0, 1)  # be=0
    be[3, ] = 0
    mse[3, ] = msefunction(al[3, ], be[3, ], B)

    al[4, ] = np.clip((-1/2)*(B["11"]+B["10"])
                      / (B["22"]+B["21"]+B["20"]), 0, 1)  # be=1
    be[4, ] = 1
    mse[4, ] = msefunction(al[4, ], be[4, ], B)
    
    alpha = np.zeros(K)
    beta = np.zeros(K)
    min_idx = np.argmin(mse, axis=0)
    for k in range(K):
        alpha[k] = al[min_idx[k], k]
        beta[k] = be[min_idx[k], k]

    # averaging of the tuning parameters
    if averaging:
        alpha = alpha.mean()*np.ones(K)
        beta = beta.mean()*np.ones(K)

    # covariance matrix estimates
    out = dict()

    if not compute_inverse:
        Sigmas = list()
        for k in range(K):
            Sb = beta[k]*params['SCM'][k] + (1-beta[k])*params['Sp']
            if T == 'Sk':
                Sigmas.append(
                        alpha[k]*Sb + (1-alpha[k])*params['eta'][k]*np.eye(p))
            elif T == 'Sp':
                Sigmas.append(
                        alpha[k]*Sb + (1-alpha[k])*np.trace(params['Sp'])
                        / p*np.eye(p))
        out['Sigmas'] = Sigmas
    elif compute_inverse:
        temp = compute_inverse_rscm(params, alpha, beta, T)
        out['Sigmas'] = temp[0]
        out['iSigmas'] = temp[1]

    out['alpha'] = alpha
    out['beta'] = beta
    return out


def compute_inverse_rscm(params, al, be, method):
    """Compute the inverse covariance matrix faster. Only faster if the
    dimension is greater than the combined number of samples of all classes.

    Parameters
    ----------

    params : dict
             A dictionary of parameter estimates computed by the function
             estimate_parameters().

    al, be : ndarray of shape (1,n_classes)
             Regularization parameters.

    method : str
             If the default estimator is used then use 'Sb'. If the streamlined
             analytical estimator is used, then use either 'Sp' or 'Sk'
             depending on the used target.

    Returns
    -------
    out : list of ndarrays of size (n_features,n_features).
          The covariance matrix estimates and their inverses.
    """
    K = params['K']  # number of classes
    n = params['n']  # number of samples
    p = params['p']  # dimension
    SCM = params['SCM']
    Sp = params['Sp']
    PI = params['PI']
    if sum(n) < p:
        X = params['centered_data']

    Sigmas = list()
    iSigmas = list()
    for k in range(K):
        # covariance matrices
        Sb = be[k]*SCM[k] + (1-be[k])*Sp
        if method == 'Sb':
            Sigmas.append(al[k]*Sb + (1-al[k])*np.trace(Sb)/p*np.eye(p))
        elif method == 'Sp':
            Sigmas.append(al[k]*Sb + (1-al[k])*np.trace(Sp)/p*np.eye(p))
        elif method == 'Sk':
            Sigmas.append(al[k]*Sb + (1-al[k])*params['eta'][k]*np.eye(p))

        # inverse covariance matrices
        if al[k] < 1 and sum(n) < p:
            c = np.array([])
            for j in range(K):
                if j == k:
                    cj = np.sqrt((be[k]+(1-be[k])*PI[j])
                                 / (n[j]-1))*np.ones(n[j])
                else:
                    cj = np.sqrt((1-be[k])*PI[j]/(n[j]-1))*np.ones(n[j])
                c = np.append(c, cj)
            A = c.reshape(sum(n), 1)*X

            # choose om based on which method is used.
            if method == 'Sb':
                om = (1-al[k])*np.trace(Sb)/p
            elif method == 'Sp':
                om = (1-al[k])*np.trace(Sp)/p
            elif method == 'Sk':
                om = (1-al[k])*params['eta'][k]
            alom = al[k]/om
            iSigmas.append(
                    (1/om)*(np.eye(p)
                            - alom*np.dot(A.T, linalg.solve(np.eye(sum(n))
                                          + alom*np.dot(A, A.T), A)))
                            )
        else:
            iSigmas.append(linalg.inv(Sigmas[k]))
    return list([Sigmas, iSigmas])


def linpool(params, linmethod='LIN2'):
    """Linear pooling method from the paper E. Raninen, D. E. Tyler, and E.
    Ollila, “Linear pooling of sample covariance matrices,” IEEE Transactions
    on Signal Processing, vol. 70, pp.  659-672, Dec. 2021.

    Parameters
    ----------

    params : dict
             A dictionary of parameter estimates computed by the function
             estimate_parameters().
    linmethod : str
                A string describing the method to be used.
                'LIN1' default value
                'LIN2' incorporates shrinkage toward the identity matrix

    Notes
    -----

    The LIN1 linpool estimator M_k for class k = 1,2,...,K is defined as

    M_k = sum_(i=1)^K a_k * SCM_k,

    where a_k are tuning parameters and SCM_k is the sample covariance matrix
    of class k. The LIN2 linpool estimator also incorporates shrinkage toward
    identity and is defined by

    M_k = sum_(i=1)^K a_k * SCM_k + a_I * I,

    where a_I is the tuning parameter for the identity matrix I.

    References
    ----------

    E. Raninen, D. E. Tyler, and E.  Ollila, “Linear pooling of sample
    covariance matrices,” IEEE Transactions on Signal Processing, vol. 70, pp.
    659-672, Dec. 2021.

    """
    K = params['K']
    p = params['p']

    C = params['trCiCj']/p
    D = np.diag(params['MSE_Sk'])/p

    if linmethod == 'LIN1':
        A = np.zeros((K, K))
    elif linmethod == 'LIN2':
        C = np.vstack((C, params['eta']))
        C = np.hstack((C, np.append(params['eta'], 1).reshape(K+1, 1)))
        D = np.vstack((D, np.zeros((1, K))))
        D = np.hstack((D, np.zeros((K+1, 1))))
        A = np.zeros((K+1, K+1))

    # The unconstrained solution. Used in case it is non-negative.
    A0 = np.dot(linalg.inv(C + D), C)

    # Solve the pooled estimators for each class
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    
    bnds = Bounds(0, np.inf)
    objfun = lambda x: 0.5*(x.dot(D+C)).dot(x) - C[:, k].dot(x)
    jacobian = lambda x: x.dot(D+C) - C[:, k]
    
    # For covariance estimates
    Sigmas = list()


    
    for k in range(K):
        # solve coefficients of linear combination
        if any(A0[:, k] < 0):
            # optimizes for nonnegative coefficients
            res = minimize(objfun, A0[:, k], jac=jacobian, bounds=bnds)
            A[:, k] = res.x
        else:
            A[:, k] = A0[:, k]
        # compute covariance matrix estimate of class k
        Sigma = np.zeros((p, p))
        for j in range(K):
            Sigma = Sigma + A[j, k]*params['SCM'][j]
        if linmethod == 'LIN2':
            Sigma = Sigma + A[K, k]*np.eye(p)
        Sigmas.append(Sigma)

    out = dict()
    out['Sigmas'] = Sigmas
    out['A'] = A[:, 0:K]
    return out


def generate_dataset(mu, C, df, n, p, K):
    """Generate data for multiple classes.

    Parameters
    ----------
    mu : list of arrays of shape (n_classes, n_features)
         The means of the classes.

    C :  list of arrays of shape (n_classes, n_features, n_features)
         Covariance matrices of the classes.

    df : list of shape (n_classes,)
         Degrees of freedom of the multivariate t distributed samples of the
         classes.

    n : list of shape (n_classes,)
        Number of samples per class.

    p : int
        Dimension (number of features).

    K : int
        Number of classes.
    Returns
    -------
    dataset : pandas.DataFrame of shape (sum(n), n_features + 1)
              The dataset, where the last column indicates the class.
    """
    for k in range(K):
        X = pd.DataFrame(
                generate_multivariate_t_samples(mu[k], C[k], df[k], n[k], p))
        X['class'] = k
        if k == 0:
            dataset = X
        else:
            dataset = pd.concat([dataset, X])
    return dataset
