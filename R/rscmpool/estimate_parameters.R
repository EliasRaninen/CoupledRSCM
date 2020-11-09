estimate_parameters = function(train.set, train.responses){
  # train.set is the training data
  # train.responses is the training data response
  
  require(moments) # for kurtosis
  require(SpatialNP)  # for spatial sign covariance

  # frobenius inner product
  ip = function(A,B){
    return(sum(A*t(B)))
  }
  
  params = list() # save output parameters to list
  
  
  rnames <- unique(train.responses) # names of responses
  K <- length(unique(train.responses)) # number of classes
  n <- array(NA,K)
  for (k in 1:K){
    n[k] = sum(rnames[k] == train.responses)
  }
  p <- ncol(train.set) # dimension
  PI = n/sum(n) # proportions of samples
  
  params[["p"]] <- p
  params[["K"]] <- K
  
  # estimate necessary statistics
  Sk <- array(NA,c(p,p,K)) # SCM
  SSCMk <- array(NA,c(p,p,K)) # spatial sign covariance (SSCM)
  kappa <- array(NA,K) # elliptical kurtosis
  eta <- array(NA,K) # scale parameter tr(C)/p = trace(covariancematrix)/p
  gam <- array(NA,K) # sphericity: p*tr(C^2)/tr(C)^2
  
  # kappa, eta, gamma
  for (k in 1:K){
    classk <- (rnames[k] == train.responses) # indices
    xk <- as.matrix(train.set[classk,]) # data
    Sk[,,k] <- cov(xk)
    SSCMk[,,k] <- SpatialNP::SCov(xk)

    kappa[k] <- mean(pmax(kurtosis(xk)/3-1, -2/(p+2)), na.rm = TRUE)
    eta[k] <- sum(diag(Sk[,,k]))/p
    gam[k] <- min(p,max(1,p*n[k]/(n[k]-1)*(ip(SSCMk[,,k],SSCMk[,,k])-1/n[k])))
  }
  params[["kappa"]] = kappa
  params[["eta"]] = eta
  params[["sphericity"]] = gam
  
  trCk <- eta*p
  tr_Ck2 <- p*eta^2*gam
  trCk_2 <- p^2*eta^2
  
  tau1 <- 1/(n-1)+kappa/n
  tau2 <- kappa/n
  
  Etr_Sk2 <- tau1*trCk_2 + (1 + tau1 + tau2)*tr_Ck2
  EtrSk_2 <- 2*tau1*tr_Ck2+(1+tau2)*trCk_2
  
  # estimate inner products of true covariance matrices
  trCiCj <- array(NA,c(K,K))
  trCitrCj <- array(NA,c(K,K))
  
  for (k in 1:K){
    trCiCj[k,k] = tr_Ck2[k]
    trCitrCj[k,k] = trCk_2[k]
    
    j.start = k+1
    if (j.start <= K){
      for (j in j.start:K){
        trCitrCj[k,j] <- eta[k]*eta[j]*p^2
        trCitrCj[j,k] <- trCitrCj[k,j]

        trCiCj[k,j] <- ip(SSCMk[,,k],SSCMk[,,j])*trCitrCj[k,j]
        trCiCj[j,k] <- trCiCj[k,j]
      }
    }
  }
  params[["trCiCj"]] <- trCiCj
  params[["trCitrCj"]] <- trCitrCj
  
  # expectation of the inner products of SCMs
  EtrSiSj <- trCiCj
  EtrSitrSj <- trCitrCj
  for (k in 1:K){
    EtrSiSj[k,k] <- Etr_Sk2[k]
    EtrSitrSj[k,k] <- EtrSk_2[k]
  }
  params[["EtrSiSj"]] = EtrSiSj
  params[["EtrSitrSj"]] = EtrSitrSj
  
  # inner products involving pooled sample covariance matrix
  PIiPIj <- outer(PI,PI)
  PImat <- matrix(rep(PI,K), nrow=K, ncol=K)
  params[["Etr_S2"]] <- sum(PIiPIj*EtrSiSj)
  params[["EtrS_2"]] <- sum(PIiPIj*EtrSitrSj)
  params[["EtrSkS"]] <- colSums(PImat*EtrSiSj)
  params[["EtrCkS"]] <- colSums(PImat*trCiCj)
  params[["EtrSktrS"]] <- colSums(PImat*EtrSitrSj)
  params[["EtrCktrS"]] <- colSums(PImat*trCitrCj)
  
  return(as.array(params))
}
