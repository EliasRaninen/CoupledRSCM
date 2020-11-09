rscmpools = function(params, T){
  #   Compute regularization parameters for the shrinkage covariance matrix estimator
  #   proposed in E. Raninen and E. Ollila (2020). This is the streamlined analytical version
  #   of the estimator.
  # 
  #   Parameters
  #   ----------
  #   params : A list of parameter estimates computed by the function
  #            estimate_parameters.
  #            
  #   T : Default "S". If "S", the pooled SCM is used for scaling the identity 
  #       matrix target. If "Sk", the SCM is used for scaling the identity matrix
  #       target.
  #       
  #   Returns
  #   -------
  #   out : a list containingt the estimated tuning parameters alpha and beta.
  #
  
  p <- params$p # dimension
  K <- params$K # number of classes
  
  # Terms for computing the coefficients of the MSE polynomial
  S2 <- params$Etr_S2
  IS2 <- params$EtrS_2/p

  Sk2 <- diag(params$EtrSiSj)
  ISk2 <- diag(params$EtrSitrSj)/p

  SkS <- params$EtrSkS
  ISkIS <- params$EtrSktrS/p

  SCk <- params$EtrCkS
  ISCk <- params$EtrCktrS/p

  Ck2 <- diag(params$trCiCj)
  SkCk <- Ck2
  ISkCk <- diag(params$trCitrCj)/p

  if (T == "Sk"){
    ISkIT = ISk2
    ISIT = ISkIS
    IT2 = ISk2
    ITCk = ISkCk
  } else if (T == "S"){
    ISkIT = ISkIS
    ISIT = IS2
    IT2 = IS2
    ITCk = ISCk
  }
  
  # Coefficients of MSE polynomial
  B22 = Sk2 + S2 - 2*SkS
  B21 = 2*(SkS - S2 - ISkIT + ISIT)
  B20 = S2 + IT2 - 2*ISIT
  B11 = 2*(ISkIT - SkCk - ISIT + SCk)
  B10 = 2*(ISIT - SCk - IT2 + ITCk)
  B00 = IT2 + Ck2 - 2*ITCk
  
  # MSE polynomial
  msefunction <- function(a,b){
    (a^2*b^2*B22+a^2*b*B21+a^2*B20+a*b*B11+a*B10+B00)
  }
  
  # function for projecting to [0,1]
  clamp <- function(x) min(max(0,x),1)
  
  # initialize arrays
  al <- matrix(NA,5,K)
  be <- matrix(NA,5,K)
  mse <- matrix(NA,5,K)
  
  # when (alpha,beta) is in the interior of [0,1] X [0,1]
  al[1,] = sapply((2*B10*B22-B11*B21)/(B21^2-4*B20*B22), clamp)
  be[1,] = sapply((2*B11*B20-B10*B21)/(2*B10*B22-B11*B21), clamp)
  mse[1,] = msefunction(al[1,],be[1,])
  
  # at the borders of [0,1] X [0,1]
  al[2,] = 0
  be[2,] = 0 # when al = 0, the estimator doesn't depend on be
  mse[2,] = msefunction(al[2,],be[2,])
  
  al[3,]  = 1
  be[3,] = sapply((-1/2)*(B21+B11)/B22,clamp) # al=1
  mse[3,] = msefunction(al[3,],be[3,])
  
  al[4,] = sapply((-1/2)*B10/B20,clamp) # be=0
  be[4,] = 0
  mse[4,] = msefunction(al[4,],be[4,])
  
  al[5,] = sapply((-1/2)*(B11+B10)/(B22+B21+B20),clamp) # be=1
  be[5,] = 1
  mse[5,] = msefunction(al[5,],be[5,])
  
  # choose best alpha and beta
  alopt = rep(NA,K)
  beopt = rep(NA,K)
  for (k in 1:K){
    bestidx <- which(mse[,k] == min(mse[,k]), arr.ind = TRUE)
    alopt[k] <- al[bestidx[1],k]
    beopt[k] <- be[bestidx[1],k]
  }
  
  # save results
  result <- list()
  result[["alAve"]] <- mean(alopt)
  result[["beAve"]] <- mean(beopt)
  result[["al"]] <- alopt
  result[["be"]] <- beopt
  
  return(result)
}