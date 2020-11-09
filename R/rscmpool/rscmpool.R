rscmpool = function(params){
  #   Compute regularization parameters for the shrinkage covariance matrix estimator
  #   proposed in E. Raninen and E. Ollila (2020).
  #
  #   Parameters
  #   ----------
  #   params : A list of parameter estimates computed by the function
  #            estimate_parameters.
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
  SI2 <- S2 - IS2
  
  Sk2 <- diag(params$EtrSiSj)
  ISk2 <- diag(params$EtrSitrSj)/p
  SkI2 <- Sk2 - ISk2
  
  SkS <- params$EtrSkS
  ISkIS <- params$EtrSktrS/p
  SkISI <- SkS - ISkIS
  
  SCk <- params$EtrCkS
  ISCk <- params$EtrCktrS/p
  SICk <- SCk - ISCk
  
  Ck2 <- diag(params$trCiCj)
  SkCk <- Ck2
  ISkCk <- diag(params$trCitrCj)/p
  SkICk <- SkCk - ISkCk

  # Coefficients of MSE polynomial
  C22 <- SkI2 + SI2 - 2*SkISI
  C21 <- 2*SkISI - 2*SI2
  #C12 <- 0
  C20 <- SI2
  C02 <- ISk2 + IS2 - 2*ISkIS
  C11 <- -2*(SkICk - SICk)
  C10 <- -2*SICk
  C01 <- 2*(ISkIS - ISkCk - IS2 + ISCk)
  C00 <- IS2 + Ck2 - 2*ISCk
  
  # MSE polynomial
  MSE <- function(a,b){
    (a^2*b^2*C22+a^2*b*C21+a^2*C20+b^2*C02+a*b*C11+a*C10+b*C01+C00)
  }
  
  # function for projecting to [0,1]
  clamp <- function(x) min(max(0,x),1)

  # function for optimizing alpha given beta
  optimizeAlpha = function(b){
    ao <- -1/2*(b*C11+C10)/(b^2*C22+b*C21+C20)
    return(ao)
  }

  # function for optimizing beta given alpha
  optimizeBeta = function(a){
    bo <- -1/2*(a^2*C21+a*C11+C01)/(a^2*C22+C02)
    return(bo)
  }
  
  # Define grid of alpha and beta values
  grid.length <- 21
  grid.be <- matrix(seq(0,1,length.out=grid.length),nrow=grid.length,ncol=K)
  grid.al <- grid.be
  
  # initialize arrays
  grid.mse <- array(NA,c(grid.length,grid.length,K))
  best.idx <- matrix(NA,K,2)
  al <- rep(NA,K)
  be <- rep(NA,K)
  
  # loop over classes and grid
  for (i in 1:grid.length){
    # uniform grid
    for (j in 1:grid.length){
      grid.mse[i,j,] <- MSE(grid.al[i],grid.be[j])
    }
  }
  # best from uniform grid
  for (k in 1:K){
    best.idx[k,] <- which(grid.mse[,,k] == min(grid.mse[,,k]), arr.ind = TRUE)
    al[k] <- grid.al[best.idx[k,1]]
    be[k] <- grid.be[best.idx[k,2]]
  }
  
  # uncomment for plotting
  # k.plot = 2
  # al <- grid.be[,k.plot]
  # be <- al
  # persp(al,be,grid.mse[,,k.plot], phi=5, theta=45)
  
  # fine tune solution
  iterMAX <- 1000
  enorm <- function(x) sqrt(sum(x^2))
  for (iter in 1:iterMAX){
    al0 <- al
    be0 <- be
    al <- sapply(optimizeAlpha(be),clamp)
    be <- sapply(optimizeBeta(al),clamp)
    crit = enorm(c(al0-al,be0-be))/enorm(c(al0,be0))
    if (crit < 1e-8){
      break
    }
  }
  if (iter == iterMAX){
    print("slow convergence.")
  }
  
  # save results
  out <- list()
  out[["alAve"]] <- mean(al)
  out[["beAve"]] <- mean(be)
  out[["al"]] <- al
  out[["be"]] <- be
  
  return(out)
}