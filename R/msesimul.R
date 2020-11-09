rm(list=ls())
library("MASS")
# the files for the proposed method
source("rscmpool/estimate_parameters.R")
source("rscmpool/rscmpool.R")
source("rscmpool/rscmpools.R")


set.seed(0)

# function for creating a first order autoregressive AR(1) covariance matrix
ARcov = function(r,p){
  return (r^abs(outer(1:p,1:p,"-")))
}
# function for creating a compound symmetry matrix
CScov = function(r,p){
  return (r*matrix(1,p,p) + (1-r)*diag(p))
}

# function for computing normalized squared error
computeNSE = function(A,Sigma){
  return (norm(A-Sigma,"F")^2 / norm(Sigma,"F")^2)
}

# Common settings for simulation setups
mctrials <- 4000 # number of Monte Carlo trials
K <- 4 # number of classes
p <- 200 # dimension

# Setup A: AR(1) structured covariance matrices
n <- c(25,50,75,100) # number of samples
dof <- c(8,8,8,8) # degrees of freedom for multivariate t data
rho <- c(0.2,0.3,0.4,0.5) 
C <- list(ARcov(rho[1],p), ARcov(rho[2],p), ARcov(rho[3],p), ARcov(rho[4],p)) # covariance matrices 

# Setup B: Compound symmetry structured covariance matrices
#n <- c(25,50,75,100) # number of samples
#dof <- c(8,8,8,8) # degrees of freedom for multivariate t data
#rho <- c(0.2,0.3,0.4,0.5) 
#C <- list(CScov(rho[1],p), CScov(rho[2],p), CScov(rho[3],p), CScov(rho[4],p)) # covariance matrices

# Setup C: Mixed setup
#n <- c(100,100,100,100) # number of samples
#dof <- c(12,8,12,8) # degrees of freedom for multivariate t data
#rho <- c(0.6,0.6,0.1,0.1)
#C <- list(ARcov(rho[1],p), ARcov(rho[2],p), CScov(rho[3],p), CScov(rho[4],p)) # covariance matrices




# Main Monte Carlo loop

mu <- list(rnorm(p),rnorm(p),rnorm(p),rnorm(p)) # class means remain fixed over Monte Carlo trials
PI <- n / sum(n) # proportion of samples in each class

# Initialize matrices for storing normalized squared error
NSE1 <- matrix(NA,mctrials,K)
NSE2 <- matrix(NA,mctrials,K)
NSE1Ave <- matrix(NA,mctrials,K)
NSE2Ave <- matrix(NA,mctrials,K)

for (mc in 1:mctrials){
  
  Sk <- array(NA,c(p,p,K)) # for sample covariance matrices
  Sp <- matrix(0,p,p) # for the pooled sample covariance matrix
  
  # generate data
  X <- matrix(0,0,p)
  y <- matrix(0,0,1)
  for (k in 1:K){
    Xk <- rep(mu[[k]], each=n[k]) + mvtnorm::rmvt(n[k], sigma = (dof[k]-2)/dof[k]*C[[k]], df = dof[k])
    yk <- matrix(k,n[k],1)
    
    X <- rbind(X,Xk)
    y <- rbind(y,yk)
    
    Sk[,,k] = cov(Xk)
    Sp <- Sp + PI[k]*Sk[,,k]
  }
  
  # estimate parameters and tuning parameters
  params <- estimate_parameters(X,y)
  
  # estimator
  POLYmethod <- rscmpool(params)
  
  # streamlined analytical estimator
  Target = "S"
  POLYsmethod <- rscmpools(params,Target)
  
  # construct covariance matrix estimates and compute normalized squared error
  for (k in 1:K){
    
    # POLY method
    al <- POLYmethod$al[k] # alpha tuning parameter of class k
    be <- POLYmethod$be[k] # beta tuning parameter of class k
    Sb   <- be*Sk[,,k] + (1-be)*Sp # partially pooled estimator
    ISb  <- sum(diag(Sb)) * diag(p) / p # scaled identity matrix target
    Chat <- al*Sb + (1-al)*ISb # final covariance matrix estimate
    NSE1[mc,k] <- computeNSE(Chat,C[[k]]) # compute normalized squared error
    
    # POLY method with averaging
    al <- POLYmethod$alAve
    be <- POLYmethod$beAve
    Sb   <- be*Sk[,,k] + (1-be)*Sp # partially pooled estimator
    ISb  <- sum(diag(Sb)) * diag(p) / p # scaled identity matrix target
    Chat <- al*Sb + (1-al)*ISb # final covariance matrix estimate
    NSE1Ave[mc,k] <- computeNSE(Chat,C[[k]]) # compute normalized squared error
    
    # POLYs method (streamlined analytical)
    al <- POLYsmethod$al[k]
    be <- POLYsmethod$be[k]
    Sb <- be*Sk[,,k] + (1-be)*Sp
    if (Target == "S"){
      IT = sum(diag(Sp))/p*diag(p)
    } else if (Target == "Sk"){
      IT = sum(diag(Sk[,,k]))/p*diag(p)
    }
    Chat <- al*Sb + (1-al)*IT
    NSE2[mc,k] <- computeNSE(Chat,C[[k]])
    
    # POLYs method (streamlined analytical) with averaging
    al <- POLYsmethod$alAve
    be <- POLYsmethod$beAve
    Sb <- be*Sk[,,k] + (1-be)*Sp
    if (Target == "S"){
      IT = sum(diag(Sp))/p*diag(p)
    } else if (Target == "Sk"){
      IT = sum(diag(Sk[,,k]))/p*diag(p)
    }
    Chat <- al*Sb + (1-al)*IT
    NSE2Ave[mc,k] <- computeNSE(Chat,C[[k]])
  }
  # print progress
  if (mc %% 50 == 0){
    print(mc)
  }
}

# Compute empirical NMSE and standard deviation

# NMSE of classes
NMSE_POLY  <- c(t(apply(NSE1,2,mean)),mean(apply(NSE1,1,sum)))
NMSE_POLYs <- c(t(apply(NSE2,2,mean)),mean(apply(NSE2,1,sum)))
NMSE_POLYAve  <- c(t(apply(NSE1Ave,2,mean)),mean(apply(NSE1Ave,1,sum)))
NMSE_POLYsAve <- c(t(apply(NSE2Ave,2,mean)),mean(apply(NSE2Ave,1,sum)))

NMSE <- as.data.frame(rbind(NMSE_POLY,NMSE_POLYs,NMSE_POLYAve,NMSE_POLYsAve))
colnames(NMSE) = c("class1","class2","class3","class4","sum")

# standard deviation of NSE
STD_POLY  <- c(t(apply(NSE1,2,sd)),sd(apply(NSE1,1,sum)))
STD_POLYs <- c(t(apply(NSE2,2,sd)),sd(apply(NSE2,1,sum)))
STD_POLYAve  <- c(t(apply(NSE1Ave,2,sd)),sd(apply(NSE1Ave,1,sum)))
STD_POLYsAve <- c(t(apply(NSE2Ave,2,sd)),sd(apply(NSE2Ave,1,sum)))
STD <- as.data.frame(rbind(STD_POLY,STD_POLYs,STD_POLYAve,STD_POLYsAve))
colnames(STD) = c("class1","class2","class3","class4","sum")

print(NMSE*10, digits=3)
print(STD*10, digits=3)
print("Note that the NMSE and STD are multiplied by 10", quote=FALSE)
