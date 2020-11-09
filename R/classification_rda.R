rm(list=ls())
library(mlbench)

# ------- load Ionosphere data set
fname = "ionosphere"
data("Ionosphere")
Ionosphere = Ionosphere[,-c(1,2)]
DS = Ionosphere

# ------- load Sonar data set
#fname <- "sonar"
#data(Sonar)
#DS <- Sonar

# ------- load Vowel data set
#fname <- "vowel"
#data(Vowel)
#Vowel <- Vowel[,-1]
#DS <- Vowel

# ------- simulation settings
plotresults = TRUE
preprocess = "no_preprocessing"

training_ratio = seq(from = 0.05, to = 0.8, length.out = 16)
tplen = length(training_ratio)
mctrials = 10 # mc trials per training ratio

# save results to folder
#folder = "../latex/results/"
folder = ""


#--------

library(caret)
library(klaR)
library(tictoc)

# the files needed for our method
source("rscmpool/estimate_parameters.R")
source("rscmpool/rscmpool.R")

set.seed(1)

# ------- initialize

# for saving the tuning parameters
rda.caret.cv5.tuningparameters  = array(NA,c(tplen,2,mctrials))
rda.caret.cv10.tuningparameters = array(NA,c(tplen,2,mctrials))
rda.klaR.cv5.tuningparameters   = array(NA,c(tplen,2,mctrials))
rda.klaR.cv10.tuningparameters  = array(NA,c(tplen,2,mctrials))
rda.pool.tuningparameters       = array(NA,c(tplen,2,mctrials))

# for saving the accuracy of classification
rda.caret.cv5.acc  = matrix(NA,mctrials,tplen)
rda.caret.cv10.acc = matrix(NA,mctrials,tplen)
rda.klaR.cv5.acc   = matrix(NA,mctrials,tplen)
rda.klaR.cv10.acc  = matrix(NA,mctrials,tplen)
rda.pool.acc       = matrix(NA,mctrials,tplen)

# for saving the computation time of training the methods
rda.caret.cv5.time  = matrix(NA,mctrials,tplen)
rda.caret.cv10.time = matrix(NA,mctrials,tplen)
rda.klaR.cv5.time   = matrix(NA,mctrials,tplen)
rda.klaR.cv10.time  = matrix(NA,mctrials,tplen)
rda.pool.time       = matrix(NA,mctrials,tplen)

# for caret train function
rda.caret.cv5.trainControl  = trainControl(method = "cv", number = 5)
rda.caret.cv10.trainControl = trainControl(method = "cv", number = 10)

# ------- main loop
for (iperc in 1:tplen) { # loop over training set sizes
  for (mc in 1:mctrials) { # loop over Monte Carlo trials
    
    # partition into training and test sets
    inTraining = createDataPartition(DS$Class, p = training_ratio[iperc], list = FALSE)
    
    # if preprocessing is applied
    if (preprocess == "centered_and_scaled") {
      preProc = preProcess(DS[inTraining,], method=c("center","scale"))
      training = predict(preProc, newdata = DS[inTraining,])
      testing = predict(preProc, newdata = DS[-inTraining,])
    } else {
      training = DS[ inTraining,]
      testing  = DS[-inTraining,]
    }
    
    # -------- the proposed method
    tic()
    train.set.pool = training[,names(training) != "Class"]
    train.responses.pool = training$Class
    params = estimate_parameters(train.set.pool,train.responses.pool)
    res = rscmpool(params)
    rda.pool.fit = rda(Class~., data = training, gamma = 1-res$alAve, lambda = 1-res$beAve, prior = 1)
    tmp = toc()
    rda.pool.time[mc,iperc] = (tmp$toc-tmp$tic)
    
    # -------- cross-validation using 2D-grid (uses KlaR package via caret)
    
    # --- 5-fold CV
    tic()
    rda.caret.cv5.fit = train(Class~., 
                        data = training, 
                        method = "rda", 
                        trControl = rda.caret.cv5.trainControl, 
                        tuneLength = 5,
                        prior = 1)
    tmp = toc()
    rda.caret.cv5.time[mc,iperc] = (tmp$toc-tmp$tic)
    
    # --- 10-fold CV
    tic()
    rda.caret.cv10.fit = train(Class~., 
                         data = training, 
                         method = "rda", 
                         trControl = rda.caret.cv10.trainControl, 
                         tuneLength = 9,
                         prior = 1)
    tmp = toc()
    rda.caret.cv10.time[mc,iperc] = (tmp$toc-tmp$tic)
    
    # -------- cross-validation using Nelder-Mead method (KlaR package)
    # 5-fold CV
    tic()
    rda.klaR.cv5.fit = rda(Class~., data = training, crossval=TRUE, fold=5, prior=1)
    tmp = toc()
    rda.klaR.cv5.time[mc,iperc] = (tmp$toc-tmp$tic)
    
    # 10-fold CV
    tic()
    rda.klaR.cv10.fit = rda(Class~., data = training, crossval=TRUE, fold=10, prior=1)
    tmp = toc()
    rda.klaR.cv10.time[mc,iperc] = (tmp$toc-tmp$tic)


    # -------- make predictions on test data
    rda.caret.cv5.pred  = predict(rda.caret.cv5.fit,  newdata = testing)
    rda.caret.cv10.pred = predict(rda.caret.cv10.fit, newdata = testing)
    rda.klaR.cv5.pred   = predict(rda.klaR.cv5.fit,   newdata = testing)$class
    rda.klaR.cv10.pred  = predict(rda.klaR.cv10.fit,  newdata = testing)$class
    rda.pool.pred       = predict(rda.pool.fit,       newdata = testing)$class
    
    rda.caret.cv5.acc[mc,iperc]  = mean(rda.caret.cv5.pred == testing$Class)
    rda.caret.cv10.acc[mc,iperc] = mean(rda.caret.cv10.pred == testing$Class)
    rda.klaR.cv5.acc[mc,iperc]   = mean(rda.klaR.cv5.pred == testing$Class)
    rda.klaR.cv10.acc[mc,iperc]  = mean(rda.klaR.cv10.pred == testing$Class)
    rda.pool.acc[mc,iperc]       = mean(rda.pool.pred == testing$Class)
    
    rda.caret.cv5.tuningparameters[iperc,,mc]  = 1-as.matrix(rda.caret.cv5.fit$finalModel$regularization)
    rda.caret.cv10.tuningparameters[iperc,,mc] = 1-as.matrix(rda.caret.cv10.fit$finalModel$regularization)
    rda.klaR.cv5.tuningparameters[iperc,,mc]   = 1-as.matrix(rda.klaR.cv5.fit$regularization)
    rda.klaR.cv10.tuningparameters[iperc,,mc]  = 1-as.matrix(rda.klaR.cv10.fit$regularization)
    rda.pool.tuningparameters[iperc,,mc]       = 1-as.matrix(c(gamma=1-res$alAve,lambda=1-res$beAve))
    
    # print progress
    print(paste('mc:',mc,'/',mctrials,' and ','training ratio: ', iperc, '/', length(training_ratio)))
  }
}

# ------- compute averages over Monte Carlos

# average accuracy over mc runs
rda.caret.cv5.acc.mean  = colMeans(rda.caret.cv5.acc)
rda.caret.cv10.acc.mean = colMeans(rda.caret.cv10.acc)
rda.klaR.cv5.acc.mean   = colMeans(rda.klaR.cv5.acc)
rda.klaR.cv10.acc.mean  = colMeans(rda.klaR.cv10.acc)
rda.pool.acc.mean       = colMeans(rda.pool.acc)

# average regularization parameters alpha and beta over the mc runs
rda.caret.cv5.tuningparameters.mean  = apply(rda.caret.cv5.tuningparameters,c(1,2), mean)
rda.caret.cv10.tuningparameters.mean = apply(rda.caret.cv10.tuningparameters,c(1,2), mean)
rda.klaR.cv5.tuningparameters.mean   = apply(rda.klaR.cv5.tuningparameters,c(1,2), mean)
rda.klaR.cv10.tuningparameters.mean  = apply(rda.klaR.cv10.tuningparameters,c(1,2), mean)
rda.pool.tuningparameters.mean       = apply(rda.pool.tuningparameters,c(1,2), mean)
colnames(rda.caret.cv5.tuningparameters.mean)  = c("alpha.caret.cv5.tl5","beta.caret.cv5.tl5")
colnames(rda.caret.cv10.tuningparameters.mean) = c("alpha.caret.cv10.tl9","beta.caret.cv10.tl9")
colnames(rda.klaR.cv5.tuningparameters.mean)   = c("alpha.klaR.cv5.NelderMead","beta.klaR.cv5.NelderMead")
colnames(rda.klaR.cv10.tuningparameters.mean)  = c("alpha.klaR.cv10.NelderMead","beta.klaR.cv10.NelderMead")
colnames(rda.pool.tuningparameters.mean)       = c("alpha.poolave","beta.poolave")

# average running time over mc runs
rda.caret.cv5.time.mean  = colMeans(rda.caret.cv5.time)
rda.caret.cv10.time.mean = colMeans(rda.caret.cv10.time)
rda.klaR.cv5.time.mean   = colMeans(rda.klaR.cv5.time)
rda.klaR.cv10.time.mean  = colMeans(rda.klaR.cv10.time)
rda.pool.time.mean       = colMeans(rda.pool.time)

# median running time over mc runs
rda.caret.cv5.time.median  = apply(rda.caret.cv5.time,2,median)
rda.caret.cv10.time.median = apply(rda.caret.cv10.time,2,median)
rda.klaR.cv5.time.median   = apply(rda.klaR.cv5.time,2,median)
rda.klaR.cv10.time.median  = apply(rda.klaR.cv10.time,2,median)
rda.pool.time.median       = apply(rda.pool.time,2,median)

# ------- save data to file

# combine data
datatosave = cbind(training_ratio,
                    rda.caret.cv5.acc.mean,
                    rda.caret.cv10.acc.mean,
                    rda.klaR.cv5.acc.mean,
                    rda.klaR.cv10.acc.mean,
                    rda.pool.acc.mean,
                    rda.caret.cv5.tuningparameters.mean,
                    rda.caret.cv10.tuningparameters.mean,
                    rda.klaR.cv5.tuningparameters.mean,
                    rda.klaR.cv10.tuningparameters.mean,
                    rda.pool.tuningparameters.mean,
                    rda.caret.cv5.time.mean,
                    rda.caret.cv10.time.mean,
                    rda.klaR.cv5.time.mean,
                    rda.klaR.cv10.time.mean,
                    rda.pool.time.mean,
                    rda.caret.cv5.time.median,
                    rda.caret.cv10.time.median,
                    rda.klaR.cv5.time.median,
                    rda.klaR.cv10.time.median,
                    rda.pool.time.median)

# write to file
write.table(datatosave,
            file = paste(folder,"classification_results_",fname,"_",preprocess,".dat",sep=""), 
            sep=" ",
            row.names = F,
            quote = F)

# save whole workspace
save.image(file = paste("RData/",fname,"_",preprocess,".RData",sep=""))


# ------- plot results
if (plotresults == TRUE){
# mean classification accuracy
plot(training_ratio, rda.caret.cv5.acc.mean, ylim=c(0.45,1), type="o", col="red", pch=1, xlab="training ratio", ylab="mean classification accuracy", main=paste(fname,"accuracy"))
points(training_ratio, rda.caret.cv10.acc.mean, type="o", col = "blue", pch=2)
points(training_ratio, rda.klaR.cv5.acc.mean, type="o", col = "black", pch=3)
points(training_ratio, rda.klaR.cv10.acc.mean, type="o", col = "green", pch=4)
points(training_ratio, rda.pool.acc.mean, type="o", col = "dark red", pch=5)
legend("bottomright",c("5-fold CV caret", "10-fold CV caret", "5-fold CV NM", "10-fold CV NM", "POLY-Ave"),col=c("red","blue","black","green","dark red"), pch=c(1,2,3,4,5))

# mean computation time
plot(training_ratio, rda.caret.cv5.time.mean, ylim=c(0,1), type="o", col="red", pch=1, xlab="training ratio", ylab="mean computation time", main=paste(fname,"computation time"))
points(training_ratio, rda.caret.cv10.time.mean, type="o", col="blue", pch=2)
points(training_ratio, rda.klaR.cv5.time.mean, type="o", col="black", pch=3)
points(training_ratio, rda.klaR.cv10.time.mean, type="o", col = "green", pch=4)
points(training_ratio, rda.pool.time.mean, type="o", col = "dark red", pch=5)
legend("topright",c("5-fold CV caret", "10-fold CV caret", "5-fold CV NM", "10-fold CV NM", "POLY-Ave"),col=c("red","blue","black","green","dark red"), pch=c(1,2,3,4,5))

# median computation time
plot(training_ratio, rda.caret.cv5.time.median, ylim=c(0,1), type="o", col="red", pch=1, xlab="training ratio", ylab="median computation time", main=paste(fname,"computation time"))
points(training_ratio, rda.caret.cv10.time.median, type="o", col="blue", pch=2)
points(training_ratio, rda.klaR.cv5.time.median, type="o", col="black", pch=3)
points(training_ratio, rda.klaR.cv10.time.median, type="o", col = "green", pch=4)
points(training_ratio, rda.pool.time.median, type="o", col = "dark red", pch=5)
legend("topright",c("5-fold CV caret", "10-fold CV caret", "5-fold CV NM", "10-fold CV NM", "POLY-Ave"),col=c("red","blue","black","green","dark red"), pch=c(1,2,3,4,5))


# regularization parameters
par(mfrow=c(1,2))
# alpha
plot(training_ratio, rda.caret.cv5.tuningparameters.mean[,1], ylim=c(0,1), type="o", col="red", pch=1, xlab="training ratio", ylab="alpha", main=paste(fname,"alpha"))
points(training_ratio, rda.caret.cv10.tuningparameters.mean[,1], type="o", col="blue", pch=2)
points(training_ratio, rda.klaR.cv5.tuningparameters.mean[,1], type="o", col="black", pch=3)
points(training_ratio, rda.klaR.cv10.tuningparameters.mean[,1], type="o", col="green", pch=4)
points(training_ratio, rda.pool.tuningparameters.mean[,1], type="o", col="dark red", pch=5)
legend("bottomright",c("5-fold CV caret", "10-fold CV caret", "5-fold CV NM", "10-fold CV NM", "POLY-Ave"),col=c("red","blue","black","green","dark red"), pch=c(1,2,3,4,5))

# beta
plot(training_ratio, rda.caret.cv5.tuningparameters.mean[,2], ylim=c(0,1), type="o", col="red", pch=1, xlab="training ratio", ylab="beta", main=paste(fname,"beta"))
points(training_ratio, rda.caret.cv10.tuningparameters.mean[,2], type="o", col="blue", pch=2)
points(training_ratio, rda.klaR.cv5.tuningparameters.mean[,2], type="o", col="black", pch=3)
points(training_ratio, rda.klaR.cv10.tuningparameters.mean[,2], type="o", col="green", pch=4)
points(training_ratio, rda.pool.tuningparameters.mean[,2], type="o", col="dark red", pch=5)
legend("bottomright",c("5-fold CV caret", "10-fold CV caret", "5-fold CV NM", "10-fold CV NM", "POLY-Ave"),col=c("red","blue","black","green","dark red"), pch=c(1,2,3,4,5))
}