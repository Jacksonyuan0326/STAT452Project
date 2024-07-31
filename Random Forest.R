data <- read.csv("training_data.csv", header=TRUE)
head(data)
### We will also often need to calculate MSE using an observed
### and a prediction vector. This is another useful function.
get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}

### Create k CV folds for a dataset of size n
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
  fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels
  
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  
  return(folds.rand)
}


#################################################################################
### In this tutorial, we will use random forests on the wine quality data to  ###
### predict Y using a subset of the other predictors. We will also see  ###
### a new way to do tuning using out-of-bag error                             ###
#################################################################################

library(randomForest)

set.seed(36597804)

#source("Tutorials/Read_Wine_Data.R")
#source("Tutorials/Sec05_Tutorial_Helper.R")


### We will start by fitting a single random forest so we can see how it works.
### Random forests are fit in R using the randomForest() function in the randomForest
### package. This function uses formula/data syntax. Other options include ntree,
### nodesize, and mtry for, respectively, the number of trees to fit, the terminal
### node size, and the number of predictor candidates to include for each tree. You
### can also set importance to TRUE to get variable importance, and keep.forest to
### TRUE if you want to store the resulting forest (you will pretty much always want
### to do this, and the default is TRUE, so we will just leave out keep.forest).
### Let's just leave all the settings at their default values and see how we do
### (we do need to set importance to TRUE)
fit.rf.1 <- randomForest(Y ~ ., data = data, importance = T)

### Plotting this RF object lets us see if we used enough trees. If not, you can
### change this by re-running randomForest() and setting ntree to a higher
### number (default is 500)
plot(fit.rf.1)

### We can get variable importance measures using the importance() function, and
### we can plot them using VarImpPlot()
importance(fit.rf.1)
varImpPlot(fit.rf.1)

### We can get out-of-bag (OOB) error directly from the predict() function.
### Specifically, if we don't include a new dataset, R gives the OOB predictions
### on the training set.
OOB.pred.1 <- predict(fit.rf.1)
(OOB.MSPE.1 <- get.MSPE(data$Y, OOB.pred.1))

### For reference, we can get the SMSE by using our training set with predict()
sample.pred.1 <- predict(fit.rf.1, data)
(SMSE.1 <- get.MSPE(data$Y, sample.pred.1))



###############################################################################
### Now, let's look at how to tune random forests using OOB error. We will  ###
### consider mtry = 1,2,3,4 and nodesize = 2, 5, 8.                         ###
###############################################################################

### Set parameter values
#all.mtry <- 1:4
#all.nodesize <- c(2, 5, 8)
#all.mtry <- 2:4
#all.nodesize <- c(5, 8)
#all.mtry <- c(9,10,11,12) 
#all.nodesize <- c(4,5)
all.mtry <- c(11)
all.nodesize <- c(4)
all.pars <- expand.grid(mtry = all.mtry, nodesize = all.nodesize)
n.pars <- nrow(all.pars)

### Number of times to replicate process. OOB errors are based on bootstrapping,
### so they are random and we should repeat multiple runs
M <- 5

### Create container for OOB MSPEs
OOB.MSPEs <- array(0, dim = c(M, n.pars))

for (i in 1:n.pars) {
  ### Print progress update
  print(paste0(i, " of ", n.pars))
  
  ### Get current parameter values
  this.mtry <- all.pars[i, "mtry"]
  this.nodesize <- all.pars[i, "nodesize"]
  
  ### Fit random forest models for each parameter combination
  ### A second for loop will make our life easier here
  for (j in 1:M) {
    ### Fit model using current parameter values. We don't need variable
    ### importance measures here and getting them takes time, so set
    ### importance to F
    fit.rf <- randomForest(Y ~ .,
                           data = data, importance = F,
                           mtry = this.mtry, nodesize = this.nodesize
    )
    
    ### Get OOB predictions and MSPE, then store MSPE
    OOB.pred <- predict(fit.rf)
    OOB.MSPE <- get.MSPE(data$Y, OOB.pred)
    
    OOB.MSPEs[j, i] <- OOB.MSPE # Be careful with indices for OOB.MSPEs
  }
}


### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is mtry-nodesize
names.pars <- paste0(
  all.pars$mtry, "-",
  all.pars$nodesize
)
colnames(OOB.MSPEs) <- names.pars

### Make boxplot
boxplot(OOB.MSPEs, las = 2, main = "MSPE Boxplot")


### Get relative MSPEs and make boxplot
OOB.RMSPEs <- apply(OOB.MSPEs, 1, function(W) W / min(W))
OOB.RMSPEs <- t(OOB.RMSPEs)
boxplot(OOB.RMSPEs, las = 2, main = "RMSPE Boxplot")

### Zoom in on the competitive models
boxplot(OOB.RMSPEs, las = 2, main = "RMSPE Boxplot", ylim = c(1, 1.02))

test_predictors <- read.table("test_predictors.csv",
                              header = TRUE, sep = ",", na.strings = " "
)
dim(test_predictors)
fit.rf.2 <- randomForest(Y ~ .,
                         data = data, importance = T,
                         mtry = 11, nodesize = 4
)


prediction_rf_11_4 <- predict(fit.rf.2, test_predictors)

write.table(prediction_rf_11_4,
            "prediction_rf_11_4", sep = ",", row.names = FALSE, col.names = FALSE)