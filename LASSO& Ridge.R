library(glmnet)
library(ggplot2)
library(caret)

# Load the data
data <- read.csv("training_data.csv")

folds <- floor((sample.int(nrow(data)) - 1) * V / nrow(data)) + 1

MSPEs.cv <- matrix(NA, nrow = V, ncol = 5)
colnames(MSPEs.cv) <- c("LS", "Stepwise", "ridge", "LASSO-min", "LASSO-1SE")
get.MSPE <- function(Y, Y.hat) {
  return(mean((Y - Y.hat)^2))
}
set.seed(2928893)
# Let's do 5-fold CV
V <- 10
# Abbreviating sample.int function arguments
folds <- floor((sample.int(nrow(data)) - 1) * V / nrow(data)) + 1

MSPEs.cv <- matrix(NA, nrow = V, ncol = 4)
colnames(MSPEs.cv) <- c("ridge", "LASSO-min", "LASSO-1SE", "PLS")

ncomp = c()


for (v in 1:V) {
  data.train <- data[folds != v, ]
  data.valid <- data[folds == v, ]
  n.train <- nrow(data.train)
  
  ### Get response vector
  Y.valid <- data.valid$Y
  
  ridge1 <- lm.ridge(Y ~ ., lambda = seq(0, 100, .05), 
                     data = data[folds != v, ])
  coef.ri.best1 <- coef(ridge1)[which.min(ridge1$GCV), ]
  pred.ri1 <- as.matrix(cbind(1, data[folds == v, 2:6])) %*% coef.ri.best1
  MSPEs.cv[v, 1] <- mean((data[folds == v, "Y"] - pred.ri1)^2)
  
  #Y.train <- data.train$Y
  # Y.valid <- data.valid$Y
  
  y.1 <- data[folds != v, 1]
  x.1 <- as.matrix(data[folds != v, c(2:6)])
  y.2 <- data[folds == v, 1]
  x.2 <- as.matrix(data[folds == v, c(2:6)])
  
  cv.lasso.1 <- cv.glmnet(y = y.1, x = x.1, family = "gaussian")
  # Predict both halves using first-half fit
  pred.las1.min <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  pred.las1.1se <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  MSPEs.cv[v, 2] <- mean((y.2 - pred.las1.min)^2)
  MSPEs.cv[v, 3] <- mean((y.2 - pred.las1.1se)^2)
  
  
  library(pls)
  mod.pls <- plsr(Y ~ ., data = data[folds != v, ], ncomp = 5, validation = "CV")
  #mp.cv = mod.pls$validation
  #Opt.Comps = which.min(sqrt(mp.cv$PRESS/nrow(data)))
  #pred.pls <- predict(mod.pls, data[folds == v, ], ncomp = Opt.Comps)
  CV_pls <- mod.pls$validation
  pls_comps <- CV_pls$PRESS
  n_comps <- which.min(pls_comps)
  ncomp[v] <- n_comps
  
  pred.pls <- predict(mod.pls, data[folds == v, ], ncomp = n_comps)
  MSPE.pls <- get.MSPE(Y.valid, pred.pls)
  MSPEs.cv[v, 4] <- MSPE.pls
  
  
  #ncomp <- c(ncomp,Opt.Comps)
  
  
}

ncomp
MSPEs.cv

(MSPEcv <- apply(X = MSPEs.cv, MARGIN = 2, FUN = mean))

boxplot(MSPEs.cv,
        las = 2,
        main = "MSPE \n Cross-Validation"
)

# Relative MSPE: divided by minimum error for each rep

low.c <- apply(MSPEs.cv, 1, min)
boxplot(MSPEs.cv / low.c,
        las = 2,
        main = "Relative MSPE \n Cross-Validation"
)


# Limit range to see comparison of two good models

boxplot(MSPEs.cv / low.c,
        las = 2, ylim = c(1, 1.5),
        main = "Focused Relative MSPE \n Cross-Validation"
)