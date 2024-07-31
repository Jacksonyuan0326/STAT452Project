setwd("/Users/yang/Documents/course/stat452/2023 fall - Owen G. Ward/project 1 &2/project 2")
vehdata <- read.csv("vehicle.csv")
head(vehdata)

vehdata$class = factor(vehdata$class, labels=c("small", "small","large","large" )) 
summary(vehdata$class)

  

set.seed(46685326, kind="Mersenne-Twister")
perm <- sample(x=nrow(vehdata)) 
set1 <- vehdata[which(perm <= 3*nrow(vehdata)/4), ] 
set2 <- vehdata[which(perm > 3*nrow(vehdata)/4) , ]


rescale <- function(x1, x2) {
  for (col in 1:ncol(x1)) {
    a <- min(x2[, col])
    b <- max(x2[, col])
    x1[, col] <- (x1[, col] - a) / (b - a)
  }
  x1
}

# multinom uses a formula, so need to keep data in data.frame

# Creating training and test X matrices, then scaling them.

set1.rescale <- data.frame(cbind(rescale(set1[, -19], set1[, -19]), 
                                 class = set1$class))
set2.rescale <- data.frame(cbind(rescale(set2[, -19], set1[, -19]), 
                                 class = set2$class))


library(nnet) 

mod.fit <- multinom(
  data = set1.rescale, formula = class ~ .,
  trace = TRUE
)

# Misclassification Errors
pred.class.1 <- predict(mod.fit,newdata = set1.rescale,type = "class")
pred.class.2 <- predict(mod.fit,newdata = set2.rescale,type = "class")

(mul.misclass.train <- mean(ifelse(pred.class.1 == set1$class, yes = 0, no = 1)))
(mul.misclass.test <- mean(ifelse(pred.class.2 == set2$class, yes = 0, no = 1)))
# Test set confusion matrix
table(set2$class, pred.class.2, dnn = c("Obs", "Pred"))


library(caTools)
library(pROC)

test_prob = predict(mod.fit, set2.rescale, type = "probs")
test_prob
par(mfrow=c(1,1))

test_roc = roc(set2$class, test_prob,  smoothed = TRUE)
plot(test_roc)
as.numeric(test_roc$auc)





fit.logistic <- glm(formula = factor(class) ~ ., family = binomial(link="logit"), data = set1.rescale )
test_prob = predict(fit.logistic, set2.rescale, type = "probs")
test_prob
par(mfrow=c(1,1))
test_roc = roc(set2$class, test_prob,  smoothed = TRUE)
plot(test_roc)
as.numeric(test_roc$auc)


