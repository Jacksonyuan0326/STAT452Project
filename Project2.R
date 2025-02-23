### SECTION 1
library(pROC)

# Generate some example data
set.seed(123)
n <- 1000
p <- 10
x <- matrix(rnorm(n * p), ncol = p)
y <- rbinom(n, 1, 0.5)

# Fit a logistic regression model to the data
model <- glm(y ~ x, family = binomial(link="logit"))

# Compute the predicted probabilities
prob <- predict(model, type = "response")

# Compute the ROC curve and the AUC
roc_obj <- roc(y, prob)
auc <- auc(roc_obj)

# Print the AUC value
cat("The AUC value is", auc)


###ROC Curve and AUC
vehdata <- read.csv("vehicle.csv")
head(vehdata)

vehdata$class = factor(vehdata$class, labels=c("2D", "2D","4D","4D" )) 
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
library(nnet)
library(car)

mod.fit = multinom(
  data = set1.rescale, formula = class ~ .,
  trace = TRUE
)


set1.rescale = data.frame(cbind(rescale(set1[, -19], set1[, -19]),
                                class = set1$class))
set2.rescale = data.frame(cbind(rescale(set2[, -19], set1[, -19]),
                                class = set2$class))
summary(set1.rescale[,1:3])
summary(set2.rescale[,1:3])

pred.set1 = predict(mod.fit,
                    newdata = set1.rescale,type = "class")
pred.set2 = predict(mod.fit,newdata = set2.rescale,type = "class")
misclass.set1 = mean(ifelse(pred.set1 == set1$class,yes = 0, no = 1))
misclass.set1
misclass.set2 = mean(ifelse(pred.set2 == set2$class,
                            yes = 0, no = 1
))
misclass.set2
table(set2$class, pred.set2, dnn = c("Obs", "Pred"))

library(glmnet)
logit.fit <- glmnet(
  x = as.matrix(set1.rescale[, -19]),
  y = set1.rescale[, 19], family = "multinomial"
)
logit.cv <- cv.glmnet(
  x = as.matrix(set1.rescale[, 1:18]),
  y = set1.rescale[, 19], family = "multinomial"
)
c <- coef(logit.fit, s = logit.cv$lambda.min)
cmat <- cbind(
  as.matrix(c[[1]], as.matrix(c[[2]]), as.matrix(c[[3]]))
)
round(cmat, 2)
cmat != 0
lascv.pred.train <- predict(
  object = logit.cv, type = "class",
  s = logit.cv$lambda.min,
  newx = as.matrix(set1.rescale[, 1:18])
)
lascv.pred.test <- predict(logit.cv,
                           type = "class",
                           s = logit.cv$lambda.min,
                           newx = as.matrix(set2.rescale[, 1:18])
)
lasso.mis.train <-
  mean(ifelse(lascv.pred.train == set1$class, yes = 0, no = 1))
lasso.mis.train
lasso.mis.test <-
  mean(ifelse(lascv.pred.test == set2$class, yes = 0, no = 1))
lasso.mis.test

library(MASS)
set1s = apply(set1[, -19], 2, scale)
set1s = data.frame(set1s, class = set1$class)
set2s = apply(set2[, -19], 2, scale)
set2s = data.frame(set2s, class = set2$class)
lda.fit.s = lda(data = set1s, class ~ .)
class.col = ifelse(set1$class==1, y = 53, n = ifelse(set1$class== 2, y = 68, n = ifelse(set1$class== 3,y=203,n=464)))
plot(lda.fit.s, col = colors()[class.col])

lda.pred.train = predict(lda.fit.s, newdata = set1s[, -19])$class
lda.pred.test = predict(lda.fit.s, newdata = set2s[, -19])$class
lda.train = mean(ifelse(lda.pred.train == set1$class, yes = 0, no = 1))
lda.train
lda.test = mean(ifelse(lda.pred.test == set2$class, yes = 0, no = 1))
lda.test

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


###SECTION 2

wine = read.csv("Wine_Quality.csv") 
train_sub = sample(nrow(wine),7/10*nrow(wine))
train_data = wine[train_sub,]
test_data = wine[-train_sub,]
library(pROC) 
library(e1071)
train_data$quality = as.factor(train_data$quality)
test_data$quality = as.factor(test_data$quality)

wine_svm<- svm(quality ~  fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides
               +free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, 
               data = train_data,
               type = 'C',kernel = 'radial' )
pre_svm <- predict(wine_svm,newdata = test_data)
obs_p_svm = data.frame(prob=pre_svm,obs=test_data$quality)
table(test_data$quality,pre_svm,dnn=c("Obs","Preds"))
svm_roc <- roc(test_data$quality,as.numeric(pre_svm))
plot(svm_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM-mdoel ROC kernel = radial')
