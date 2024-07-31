
wine = read.csv("Wine_Quality.xlsx") 
train_sub = sample(nrow(wine),7/10*nrow(wine))
train_data = wine[train_sub,]
test_data = wine[-train_sub,]
install.packages('e1071')
ibrary(pROC) 
library(e1071)
train_data$quality = as.factor(train_data$quality)
test_data$quality = as.factor(test_data$quality)

wine_svm<- svm(quality ~  fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides
               +free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, 
               data = train_data,
               type = 'C',kernel = 'radial' )
