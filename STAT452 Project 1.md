# STAT452 Project 1

## Name Jackson Yuan Student ID:301387501

### Abstract(What models did I try to use for this problem)

This is a report that I used several models to predict Y of test data by analyze from the given training data. I choose **LS,PLS,Ridge& LASSO, Random Forest, and Boosting** to get MSPE to decide which model is the best model to predict Y of test data. 



First, I read the csv file and check the dimension of the training data.

```R
data <- read.table("training_data.csv",header = TRUE, sep = ",", na.strings = " ")
dim(data)
```

**How did I evaluate and compare models.**

Then I start from the LS and PLS Method and LASSO and Ridge to **compare which model give me a lower MSPE**

It give the minimum MSPE for 25.5-26 after compiling the **LS and PLS** and **LASSO and ridge** method.

<img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/1701378598883.png" alt="1701378598883" style="zoom: 33%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/81d06cdc3fcdf89561ae9e70fe44e99.png" alt="81d06cdc3fcdf89561ae9e70fe44e99" style="zoom: 33%;" />

But after using Boosting and Random Forest at some initial value, I decide to compared them to choose one as my best model. Because it give me a less MSPE compared to LASSO& Ridge, LS, PLS. The parameters as the code show:

```R
##Boosting method
set.seed(12345678)
max.trees <- 10000
all.shrink <- c(0.001, 0.01, 0.1)#I choose 0.001,0.01 and 0.1 as my first guess for shrink
all.depth <- c(1,2,3)#I start from 1 to 3 as my depth
al1.pars <- expand.grid(shrink = al1.shrink, depth = all.depth)
n.pars <- nrow(a11.pars)
##RandomForest
all.mtry <- 1:4 #I start from 1 to 4 as try parameter
all.nodesize <- c(2,5,8)#I choose 2,5,8 as node size
```

<img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/f9b1ed8575b88ac790bc96b4a75a3d5.png" alt="f9b1ed8575b88ac790bc96b4a75a3d5" style="zoom: 50%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/ab9e85c030811d658f7317d8b31899d.png" alt="ab9e85c030811d658f7317d8b31899d" style="zoom: 33%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/bad3d93ce8e493ab123bb5361490ca1.png" alt="bad3d93ce8e493ab123bb5361490ca1" style="zoom: 33%;" />

**Boosting and Random Forest has tuning parameter, how do I choose those parameter.**

Boosting and Random Forest shows much less MSPE than the other four models., especially Random Forest gives as value 19 as low MSPE. And `I found that shrinkage goes from 0.001 to 0.1, and MSPE decreases as shrinkage gets larger`. `From 1 to 4, MSPE also decreases as depth grows`. I find that the MSPE of Random Forest model at the tuning parameter of 1 has much difference from the other three so I use the second round by reduce some parameter. **And I consider x18, x21, x20, x8, x12, x1, x5, x13, x11 is important predictor as Random Forest give a apparent result.**(9 true predictors)

```r
##Boosting method
all.shrink <- c(0.001, 0.01, 0.1)#Hold 0.001,0.01 and 0.1
all.depth <- c(4,5,6,7,8)#continue increase depth
##RandomForest
all.mtry <- c(2,3,4) #I start from 1 to 4 as try parameter
all.nodesize <- c(5,8)#I choose 2,5,8 as node size
```

<img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/29411622f53abc9f37e6ce94ed18e90.png" alt="29411622f53abc9f37e6ce94ed18e90" style="zoom:50%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/4ab68b94786ce5470a9fb91e1848d5b.png" alt="4ab68b94786ce5470a9fb91e1848d5b" style="zoom:33%;" />

After I increase depth parameter, **Boosting model** give decrease MSPE and we can find MSPE continue decreasing as depth parameter increase. **Random Forest** give the better MSPE than before and it apparently lower when try parameter increase and node size is lower when it is 5 than 8. So I test value of try parameter greater than 8 and node size is 4,5 to see whether the model give me a lower parameter. And I change the range of shrinkage and continue to increase the value of depth to see new MSPE.

```R
##Boosting method
all.shrink <- c(0.015,0.1,0.2)#Change the range of shrinkage
all.depth <- c(8,9,10,11)#continue increase depth
## Random Forest
all.mtry <- c(9,10,11,12) 
all.nodesize <- c(4,5)
```



<img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/a3dc5dcc27ad8efef6c1f11ec36184f.png" alt="a3dc5dcc27ad8efef6c1f11ec36184f" style="zoom:50%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/831dd60a46f1bc1c2a4dfbc5a092281.png" alt="831dd60a46f1bc1c2a4dfbc5a092281" style="zoom: 33%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/4fe9d63e3c8ac6cda4f91ca14555644.png" alt="4fe9d63e3c8ac6cda4f91ca14555644" style="zoom:50%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/a3bda850d1e57486eab60aae20a7144.png" alt="a3bda850d1e57486eab60aae20a7144" style="zoom: 33%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/c1944a9ae161964d0e03a0100a601cd.png" alt="c1944a9ae161964d0e03a0100a601cd" style="zoom:50%;" /><img src="C:/WeChat%20Files/wxid_8466534665612/FileStorage/Temp/1701405561528.png" alt="1701405561528" style="zoom: 33%;" /><img src="C:/Users/11358/OneDrive/%E6%A1%8C%E9%9D%A2/%E7%AC%94%E8%AE%B0/screenshot/1701406336068.png" alt="1701406336068" style="zoom:50%;" />

After test different sets of tuning parameter by binary search concept, **I consider Random Forest has lowest MSPE with tuning parameter as try parameter is 11 and node size is 4**. And estimate of the number of true predictors is on the last picture according to GAM model.

