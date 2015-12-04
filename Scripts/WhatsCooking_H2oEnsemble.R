rm(list=ls())
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

if (! ("SnowballC" %in% rownames(installed.packages()))) { install.packages("SnowballC") }
if (! ("dplyr" %in% rownames(installed.packages()))) { install.packages("dplyr") }
if (! ("ggplot2" %in% rownames(installed.packages()))) { install.packages("ggplot2") }
if (! ("tm" %in% rownames(installed.packages()))) { install.packages("tm") }
if (! ("caret" %in% rownames(installed.packages()))) { install.packages("caret") }
if (! ("rattle" %in% rownames(installed.packages()))) { install.packages("rattle") }
if (! ("pryr" %in% rownames(installed.packages()))) { install.packages("pryr") }

library(jsonlite)
library(SnowballC)
library(dplyr)
library(ggplot2)
library(tm) # For NLP; creating bag-of-words
library(caret)
library(rattle)

#- load data
load(file="saveWC.RData")

#Subha:
#It may be a good idea to store the target columns before removing them since they will be needed later.
test_cuisine <- test$cuisine
dataT_cuisine <- dataT$cuisine

test$cuisine <- NULL # this removes the dummy cuisine column
#Subha, I corrected a spelling error below (from "cusine" to "cuisine")
dataT$cuisine <- NULL

library(h2o)
library(h2oEnsemble)
h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # Clean slate - just in case the cluster was already running

#setwd("C:/R Learning/Kaggle Comps/Yummly")

h2o.glm.1 <- function(..., alpha = 0.1) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
#
#### Choose a subset of from the above learner library
learner <- c("h2o.glm.wrapper",
             "h2o.randomForest.1", "h2o.randomForest.2",
             "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8",
             "h2o.deeplearning.1", "h2o.deeplearning.6", "h2o.deeplearning.7")

metalearner <- "h2o.glm.wrapper"

mtarget <- dataT_cuisine

levs <- levels(as.factor(mtarget))
levs1<- "italian"

for (class in levs1){
  btarget <- sapply(mtarget, function(x) {
    if (x!=class) {cl <- 0} else {cl <- 1}
    cl
  })
  btarget <- as.factor(btarget)
  
  y <- "btarget" # "C1" is the target value of interest
  
  dtm_train[,y] <- as.factor(dtm_train[,y])  
  
  
  
  #dataT <- cbind(dataT,btarget)
  dataT$btarget <- btarget 
  dtm_train <- as.h2o(dataT,"")
  dtm_test <- as.h2o(test,"")
  
  
  #Now use btarget as the binary target, apply H2O ensemble and get the predicted probabilities for "class"
  
 
  x <- setdiff(names(dtm_train), y)
  #
  #For binary classification, the response should be encoded as factor (also known as the [enum](https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html) type in Java).  The user can specify column types in the `h2o.importFile` command, or you can convert the response column as follows:
  #
  #Subha, I suggest doing this "as.factor" things before the h2o format conversion
  
  
  #Train ensemble with new library:
  fit <- h2o.ensemble(x = x, y = y, 
                      training_frame = dtm_train,
                      family = "binomial", 
                      learner = learner, 
                      metalearner = metalearner,
                      cvControl = list(V = 5))
  
  #Generate predictions on the test set:
  pred <- predict(fit, dtm_test) 
  
}


#pred is a list with two frames (these are frames, not data frames): 
#pred$pred (the output of the ensemble) and
#pred$basepred (the output of the base classifiers)
#The third column of pred$pred is P(target==1) of the ensemble
#The columns of pred$basepred are P(target==1) of the base classifiers
#labels:
labels <- as.data.frame(test[,y])[,1]

#Base learner test AUC (for comparison with what ensemble gives)
L <- length(learner)
auc <- sapply(seq(L), function(l) AUC(predictions = as.data.frame(pred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)

#Generate predictions on the test set:
predictions <- as.data.frame(pred$pred)[,3]
#Evaluate the test set performance of ensemble: 
AUC(predictions = predictions , labels = labels)
#What I got in my run was:
#For the base classifiers:
#1    h2o.glm.wrapper 0.6871288
#2 h2o.randomForest.1 0.7818598
#3 h2o.randomForest.2 0.7775610
#4          h2o.gbm.1 0.7816863
#5          h2o.gbm.6 0.7838500
#6          h2o.gbm.8 0.7804483
#7 h2o.deeplearning.1 0.7277392
#8 h2o.deeplearning.6 0.7155204
#9 h2o.deeplearning.7 0.7324792
#
#For the ensemble:    0.7892216

library(pryr)
mem_used()
