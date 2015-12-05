##CLean Start


rm(list=ls())

library(e1071)
library(SnowballC)
library(jsonlite)
library(dplyr)
library(ggplot2)
library(tm) # For NLP; creating bag-of-words
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(rattle)
library(RColorBrewer)


setwd("C:/R Learning/WhatsCooking/Scripts")

## saveWC.RData file contains both the train and test set. 
#Data from Json files were preprocessed to make it workable with R Programs.

load(file="../data/saveWC.RData")

seed <- 2097865
ntrials <- 1
splitprob <- 2/3
targetvar="cuisine"

accuracy <- vector(mode = "numeric",length = ntrials+1)


  for (trial in 1:ntrials)
    {
          writeLines(paste("Trial number: ",trial,"..."))
      # Form a random partition
      train.index <- createDataPartition(y = dataT$cuisine, p = splitprob, list = FALSE)
       train <- dataT[ train.index,]
        test  <- dataT[-train.index,]
    
          # Set up Formula
          xpart <- " . "
          ypart <- paste(targetvar," ~ ")
          Formula <- as.formula(paste(ypart,xpart))
          
          ##Form a DT    
          fit <- rpart(Formula, data = train, method = "class",
                                  control=rpart.control(minsplit=2,cp=0.0001))
                
          ##Predict the test using the model
          rpartpredict<- predict(fit,test,"class") ## Accuracy 60%
          rpartcm <-confusionMatrix(rpartpredict, test$cuisine)
            
          accuracy[trial]<-rpartcm$overall[[1]]
     }
      
  prp(fit)
  
  prp(fit, type=1, extra=4)
  prp(fit)
  printcp(fit) # display the results
  plotcp(fit) # visualize cross-validation results
  summary(fit) # detailed summary of splits
  
  
  
  ma <- as.matrix(accuracy)
  
  
  
 
  ## Apply Qunatile to find 10% 50% and 90% Quantile of the predicted accuracy.
  ma1 <- apply(ma, 2, quantile, probs = c(0.1, 0.5, 0.9)) 
  mat <- t(ma1)
  amin<- mat[1,1]
  a <- mat[1,2]
  amax <- mat[1,3]
  rm(ma1,mat,ma)
  writeLines(paste("Accuracy @ 10% Quantile: ",round((amin*100), digits=2), "%"))
  writeLines(paste("Accuracy @ 50% Quantile: ",round((a*100), digits=2), "%"))
  writeLines(paste("Accuracy @ 90% Quantile: ",round((amax*100), digits=2), "%"))
  