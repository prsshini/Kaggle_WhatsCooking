
rm(list=ls())

install.packages("e1071")

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

#setwd("C:/R Learning/Titanic/Kaggle Comps/Yummly")
setwd("C:/R Learning/Titanic/Kaggle Comps/Yummly")

data <- fromJSON("train.json/train.json", flatten = TRUE)
testdata <- fromJSON("test.json/test.json", flatten = TRUE)

# Form full data
t1 <- testdata
t1[,"cuisine"] <- NA # this puts a dummy cuisine column
fulldata <- as.data.frame(rbind(data,t1))

# Preprocess full data
ingredients <- Corpus(VectorSource(fulldata$ingredients))
ingredients <- tm_map(ingredients, stemDocument)
ingredientsDTM <- DocumentTermMatrix(ingredients)
sparse <- removeSparseTerms(ingredientsDTM, 0.99)
ingredientsDTM <- as.data.frame(as.matrix(sparse))
ingredientsDTM$cuisine <- as.factor(fulldata$cuisine)
targetvar <- "cuisine"

# Form the realtest and train parts now
ntrain<- NROW(data)
ntest <- NROW(testdata)
nfull <- ntrain + ntest
# Line below: the parantheses around ntrain+1 is important
realtest <- as.data.frame(ingredientsDTM[(ntrain+1):nfull,] )
realtest$cuisine <- NULL # this removes the dummy cuisine column

ingredientsDTM <- as.data.frame(ingredientsDTM[1:ntrain,])

# Set up Formula
xpart <- " . "
ypart <- paste(targetvar," ~ ")
Formula <- as.formula(paste(ypart,xpart))

# Tree model
fit <- rpart(Formula, data = ingredientsDTM, method = "class",
             control=rpart.control(minsplit=2,cp=0.0001))

cols<-colnames(ingredientsDTM)
cols<-sub(".","-",cols)
colnames(ingredientsDTM[1])<-cols


rf<-randomForest(Formula, data=ingredientsDTM, mtry=2, importance = TRUE, do.trace = 100)



 
cols<-colnames(realtest)
cols<-sub("-",".",cols)
colnames(realtest)<-cols


rfpredict<- predict(rf,realtest,"class")

# Create submission file
submission <- matrix(nrow=ntest, ncol=2)
submission[,1] <- testdata$id
submission[,2] <- as.character(paste(rfpredict))

write.csv(submission,file="submission with RF.csv")


