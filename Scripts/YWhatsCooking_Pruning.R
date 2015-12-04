
rm(list=ls())

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
fulldata <- rbind(data,t1)

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
realtest <- ingredientsDTM[(ntrain+1):nfull,] 
realtest$cuisine <- NULL # this removes the dummy cuisine column

ingredientsDTM <- ingredientsDTM[1:ntrain,]


seed <- 2097865
ntrials <- 5
splitprob <- 2/3
targetvar = "cuisine"
accuracy <- vector(mode = "numeric",length = ntrials)


train.index <- createDataPartition(y = ingredientsDTM$cuisine, p = splitprob, list = FALSE)
train <- ingredientsDTM[ train.index,]
val  <- ingredientsDTM[-train.index,]
# Set up Formula
xpart <- " . "
ypart <- paste(targetvar," ~ ")
Formula <- as.formula(paste(ypart,xpart))

# Tree model
fit <- rpart(Formula, data = train, method = "class",
             control=rpart.control(minsplit=2,maxdepth=15,cp=0.005))
prp(fit)
  cpvalues <- fit$cptable[,1]
  cprows <- NROW(cpvalues)
  bestrpartaccuracy <-0
  best_prune_accuracy<-0
  bestcp<-0
  bestprunecp<-0
  for(i in 1:cprows)
    {
        pfit <- prune(fit,cpvalues[i] + 1e-6) 
        prunepredict<- predict(pfit,val,"class")
        rpartpredict<- predict(fit,val,"class")
        prunecm <- confusionMatrix(rpartpredict, val$cuisine)
        rpartcm <-confusionMatrix(rpartpredict, val$cuisine)
        prune_accuracy<-prunecm$overall[1]
        rpart_accuracy<-rpartcm$overall[1]
          if (rpart_accuracy > bestrpartaccuracy)
              {
                bestrpartaccuracy <- rpart_accuracy
                bestcp <- cpvalues[i]
          }
        if (prune_accuracy > best_prune_accuracy)
        {
          best_prune_accuracy <- prune_accuracy
          bestprunecp <- cpvalues[i]
        }
    }
      
bestrpartaccuracy
best_prune_accuracy
bestprunecp
bestcp




