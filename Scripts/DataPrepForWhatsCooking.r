#This script does data preparation for What's Cooking problem.
#It has two advantages:
#1. Unnecessary memory is not needed during the actual model building process.
#2. Unnecessary computation is avoided when making repeated runs.

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

data <- fromJSON("../data/train.json", flatten = TRUE)
testdata <- fromJSON("../data/test.json", flatten = TRUE)

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
targetval <- "Italian"
# Form the realtest and train parts now
ntrain<- NROW(data)
ntest <- NROW(testdata)
nfull <- ntrain + ntest
# Line below: the parantheses around ntrain+1 is important
test <- as.data.frame(ingredientsDTM[(ntrain+1):nfull,] )

data <- as.data.frame(ingredientsDTM[1:ntrain,])
dataT<-data

#http://www.fromthebottomoftheheap.net/2012/04/01/saving-and-loading-r-objects/
save(dataT, test, file = "../data/saveWC.RData")
