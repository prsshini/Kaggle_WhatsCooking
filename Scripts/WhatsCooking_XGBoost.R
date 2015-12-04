
#- Model 4: xgboost (single model for all cuisine types)
#- prepare the spare matrix (note: feature index in xgboost starts from 0)
xgbmat     <- xgb.DMatrix(Matrix(data.matrix(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])), label=as.numeric(dtm_train$cuisine)-1)

#- train our multiclass classification model using softmax
xgb        <- xgboost(xgbmat, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20)

#- predict on the SUBMIT set and change cuisine back to string
xgb.submit      <- predict(xgb, newdata = data.matrix(dtm_submit[, !colnames(dtm_submit) %in% c("cuisine")]))
xgb.submit.text <- levels(dtm_train$cuisine)[xgb.submit+1]

#- load sample submission file to use as a template
sample_sub <- read.csv('../input/sample_submission.csv')

#- build and write the submission file
submit_match   <- cbind(as.data.frame(submit_raw$id), as.data.frame(xgb.submit.text))
colnames(submit_match) <- c("id", "cuisine")
submit_match   <- data.table(submit_match, key="id")
submit_cuisine <- submit_match[id==sample_sub$id, as.matrix(cuisine)]

submission <- data.frame(id = sample_sub$id, cuisine = submit_cuisine)
write.csv(submission, file = 'xgboost_multiclass.csv', row.names=F, quote=F)

# plot the most important features
names <- colnames(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:30,])

