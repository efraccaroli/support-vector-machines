# OCR with support vector machines
# created on 8/30/14
# SVMs can be used for classification/numeric prediction

setwd("~/Dropbox/Development/packt/chapter 7")

letters <- read.csv("letterdata.csv")

library(caret)
intrain <- createDataPartition(letters$letter, p = .8, list = FALSE)
letters_train <- letters[intrain, ]
letters_test <- letters[-intrain, ]

library(kernlab)
# train model
model <- ksvm(letter ~., data = letters_train) # kernel refers to dimension conversion
# training error aka in sample = 5.2%

# evaluate model
pred <- predict(model, letters_test)
table(pred, letters_test$letter)

# calculate out of sample error rate
match <- pred == letters_test$letter
table(match)

# out of sample error rate is 6.3%
