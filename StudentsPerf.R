library(tidyverse)
library(fastDummies)
library(randomForest)
library(xgboost)
library(caret)

data <- read.csv("StudentsPerformance.csv", header = TRUE)


widedata <- fastDummies::dummy_cols(data)
widedata <- widedata[6:25]
widedata <- widedata[c(-4, -8, -11, -17, -19)]
names(widedata) <- c("math_score", "reading_score", "writing_score", "gender_male", "race_ethnicity_group_B", 
                     "race_ethnicity_group_C", "race_ethnicity_group_D", "race_ethnicity_group_E", "parental_level_of_education_some_college",
                     "parental_level_of_education_master_degree", "parental_level_of_education_associate_degree", "parental_level_of_education_high_school",       
                     "parental_level_of_education_some_high_school", "lunch_free_or_reduced", "test_preparation_course_completed")

widedata <- widedata %>%
  mutate(lunch_free_or_reduced= as.factor(lunch_free_or_reduced))

train <- widedata[1:800, ]
test <- widedata[801:1000, c(1:13,15)]
testlabel <- widedata[801:1000, 14]

# Classification Problem
model <- randomForest::randomForest(lunch_free_or_reduced~., data=train, xtest=test, ytest=testlabel)


abc <- as.data.frame(model$test$votes)
pred <- ifelse (abc$`1` > 0.36,1,0)
pred<- as.factor(as.vector(pred))
#pred <- model$test$predicted
conf.matrix <- confusionMatrix(pred, testlabel)
conf.matrix

widedata <- widedata %>%
  mutate(lunch_free_or_reduced= as.numeric(as.character(lunch_free_or_reduced)))

train <- as.matrix(widedata[1:800, c(1:13,15)])
train.label <- widedata[1:800, 14]
test<- as.matrix(widedata[801:1000, c(1:13,15)])
test.label <- widedata[801:1000, 14]

dtrain <- xgb.DMatrix(data = train, label = train.label)
model.xgb <- xgboost(data = dtrain, max.depth = 6, eta = .4, nthread = 2, nrounds = 2, objective = "binary:logistic")

pred.test<- as.data.frame(predict(model.xgb, test))
pred.train<- as.data.frame(predict(model.xgb, train))

msa.now.matrix

