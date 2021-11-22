library(class)
library(ggplot2)
library(fastDummies)
library(tidyr)
library(dplyr)
library(ggcorrplot)
library(sampling)
library(pROC)
library(zoo)
library(forecast)

### FUNCTION DEFINITIONS ###
# ALL PARAMETERS ARE NOT PASSED BY REFERENCE BUT ARE A COPY IN FUNCTIONS

## FUNCTIONS FOR DATASET AND EDA ##
# First preparation of the dataset
prepare_dataset <- function(df) {
  # convert drafted to integer
  df$drafted[df$drafted == "False"] <- 0
  df$drafted[df$drafted == "True"] <- 1
  # round games_played and games_started and height, weight
  df$games_played <- round(df$games_played, 3)
  df$games_started <- round(df$games_started, 3)
  df$height <- round(df$height, 3)
  df$weight <- round(df$weight, 3)
  # calculate wrong three points percentage and remove na TEMPORARY IF SCRAPED
  df$three_pct <- round(df$three_ptrs/df$three_pattmp, 3)
  df[is.na(df)] <- 0
  df$weight <- round(df$three_pct, 3)
  # make dummy for categorical position
  df$drafted <- as.integer(df$drafted)
  df <- dummy_cols(df, select_columns = 'position')
  # remove useless columns
  drop <- c("name","school","start_season","position")
  # move drafted to last column
  df = df %>% relocate(drafted, .after = last_col())
  df = df[,!(names(df) %in% drop)]
  
  #return dataframe
  return(df)
}

# Full dataset preparation for classification
full_dataset_for_classification <- function(df) {
  # delete a dummy variable
  df =  df %>% select(-position_Center)
  # n features before (not counting end_season):
  print(ncol(df))
  # remove total rebound since it makes the matrix singular (total reb = def + off reb)
  df =  df %>% select(-total_reb)
  # (ALREADY TESTED) Removing total points covariates because they cause linear relation
  df =  df %>% select(-c(field_goal, field_attmps, field_pct))
  # Remove throws that scored points and keep percentage and total throws because collinearity
  df =  df %>% select(-c(two_pattamps, three_pattmp, free_attmps))
  # remove points because linear combination of the other point throws
  df = df %>% select(-c(points))
  # remove weight and seasons variable because consideration on the correlation mtx and boxplots
  df =  df %>% select(-c(weight, seasons))
  # n features after (not counting end_season):
  print(ncol(df))
  # IT KEEPS END_SEASON FOR TRAIN TEST SPLIT
  return(df)
}

# Split dataset in train / test based on end_season
train_test_split <- function(df, split_season){
  # players per year
  print(table(df$end_season))
  # drafted per year
  table(df$end_season,df$drafted)
  #total players
  nrow(df)
  # test candidate:
  ratio = (1420+1450)/15633 * 100
  print(ratio)
  
  # make the split
  index_test  = which(df$end_season > split_season)
  length(index_test)
  #index_test
  test <- df[index_test, ]
  train <- df[-index_test, ]
  
  #remove end_season from both 
  # remove the end season column
  train =  train %>% select(-end_season)
  test = test %>% select(-end_season)
  
  # return both train and test
  results <- list(train, test)
  head(results$train)
  return(results)
}

# CORRELATION MATRIX
full_cor_matrix <- function(df) {
  mtx = cor(df)
  print("full correlation matrix:")
  print(mtx)
  print("correlation for drafted:")
  print(mtx[, "drafted"])
  # correlation matrix graph
  ggcorrplot(mtx, hc.order = TRUE, outline.col = "white", lab=TRUE)
  
  
}

# PLOT BOXPLOT WITH DRAFTED
boxplot_with_drafted <- function(df) {
  df<-as.data.frame(df)
  par(mfrow=c(6,4))
  # 1:19 lascia fuori drafted e le due dummy
  for (i in 1:18) {
    boxplot(df[,i] ~ df$drafted, main = names(df[i]), xlab = ' ', ylab = ' ',
            col = c('red', 'green'))
  }
  par(mfrow=c(1,1))
}


## FUNCTIONS FOR MODELS ##

# Baseline classifiers performance
model_baseline_report <- function(test) {
  set.seed(42)
  # a complete random classifier random
  pred.random = sample(c(0,1), replace=TRUE, size=nrow(test))
  # a classifier that only sets 0
  pred.zeros = sample(c(0), replace=TRUE, size=nrow(test))
  print("performance for Random Classifier")
  print(report_confusion_matrix(test$drafted, pred.random))
  print(report_all_metrics(test$drafted, pred.random))
  # NON VA LA CONFUSION MATRIX SICCOME NON SONO PREDETTI 1
  # print("performance for Zeros Classifier")
  # print(report_confusion_matrix(test$drafted, pred.zeros))
  # print(report_all_metrics(test$drafted, pred.zeros))
}

# GLM
model_glm <- function(train) {
  mod <- glm(drafted~., data = train, family = binomial)
  return(mod)
}
# LDA
model_lda <- function(train) {
  mod <- lda(drafted~., data = train)
  return(mod)
}

# QDA
model_qda <- function(train) {
  mod <- qda(drafted~., data = train)
  return(mod)
}

# KNN cross validation (no function for model since the function requires the test set)
search_best_knn <- function(train, maxk) {
  # finds out the best performant knn from 1 to maxk with a dedicated test set ( no k fold option )
  # Shuffle the rows
  set.seed(42)
  rows <- sample(nrow(train))
  train <- train[rows, ]
  # get a stratified sampling we want 20% as test
  train.samplesize = round(0.20*nrow(train))
  # get sample proportions
  zeros = table(train$drafted)[1]
  ones =table(train$drafted)[2]
  prop.zeros = round(zeros / nrow(train) , 3)
  prop.ones = round(1 - prop.zeros , 3)
  nrows.zeros = round(train.samplesize * prop.zeros , 0)
  nrows.ones = train.samplesize - nrows.zeros
  # now we sample these rows from drafted and non drafted
  sample.drafted <- train %>% filter(drafted == 1) %>% slice(1:nrows.ones)
  sample.notdrafted <- train %>% filter(drafted == 0) %>% slice(1:nrows.zeros)
  sample.test <- rbind(sample.drafted, sample.notdrafted)
  # remove the ids from the training set, anti join defaults to all columns
  train <- anti_join(train, sample.test)
  # now we have our training and stratified test set with respect to the classes
  f1 <- c()
  k.values <- seq(1,maxk,1)
  for (i in k.values) {
    knn.pred <- knn(train[1:12], sample.test[1:12], cl = train$drafted, k = i)
    f1 <- c(f1, report_f1(sample.test$drafted, knn.pred))
  }
  plot(k.values, f1, type = 'l', xlab = 'Number of neighbors k', ylab = 'F1 Score')
  best <- k.values[which.max(f1)]
  points(best, f1[which.max(f1)], col = 'red', pch = 20, cex = 2)
  print(f1)
  print("best k value for F1 Score")
  print(best)
  # return best k  
  return(best)
}

## FUNCTIONS FOR METRICS REPORT ##

# Confusion matrix
report_confusion_matrix <- function(values, pred) {
  mtx <- as.matrix(table(values, pred))
  return (mtx)
}

# accuracy
report_accuracy <- function(values, pred) {
  mtx <- as.matrix(table(values, pred))
  accuracy <- round((mtx[1,1] + mtx[2,2]) / sum(mtx), 3)
  return(accuracy)
}

# precision
report_precision <- function(values, pred) {
  mtx <- as.matrix(table(values, pred))
  precision <-  round(mtx[2, 2] / sum(mtx[, 2]), 3)
  return(precision)
}

# recall
report_recall <- function(values, pred) {
  mtx <- as.matrix(table(values, pred))
  recall = round(mtx[2,2] / sum(mtx[2,]), 3)
  return(recall)
}

# specificity 
report_specificity <- function(values, pred) {
  mtx <- as.matrix(table(values, pred))
  specificity = round(mtx[1,1] / sum(mtx[1, ]), 3)
  return(specificity)
}

# f1_score
report_f1 <- function(values, pred) {
  precision = report_precision(values, pred)
  recall = report_recall(values, pred)
  f1_score = round(2*((precision*recall)/(precision+recall)), 3)
  return(f1_score)
}

# balanced_accuracy
report_balanced_accuracy <- function(values, pred) {
  recall = report_recall(values, pred)
  specificity = report_specificity(values, pred)
  balanced_accuracy = round((recall+specificity)/2, 3)
  return(balanced_accuracy)
}

# table with other metrics
report_all_metrics <- function(values, pred) {
  mtx <- as.matrix(table(values, pred))
  accuracy = round((mtx[1,1] + mtx[2,2]) / sum(mtx), 3)
  precision =  round(mtx[2, 2] / sum(mtx[, 2]), 3)
  recall = round(mtx[2,2] / sum(mtx[2,]), 3) # also called sensitivity
  specificity = round(mtx[1,1] / sum(mtx[1, ]), 3)
  f1_score = round(2*((precision*recall)/(precision+recall)), 3)
  # balanced accuracy is (sensitivity + specificity)/2
  balanced_accuracy = round((recall+specificity)/2, 3)
  # build dataframe
  metrics.data <- data.frame(accuracy, balanced_accuracy,
                             precision, recall, f1_score)
  # return dataframe
  return(metrics.data)
}

# table with value and graph of auc 
report_auc <- function(values, pred) {
  metrics.roc <- roc(values, pred, levels=c('1','0'))
  # plotting
  plot(metrics.roc, print.auc = TRUE, legacy.axes = TRUE, xlab = 'False positive rate',
       ylab = 'True positive rate')
  # print best threshold
  print(coords(metrics.roc, 'best'))
  # return the auc score
  metrics.auc = auc(metrics.roc)
  return(metrics.auc)
}

## FUNCTIONS FEATURE SELECTION ##

### MAIN ###
df <- read.csv("dataset_final.csv")
head(df)

df = prepare_dataset(df)
df = full_dataset_for_classification(df)
results = train_test_split(df,  2018)
train = results[[1]]
test = results[[2]]
head(train)
head(test)







