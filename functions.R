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
library(bestglm)
library(Matrix)
library(glmnet)
library(tsutils)

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
  # make dummy for categorical position
  df$drafted <- as.integer(df$drafted)
  df <- dummy_cols(df, select_columns = 'position')
  # remove useless columns
  df =  df %>% select(-c(name, start_season, position))
  #drop <- c("name","start_season","position")
  # move drafted to last column
  df = df %>% relocate(drafted, .after = last_col())
  #df = df[,!(names(df) %in% drop)]
  
  #return dataframe
  return(df)
}

# Full dataset preparation for classification
full_dataset_for_classification <- function(df) {
  # delete a dummy variable WE DECIDED TO KEEP 3 DUMMIES (we can't because multicollinearity)
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
  df =  df %>% select(-c(seasons))
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
  ratio = (1437+1474)/15797 * 100
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
  print("performance for Zeros Classifier")
  print(report_confusion_matrix(test$drafted, pred.zeros))
  print(report_all_metrics(test$drafted, pred.zeros))
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
    knn.pred <- knn(train[1:18], sample.test[1:18], cl = train$drafted, k = i)
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
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  return (mtx)
}

# accuracy
report_accuracy <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  accuracy <- round((mtx[1,1] + mtx[2,2]) / sum(mtx), 3)
  return(accuracy)
}

# precision
report_precision <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  precision <-  round(mtx[2, 2] / sum(mtx[, 2]), 3)
  return(precision)
}

# recall
report_recall <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
  recall = round(mtx[2,2] / sum(mtx[2,]), 3)
  return(recall)
}

# specificity 
report_specificity <- function(values, pred) {
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
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
  mtx <- as.matrix(table(factor(values,levels = 0:1) , factor(pred, levels =0:1)))
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
  metrics.roc <- roc(values, pred, levels=c(0,1))
  print(metrics.roc)
  # plotting
  plot(metrics.roc, print.auc = TRUE, legacy.axes = TRUE, xlab = 'False positive rate',
       ylab = 'True positive rate', col = 'red')
  # print best threshold
  print(coords(metrics.roc, 'best'))
  # return the auc score
  metrics.auc <- auc(metrics.roc)
  return(metrics.roc)
}

## FUNCTIONS FEATURE SELECTION ##

# AIC FEATURE SELECTION
select_glm_AIC <- function(train) {
  # TESTED : IT WORKS AS INTENDED
  # glm.AIC <- bestglm(train, IC = 'AIC', method = "backward", family=binomial) NOME SOLO EXAUSTED CON BINOMIAL
  # intializing starting AIC (full model) and covariate list
  aics.best <- c(AIC(glm(drafted~., data = train, family = binomial)))
  columns.kept <- c(names(train[-19]))
  columns.deleted <- c()
  while (TRUE) {
    aics.loop <- c()
    columns.loop <- c()
    for (i in columns.kept) {
      # get all the aics for removing one of the remaining column
      aics.loop <- c (aics.loop, 
                      AIC(glm(drafted~., data = train[,!(names(train) %in% c(columns.deleted, i))], family = binomial)))
      columns.loop <- c(columns.loop, i)
    }
    # break if no improvements
    if(min(aics.loop) > tail(aics.best,1)) {
      break
    }
    # add the best removal to the removed list, this gives the index for columns.loop
    index = which(aics.loop == min(aics.loop))[1]
    columns.deleted <- c(columns.deleted, columns.loop[index])
    # remove the column removed from the remaining columns list
    columns.kept <- columns.kept[! columns.kept %in% c(columns.loop[index])]
    # add the new AIC value to the best list
    aics.best <- c(aics.best, min(aics.loop))
    # break if we removed all the columns
    if (length(columns.kept) == 0) {
      break
    }
  }
  # return the columns kept for the moment
  print(columns.deleted)
  print(aics.best)
  return(columns.kept)
}

# BIC FEATURE SELECTION
select_glm_BIC <- function(train) {
  # TESTED : IT WORKS AS INTENDED
  # glm.AIC <- bestglm(train, IC = 'AIC', method = "backward", family=binomial) NOME SOLO EXAUSTED CON BINOMIAL
  # intializing starting AIC (full model) and covariate list
  bics.best <- c(BIC(glm(drafted~., data = train, family = binomial)))
  columns.kept <- c(names(train[-19]))
  columns.deleted <- c()
  while (TRUE) {
    bics.loop <- c()
    columns.loop <- c()
    for (i in columns.kept) {
      # get all the bics for removing one of the remaining column
      bics.loop <- c (bics.loop, 
                      BIC(glm(drafted~., data = train[,!(names(train) %in% c(columns.deleted, i))], family = binomial)))
      columns.loop <- c(columns.loop, i)
    }
    # break if no improvements
    if(min(bics.loop) > tail(bics.best,1)) {
      break
    }
    # add the best removal to the removed list, this gives the index for columns.loop
    index = which(bics.loop == min(bics.loop))[1]
    columns.deleted <- c(columns.deleted, columns.loop[index])
    # remove the column removed from the remaining columns list
    columns.kept <- columns.kept[! columns.kept %in% c(columns.loop[index])]
    # add the new AIC value to the best list
    bics.best <- c(bics.best, min(bics.loop))
    # break if we removed all the columns
    if (length(columns.kept) == 0) {
      break
    }
  }
  # return the columns kept for the moment
  print(columns.deleted)
  print(bics.best)
  return(columns.kept)
  
}

# LASSO FEATURE SELECTION
select_glm_lasso <- function(train) {
  set.seed(42)
  # grid <- 10ˆseq(7, -5, length = 100)
  # Using lambdaseq to find list of lambdas, the max value sets all coefficients to 0
  x.seq <- as.matrix(train[,1:18])
  y.seq <- train[,19]
  lambda.seq <- lambdaseq(x.seq, y.seq)$lambda
  X.vars <- model.matrix(train$drafted~. , train)[,-1] #removes column of ones
  Y.vars <- train$drafted
  # TESTING FOR WEIGHTED CLASSIFICATION (not producing better selection so scrapped)
  # fraction_0 <- rep(1 - sum(Y.vars == 0) / length(Y.vars), sum(Y.vars == 0))
  # fraction_1 <- rep(1 - sum(Y.vars == 1) /length(Y.vars), sum(Y.vars == 1))
  # weights <- length(Y.vars)
  # weights[Y.vars == 0] <- fraction_0
  # weights[Y.vars == 1] <- fraction_1
  # glm.lasso <- cv.glmnet(X.vars, Y.vars, lambda = lambda.seq, weights = weights, type.measure = 'class', family ='binomial')
  # nfolds 10 default
  mod.glm.cv.lasso <- cv.glmnet(X.vars, Y.vars, lambda = lambda.seq, type.measure = 'class', family ='binomial')
  # print results
  print(mod.glm.cv.lasso$cvm)
  # get best lambda and plot results
  best.lambda <- mod.glm.cv.lasso$lambda.min
  print(best.lambda)
  plot(mod.glm.cv.lasso)
  # train model and get coefficients
  mod.glm.lasso <- glmnet(X.vars, Y.vars, alpha = 1, lambda = best.lambda, family = 'binomial')
  print(coef(mod.glm.lasso))
  # TEST should set all to 0: WORKING
  # glm.model.lasso <- glmnet(X.vars, Y.vars, alpha = 1, lambda = lambda.seq[1], family = 'binomial')
  # coef(glm.model.lasso)
  
  return(mod.glm.lasso)
  }


### MAIN ###
df <- read.csv("dataset_finale.csv")
head(df)

df = prepare_dataset(df)
df = full_dataset_for_classification(df)
results = train_test_split(df,  2018)
train = results[[1]]
test = results[[2]]
head(train)
head(test)

# test prediction and report with glm threshold 0.5
mod.glm <- glm(drafted~., data = train, family = binomial)
summary(mod.glm)
pred.glm <- predict(mod.glm, test, type= "response")
report_auc(test$drafted, pred.glm)
tr <- 0.5
pred.glm[pred.glm >= tr] <- 1
pred.glm[pred.glm < tr] <- 0
report_all_metrics(test$drafted, pred.glm)
report_confusion_matrix(test$drafted, pred.glm)

# Get lasso
set.seed(42)
x.seq <- as.matrix(train[,1:18])
y.seq <- train[,19]
lambda.seq <- lambdaseq(x.seq, y.seq)$lambda
X.vars <- model.matrix(train$drafted~. , train)[,-1] #removes column of ones
Y.vars <- train$drafted
mod.glm.cv.lasso <- cv.glmnet(X.vars, Y.vars, lambda = lambda.seq, type.measure = 'class', family ='binomial')
mod.glm.cv.lasso$cvm
best.lambda <- mod.glm.cv.lasso$lambda.min
plot(mod.glm.cv.lasso)
mod.glm.lasso <- glmnet(X.vars, Y.vars, alpha = 1, lambda = best.lambda, family = 'binomial')
coef(mod.glm.lasso)
# find threshold
X.vars.test <- model.matrix(test$drafted~. , test)[,-1]
# Deprecated use a matrix as predictor. Unexpected results may be produced, please pass a numeric vector.
pred.glm.lasso <- data.frame(round(predict(mod.glm.lasso, x.test, type= "response"), 4))
#pred.glm.lasso <- predict(mod.glm.lasso, x.test, type= "response")
mod.glm.lasso.roc <- report_auc(test$drafted, pred.glm.lasso)
coords(mod.glm.lasso.roc, 'best')
tr.lasso <- coords(mod.glm.lasso.roc, 'best')$threshold
pred.glm[pred.glm >= tr] <- 1
pred.glm[pred.glm < tr] <- 0
report_all_metrics(test$drafted, pred.glm.lasso)
report_confusion_matrix(test$drafted, pred.glm.lasso)

### ORDER OF REMOVAL BACKWARD STEPWISE SELECTION ###
# mod.glm <- glm(drafted~.-free_pct, data = train, family = binomial)
# summary(mod.glm)
# 1. free pct 0.830121 
# 2. off_reb 0.364819
# 3. games_started 0.300908
# 4. two_pct 0.046028
# 5. three_pct 0.044832
# 6. def_reb 0.017083
# END, all 3 *

mod.glm=select_glm_lasso(train)





